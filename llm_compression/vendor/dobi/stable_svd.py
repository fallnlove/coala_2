import torch.nn as nn
import torch

computeSVD_dtype = torch.float32
model_load_dtype = torch.float16
val_epsilon = 1e-10
grad_epsilon = 1e2
K = 10
diff_epsilon = 1e-10
nTaylor = K + 1


class stable_lowrank_SVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rank):
        # print('forward: Custom_lowrank_SVD')
        U, S, V = torch.svd_lowrank(x, q=rank, niter=2)
        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, gU, gS, gV):

        U, S, V = ctx.saved_tensors
        Vh = V.T
        gVh = gV.T

        assert U.dim() >= 2 and Vh.dim() >= 2

        if gS is None and gU is None and gVh is None:
            return None

        if gU is None and gVh is None:
            return U @ torch.diag_embed(gS) @ Vh

        """taylor and set epsilon algorithm
        """
        I = torch.eye(S.shape[0], device=S.device).to(S.device)
        mask_no_I = ~I.bool()
        mask_lower_tri = torch.tril(
            torch.ones(S.size(0), S.size(0), dtype=torch.bool)
        ).to(S.device)  # 只对左下对角线进行运算
        mask_init = mask_no_I & mask_lower_tri

        S_clamp = torch.clamp(S, min=val_epsilon)
        lambda_i = S_clamp.unsqueeze(0)
        lambda_j = S_clamp.unsqueeze(1)
        ratio = lambda_j / lambda_i

        # form E
        E = torch.ones_like(ratio).to(S.device)

        # [1]too small
        mask_too_small = mask_init & (ratio == 1) & (lambda_i == val_epsilon)
        E[mask_too_small] = grad_epsilon

        # [2]taylor - arithmetic sequence
        mask_normal = mask_init & (~mask_too_small)
        del mask_too_small, mask_init
        diff = torch.abs(lambda_i - lambda_j)
        mask_normal_and_equal = mask_normal & (diff == 0)  # arithmetic sequence
        i_mtrx = S_clamp.repeat(S_clamp.shape[0], 1)  # get lambda
        E[mask_normal_and_equal] = (
            1 / (i_mtrx[mask_normal_and_equal].pow(2))
        ) * nTaylor
        del mask_normal_and_equal, lambda_i, lambda_j
        # [3]taylor - geometric sequence
        mask_normal_and_close = (
            mask_normal & (diff <= diff_epsilon) & (diff > 0)
        )  # geometric sequence
        q_2 = ratio[mask_normal_and_close].pow(2)
        E[mask_normal_and_close] = (1 / (i_mtrx[mask_normal_and_close].pow(2))) * (
            (1 - q_2**nTaylor) / (1 - q_2)
        )
        del mask_normal_and_close, q_2, ratio
        # [4]other
        mask_other = mask_normal & (diff > diff_epsilon)
        j_mtrx = i_mtrx.t()
        E[mask_other] = 1 / (
            (i_mtrx[mask_other] - j_mtrx[mask_other])
            * (i_mtrx[mask_other] + j_mtrx[mask_other])
        )
        del mask_other, diff, i_mtrx, j_mtrx

        mask_pad = mask_no_I & (~mask_lower_tri)
        E[mask_pad] = -1.0 * (E.t()[mask_pad])
        del mask_no_I, mask_pad, mask_lower_tri
        # ic(1/E)

        skew = lambda X: X - X.T

        UhgU_skew = skew(U.T @ gU) * E if gU is not None else torch.zeros_like(U)
        VhgV_skew = skew(Vh @ gVh.T) * E if gVh is not None else torch.zeros_like(Vh)
        gA_core = (
            U
            @ (
                UhgU_skew @ S.diag_embed()
                + S.diag_embed() @ VhgV_skew
                + torch.diag_embed(gS)
            )
            @ Vh
        )

        gUSinv = gU / S_clamp.unsqueeze(-2)
        additional_term1 = (gUSinv - U @ (U.T @ gUSinv)) @ Vh

        SinvgVh = gVh / S_clamp.unsqueeze(-1)
        additional_term2 = U @ (SinvgVh - (SinvgVh @ Vh.T) @ Vh)

        gA_core += additional_term1 + additional_term2

        return gA_core, None


class SVDTransformLayer(nn.Module):
    def __init__(
        self,
        gamma=None,
        SEQ_LEN=None,
        beta=None,
        input_size=None,
        output_size=None,
        weight_size=None,
        weight=None,
        bias=None,
        name=None,
        device=None,
        svd_impl="stable",
        svd_eps=1e-8,
        svd_differentiation="tikhonov",
    ):
        super(SVDTransformLayer, self).__init__()
        # Create gamma directly on `device` so it stays a leaf Parameter (a
        # trailing .to(device) that moves devices yields a non-leaf tensor whose
        # requires_grad cannot be set). gamma is trainable.
        self.gamma = nn.Parameter(
            torch.tensor(gamma, dtype=computeSVD_dtype, device=device)
        )
        if bias is None:
            self.ori = nn.Linear(input_size, output_size, bias=False).to(device)
        else:
            self.ori = nn.Linear(input_size, output_size, bias=True).to(device)
            self.ori.bias = nn.Parameter(bias).to(device)
        self.ori.weight = nn.Parameter(weight).to(device)
        if name:
            self.name = name
        self.ori_weight_size = weight_size
        in_plus_out_size = input_size + output_size
        nblocks = input_size / SEQ_LEN
        self.nblocks_total_size = in_plus_out_size * nblocks
        self.beta = beta
        # Differentiable-SVD implementation: "stable" = Dobi's truncated
        # svd_lowrank + Taylor backward; "cans" = cans_svd full SVD via polar
        # decomposition (eps = backward regularization constant).
        self.svd_impl = svd_impl
        self.svd_eps = svd_eps
        self.svd_differentiation = svd_differentiation

    def _decompose(self, x, full_rank):
        if self.svd_impl == "cans":
            from cans_svd import svd as cans_svd_fn

            U, S, Vt = cans_svd_fn(
                x,
                backend="torch",
                differentiation=self.svd_differentiation,
                eps=self.svd_eps,
            )
            return U, S, Vt.mT
        gamma_range = self.gamma.detach()
        gamma_range = int(gamma_range.int() + 5)
        gamma_range = min(full_rank, max(1, gamma_range))
        return stable_lowrank_SVD.apply(x, gamma_range)

    def _forward_impl(self, x):
        orig_dtype = x.dtype
        x = self.ori(x).to(computeSVD_dtype)

        if x.dim() == 3:
            x = x.squeeze(0)
            squeeze_need = 1
        else:
            squeeze_need = 0

        assert x.dim() == 2
        m, n = x.shape
        full_rank = min(m, n)

        U, S, V = self._decompose(x, full_rank)
        sequence = torch.arange(1, len(S) + 1).to(x.device)
        # Differentiable clamp: min(len(S), max(1, gamma)) returns a *Python int*
        # when gamma >= len(S), silently detaching gamma (zero grad). clamp keeps
        # it a connected tensor for any value.
        real_gamma = torch.clamp(self.gamma, min=1.0, max=float(len(S)))

        Trunc = 0.5 * torch.tanh(self.beta * (real_gamma - sequence)) + 0.5
        S_transformed = (S) * (Trunc)
        S_diag = torch.diag_embed(S_transformed)
        x_transformed = torch.matmul(torch.matmul(U, S_diag), V.T)  # U @ S_diag @ V.T
        # The SVD + gamma-gate math always runs in computeSVD_dtype (fp32) — half
        # precision backward through the SVD yields NaN gamma gradients. The
        # OUTPUT returns to the incoming activation dtype: identical to the old
        # fp32-everywhere behavior when the model is fp32 (default rank
        # training), and keeps the residual stream bf16 in low-mem mode.
        real_x = x_transformed.to(orig_dtype)

        if squeeze_need == 1:
            real_x = real_x.unsqueeze(0)
        return real_x

    def forward(self, x):
        # Low-mem rank training (base model kept in bf16 instead of the fp32
        # upcast): the per-layer fp32 SVD tensors (U/S/V + activation upcasts)
        # would otherwise be retained by autograd for every layer until
        # backward. Checkpointing recomputes them in backward instead, saving
        # only the small half-precision input. torch.utils.checkpoint with
        # use_reentrant=False preserves/restores RNG state, so the randomized
        # svd_lowrank recomputes identically. Detected via the wrapped weight's
        # dtype: fp32 weights (default rank training) keep the direct path.
        if self.ori.weight.dtype != computeSVD_dtype and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)
