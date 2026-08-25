"""Learned non-uniform ranks for COALA (Dobi-style rank selection).

COALA compresses each target Linear to a low-rank weight ``W' = U_r U_rᵀ W`` where
``U`` = left singular vectors of ``W @ S`` and ``S`` whitens the calibration
activations ``X`` (see ``vendor/lib/Solvers.py:weight_approx`` /
``weight_approx_with_regularization``). This module learns the per-layer rank by
differentiating through that approximation, mirroring ``pipeline/dobi.py``:

* a ``COALARankLayer`` replaces each target Linear during rank selection — it
  precomputes the COALA basis once (``cans_svd`` replaces ``torch.linalg.svd``),
  caches ``U_K`` / ``P_K = U_Kᵀ W`` (frozen), drops the original weight, and gates
  the kept directions with a smooth gate over a trainable ``gamma`` (Dobi-style);
* ``gamma`` is trained to minimise perplexity + a compression-ratio regulariser
  (the shared ``dobi._train_gammas``, fp32 — fp16 backward produces NaN gammas);
* the learned integer ranks are converted to per-layer ratios (midpoint formula,
  exact through ``COALA_Linear``'s ``r = ceil(ratio·m·n/(m+n))``) and fed to the
  normal COALA compression via ``inject_coala(..., ranks=...)``.

The differentiable SVD is computed **once per layer** (its torch path runs numpy
Remez per call, and ``gamma`` gates strictly post-SVD, so the SVD need not be in
the autograd graph — same structural inertness as Dobi's ``eps``).
"""

from __future__ import annotations

import gc
import math
import os
import tempfile
from types import SimpleNamespace
from typing import Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline import data as data_mod
from pipeline import dobi  # reuse _train_gammas, _load_rank_checkpoint

COMPUTE_DTYPE = torch.float32
MODEL_DTYPE = torch.float16  # final/compression model dtype
RANK_DTYPE = torch.float32  # rank training is fp32 (fp16 -> NaN gammas)


# --------------------------------------------------------------------------- #
# Differentiable COALA-rank layer
# --------------------------------------------------------------------------- #
def _coala_left_svd(W, X, K, cans_eps, cans_diff):
    """Top-``K`` left singular vectors of ``W @ whiten(X)`` via cans_svd.

    Mirrors the branch in ``Solvers.weight_approx`` (QR-whiten when X is wide).
    """

    if X.shape[0] <= X.shape[1]:
        _, S = torch.linalg.qr(X.T, mode="r")
        M = W @ S.T
    else:
        M = W @ X
    U, _, _ = torch.linalg.svd(M, driver="gesvd", full_matrices=False)
    return U[:, :K]


def _build_factors(W, X, K, r_uni, use_reg, adaptive, mu, cans_eps, cans_diff):
    """Build the cached COALA factors ``(U_K, P_K=U_Kᵀ W)`` for one layer (detached).

    ``use_reg``/``adaptive``/``mu`` reproduce ``weight_approx_with_regularization``
    and the adaptive-mu rule in ``vendor/coala/Linear.py``.
    """
    eff_mu = mu if use_reg else 0.0
    if use_reg and adaptive:
        U0 = _coala_left_svd(W, X, K, cans_eps, cans_diff)
        Ur = U0[:, :r_uni]
        tmp = W - Ur @ (Ur.T @ W)
        err = torch.linalg.norm(tmp @ X) / (torch.linalg.norm(tmp) + 1e-12)
        eff_mu = mu * float(err**2)
    if eff_mu > 0:
        n = X.shape[0]
        ident = math.sqrt(eff_mu) * torch.eye(n, device=X.device, dtype=X.dtype)
        X = torch.cat([X, ident], dim=1)
    U_K = _coala_left_svd(W, X, K, cans_eps, cans_diff)
    P_K = U_K.T @ W
    return U_K.detach(), P_K.detach()


class COALARankLayer(nn.Module):
    """Differentiable stand-in for a COALA target Linear during rank selection.

    Caches the COALA basis ``U_K`` and projection ``P_K = U_Kᵀ W`` (frozen) and
    forwards the smoothly rank-gated weight ``W'(gamma) = U_K diag(g) P_K``.
    """

    def __init__(
        self,
        U_K,
        P_K,
        gamma_init,
        beta,
        max_rank,
        bias,
        name,
        device,
        in_features,
        out_features,
        factor_dtype=None,
    ):
        super().__init__()
        # gamma/_seq/gate ALWAYS stay COMPUTE_DTYPE (fp32) so the score gradient
        # is fp32-stable; only the frozen factors may be stored at a smaller
        # `factor_dtype` (fp16 in low-mem mode) — the forward upcasts them back to
        # fp32 for the matmuls, so accuracy is unchanged, storage is halved.
        self._factor_dtype = factor_dtype or COMPUTE_DTYPE
        # leaf Parameter created directly on device (a trailing .to(device) that
        # moves devices un-leafs it and breaks requires_grad).
        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init), dtype=COMPUTE_DTYPE, device=device)
        )
        self.register_buffer("U_K", U_K.to(device, self._factor_dtype))
        self.register_buffer("P_K", P_K.to(device, self._factor_dtype))
        self.register_buffer(
            "_seq",
            torch.arange(1, int(max_rank) + 1, device=device, dtype=COMPUTE_DTYPE),
        )
        self.bias = nn.Parameter(bias.detach().to(device)) if bias is not None else None
        self.beta = float(beta)
        self.max_rank = int(max_rank)
        self.name = name
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # When set to an int r, forward uses a HARD rank-r truncation (gate
        # bypassed): W' = U_K[:, :r] @ P_K[:r, :]. Used to measure the uniform-rank
        # COALA reference (r = r_uni) before training. None -> soft-gated training.
        self.hard_rank = None

    def gate(self):
        return 0.5 * torch.tanh(self.beta * (self.gamma - self._seq)) + 0.5

    def _forward_impl(self, x):
        # Apply the factors directly instead of reconstructing W'=[out,in] every
        # step: y = ((x @ P_Kᵀ) * g) @ U_Kᵀ. Identical result, ~2x fewer FLOPs and
        # no [out,in] weight materialised (big win for the MLP layers). The matmuls
        # run in COMPUTE_DTYPE (fp32); factors are upcast per-forward when stored
        # fp16 (transient — see forward() for how that transience is enforced
        # under autograd). The gamma gradient therefore stays fp32 even when the
        # model runs in bf16 (output downcast at the end; autograd re-upcasts).
        xd = x.to(COMPUTE_DTYPE)
        U = self.U_K.to(COMPUTE_DTYPE)
        P = self.P_K.to(COMPUTE_DTYPE)
        if self.hard_rank is not None:
            r = max(1, min(int(self.hard_rank), self.max_rank))
            y = (xd @ P[:r, :].T) @ U[:, :r].T  # hard rank-r
        else:
            g = self.gate()
            y = ((xd @ P.T) * g) @ U.T
        if self.bias is not None:
            y = y + self.bias.to(COMPUTE_DTYPE)
        return y.to(x.dtype)

    def forward(self, x):
        # With reduced-precision factor STORAGE, the fp32 upcasts of U_K/P_K are
        # needed by backward (grad_gated = grad_y @ U etc.), so plain autograd
        # SAVES a full fp32 copy of every layer's factors until backward — 2x
        # the fp16 buffers, ~125 GB extra for Qwen3-32B, which is exactly what
        # OOM'd the first low_mem training runs. Activation checkpointing makes
        # backward RECOMPUTE the upcast+matmuls per layer instead (saves only
        # the small bf16 input), trading ~1.5x forward FLOPs in these layers
        # for ~2x lower resident factor memory. Only needed when grads are
        # being recorded; val/reference/no-grad paths take the direct route.
        if self._factor_dtype != COMPUTE_DTYPE and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _set_submodule(model, name, new_module):
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, new_module)


def _uniform_rank(m_, n_, ratio, objective="params"):
    """Per-layer uniform rank satisfying the compression objective at ``ratio``.

    ``params`` (default): ``ceil(ratio * m*n/(m+n))`` — the rank at which the
    factored params ``(m+n)*r`` hit ``ratio`` of the original ``m*n``.
    ``ranks``: ``ceil(ratio * m)`` — the rank-fraction budget ``sum r / sum m``
    (m = out_features), clamped to the feasible ``min(m, n)`` (wide MLP layers
    can have ``ratio*m > min(m,n)``, e.g. Mistral up/gate at ratio 0.7).
    """
    if objective == "ranks":
        return max(1, min(math.ceil(ratio * m_), m_, n_))
    return max(1, math.ceil(ratio * m_ * n_ / (m_ + n_)))


def _gamma_init_from(loaded, name, r_uni, K):
    """Initial gamma for one layer: the checkpoint's rank (rounded — Dobi
    checkpoints store float gammas) clamped to [1, K]; uniform fallback when
    the layer is absent from the file. Returns ``(gamma_init, from_ckpt)``."""
    if loaded is not None and name in loaded:
        return float(min(max(round(float(loaded[name])), 1), K)), True
    return float(min(r_uni, K)), False


def _factor_max_rank(m_, n_, r_uni, max_rank_ratio, low_mem=False):
    """Per-layer factor rank K (= gate width / gamma upper bound).

    Base rule: ``min(m, n, max(r_uni+16, ceil(max_rank_ratio*r_uni)))``. In
    low-mem mode additionally cap at the break-even rank
    ``r_cap = floor(m*n/(m+n))`` — the rank beyond which the factors
    ``m*r + n*r`` cost more than the original ``m*n`` weight, so storing more is
    never worthwhile. ``K >= 1`` always; ``gamma_init = min(r_uni, K)`` (set by
    the caller) stays a valid gate index even in the rare ``r_uni > r_cap`` case
    (ratio near 1)."""
    K = min(m_, n_, max(r_uni + 16, math.ceil(max_rank_ratio * r_uni)))
    if low_mem:
        K = min(K, max(1, math.floor(m_ * n_ / (m_ + n_))))
    return K


def _rank_selection_seed(cfg: DictConfig) -> int:
    """Distinct from calibration when datasets coincide (mirrors dobi)."""
    base = int(cfg.get("seed", 0))
    rs = cfg.compress.coala_rank_selection
    calib = cfg.get("calibration", {})
    same = str(rs.dataset) == str(calib.get("dataset", "")) and base == int(
        rs.get("seed", base)
    )
    seed = int(rs.get("seed", base))
    return seed + 1000 if same else seed


def _rank_windows(cfg: DictConfig, tokenizer):
    rs = cfg.compress.coala_rank_selection
    return data_mod.make_token_windows(
        tokenizer=tokenizer,
        name=str(rs.dataset),
        split="train",
        seq_len=int(rs.seq_len),
        num_samples=int(rs.num_samples),
        seed=_rank_selection_seed(cfg),
        dataset_cache_dir=cfg.calibration.get("dataset_cache_dir", "./datasets"),
    )


def _rank_val_windows(cfg: DictConfig, tokenizer):
    """Held-out validation windows for train/val ppl logging + early stopping.

    Drawn from the dataset's **validation** split (the same distribution the
    eval-stage / paper-table perplexity uses) with a distinct seed, so they are
    disjoint from both the training windows (train split) and the eval set
    (validation split, ``cfg.seed``). Combined with the eval-protocol perplexity
    in ``val_perplexity``, this makes the val ppl directly comparable to the table
    (e.g. C4). Empty when ``val_num_samples<=0``.
    """
    rs = cfg.compress.coala_rank_selection
    n = int(rs.get("val_num_samples", 0))
    if n <= 0:
        return []
    return data_mod.make_token_windows(
        tokenizer=tokenizer,
        name=str(rs.dataset),
        split="validation",
        seq_len=int(rs.seq_len),
        num_samples=n,
        seed=_rank_selection_seed(cfg) + 104729,
        dataset_cache_dir=cfg.calibration.get("dataset_cache_dir", "./datasets"),
    )


def _target_linears(model, target_modules):
    from peft.tuners.tuners_utils import check_target_module_exists

    ns = SimpleNamespace(target_modules=list(target_modules))
    return [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and check_target_module_exists(ns, n)
    ]


def _collect_rank_activations(model, cfg, target_modules, windows, device):
    """Collect per-layer calibration activations via COALA's hook flow (CPU)."""
    from coala.Injector import after_get_samples, prepare_get_samples

    rs = cfg.compress.coala_rank_selection
    ns = SimpleNamespace(
        target_modules=list(target_modules),
        params={"accumulate": int(rs.get("accumulate", 8))},
    )
    model.eval()
    model, hooks = prepare_get_samples(model, ns)
    with torch.no_grad():
        for w in tqdm(windows, desc="COALA: collect activations", unit="sample"):
            model(input_ids=w.unsqueeze(0).to(device))
    after_get_samples(model, ns, hooks)
    return ns.dic  # {name: [compactified X]} on CPU


def _attach_coala_rank_layers(model, cfg, dic, target_modules, device):
    """Replace each target Linear with a COALARankLayer (in place). One X on GPU
    at a time. Returns ``(gammas, gamma_inits, max_ranks, r_unis)`` where
    ``r_unis[name]`` is the per-layer uniform COALA rank (for the uniform-rank
    reference)."""
    rs = cfg.compress.coala_rank_selection
    target_ratio = float(cfg.compress.ratio)
    beta = float(rs.beta)
    use_reg = bool(rs.get("use_regularization", True))
    adaptive = bool(rs.get("adaptive", False))
    mu = float(rs.get("mu", 1.0))
    cans_eps = float(rs.get("cans_eps", 1e-8))
    cans_diff = str(rs.get("cans_differentiation", "tikhonov"))
    max_rank_ratio = float(rs.get("max_rank_ratio", 2.0))
    # Low-memory mode: store frozen factors in fp16 (built fp32) and cap the
    # factor rank at the break-even point r_cap = floor(m*n/(m+n)) — the rank
    # beyond which the factors m*r+n*r would cost more than the original m*n
    # weight. Halves factor storage AND clips gamma via the existing max_rank
    # machinery. Opt-in; default off keeps factors fp32 at the usual K.
    low_mem = bool(rs.get("low_mem", False))
    factor_dtype = torch.float16 if low_mem else COMPUTE_DTYPE
    factor_bytesize = 2 if low_mem else 4
    # Compression objective: "params" (default, sum (m+n)r / sum mn) or "ranks"
    # (sum r / sum out_features) — changes the per-layer uniform rank (and thus
    # the init, the uniform reference, and the K headroom derived from it).
    objective = str(rs.get("ratio_objective", "params"))
    # Optional gamma warm start from a ranks JSON (COALA int ranks or Dobi float
    # gammas; the shared loader warns on model/ratio mismatch).
    init_ckpt = rs.get("init_checkpoint", None)
    init_ranks = dobi._load_rank_checkpoint(str(init_ckpt), cfg) if init_ckpt else None
    n_init_missing = 0

    gammas, gamma_inits, max_ranks, r_unis = [], {}, {}, {}
    factor_bytes = 0
    targets = _target_linears(model, target_modules)
    for name, module in tqdm(targets, desc="COALA: build rank layers", unit="layer"):
        in_f, out_f = module.in_features, module.out_features
        m_, n_ = out_f, in_f  # W is [out, in]
        r_uni = _uniform_rank(m_, n_, target_ratio, objective)
        K = _factor_max_rank(m_, n_, r_uni, max_rank_ratio, low_mem)
        X = torch.cat(dic[name], dim=1).to(device, COMPUTE_DTYPE)
        W = module.weight.detach().to(device, COMPUTE_DTYPE)  # fp32 SVD input
        U_K, P_K = _build_factors(
            W, X, K, r_uni, use_reg, adaptive, mu, cans_eps, cans_diff
        )
        gamma_init, from_ckpt = _gamma_init_from(init_ranks, name, r_uni, K)
        if init_ranks is not None and not from_ckpt:
            n_init_missing += 1
        layer = COALARankLayer(
            U_K,
            P_K,
            gamma_init,
            beta,
            K,
            module.bias,
            name,
            device,
            in_f,
            out_f,
            factor_dtype=factor_dtype,
        )
        _set_submodule(model, name, layer)
        gammas.append(layer.gamma)
        gamma_inits[name] = gamma_init
        max_ranks[name] = K
        r_unis[name] = int(min(r_uni, K))
        factor_bytes += (U_K.numel() + P_K.numel()) * factor_bytesize
        del X, W, U_K, P_K, dic[name]
        gc.collect()
        torch.cuda.empty_cache()

    for p in model.parameters():
        p.requires_grad = False
    for g in gammas:
        g.requires_grad = True
    if n_init_missing:
        print(
            f"WARNING: init_checkpoint missing {n_init_missing}/{len(gammas)} "
            "layer(s) — those start at the uniform rank.",
            flush=True,
        )
    print(
        f"COALA rank layers: {len(gammas)} layers, cached factors "
        f"{factor_bytes / 1e9:.2f} GB",
        flush=True,
    )
    return gammas, gamma_inits, max_ranks, r_unis


# --------------------------------------------------------------------------- #
# Loss + training wrapper
# --------------------------------------------------------------------------- #
def _compression_loss_coala(
    model, target_ratio, lambda_reg, device, objective="params"
):
    """|achieved_ratio - target| * lambda_reg.

    ``objective="params"`` (default): COALA storage cost, ratio =
    ``sum (m+n)*soft_rank / sum m*n``. ``objective="ranks"``: aggregate
    rank fraction of the output dims, ratio = ``sum soft_rank / sum m``
    (m = out_features). Same loss form; the returned ``ratio`` (and thus the
    epoch-log ``ratio=`` column) is in the chosen objective's units.
    """
    size_new = torch.tensor(0.0, device=device)
    size_ori = 0.0
    for mod in model.modules():
        if isinstance(mod, COALARankLayer):
            soft_rank = mod.gate().sum()
            if objective == "ranks":
                size_new = size_new + soft_rank
                size_ori += mod.out_features
            else:
                size_new = size_new + (mod.in_features + mod.out_features) * soft_rank
                size_ori += mod.in_features * mod.out_features
    ratio = size_new / size_ori
    loss = lambda_reg * torch.abs(ratio - torch.tensor(target_ratio, device=device))
    return loss, ratio


def _wrong_value_loss_coala(model, device):
    """Penalise gamma outside [0, max_rank] (per layer)."""
    penalty = torch.tensor(0.0, device=device)
    for mod in model.modules():
        if isinstance(mod, COALARankLayer):
            upper = torch.tensor(float(mod.max_rank), device=device)
            penalty = (
                penalty
                + torch.relu(-mod.gamma) ** 2
                + torch.relu(mod.gamma - upper) ** 2
            )
    return penalty


class _CoalaRankTrainWrapper(nn.Module):
    """Full loss (ppl + reg + value) in one forward (DDP-correct), with logging."""

    def __init__(self, model, cfg: DictConfig):
        super().__init__()
        self.model = model
        rs = cfg.compress.coala_rank_selection
        self.target_ratio = float(cfg.compress.ratio)
        self.lambda_reg = float(rs.lambda_reg)
        self.ratio_objective = str(rs.get("ratio_objective", "params"))
        self._reset()

    def _reset(self):
        self._sum = {"ppl": 0.0, "reg": 0.0, "value": 0.0, "ratio": 0.0}
        self._n = 0

    def forward(self, input_ids):
        loss = self.model(input_ids=input_ids, labels=input_ids).loss.float()
        ppl = torch.exp(loss)
        reg, ratio = _compression_loss_coala(
            self.model,
            self.target_ratio,
            self.lambda_reg,
            input_ids.device,
            objective=self.ratio_objective,
        )
        value = _wrong_value_loss_coala(self.model, input_ids.device)
        self._n += 1
        self._sum["ppl"] += float(ppl.detach())
        self._sum["reg"] += float(reg.detach())
        self._sum["value"] += float(value.detach())
        self._sum["ratio"] += float(ratio.detach())
        return ppl + reg + value

    def epoch_means(self) -> Dict[str, float]:
        """Mean train ppl/reg/value/ratio over the epoch, then reset accumulators."""
        n = max(1, self._n)
        means = {k: v / n for k, v in self._sum.items()}
        self._reset()
        return means

    @torch.no_grad()
    def val_perplexity(self, windows, device) -> float:
        """Held-out perplexity via the EVAL protocol — token-level
        ``exp(sum NLL / num_tokens)`` (``evaluate.evaluate_perplexity``) — so it is
        directly comparable to the eval-stage / paper-table perplexity (e.g. C4),
        not the old per-window ``mean(exp(loss))``. ``windows`` is a list of 1-D
        token tensors; ``device`` is unused (samples move to ``model.device``)."""
        if not len(windows):
            return float("nan")
        from pipeline import evaluate as evaluate_mod

        self.model.eval()
        input_ids = data_mod.ppl_input_ids(windows)
        return evaluate_mod.evaluate_perplexity(self.model, input_ids)


# Early stopping (val-ppl patience + best-gamma restore) lives in dobi.py so both
# rank-selection pipelines share one implementation.
_EarlyStopper = dobi._EarlyStopper


def _extract_ranks(model, gamma_inits, max_ranks) -> Dict[str, int]:
    """Collect integer ranks; non-finite gamma falls back to init (with a warning)."""
    rank_dict, n_reset = {}, 0
    for mod in model.modules():
        if isinstance(mod, COALARankLayer):
            g = float(mod.gamma.detach().item())
            if not math.isfinite(g):
                g = gamma_inits[mod.name]
                n_reset += 1
            rank_dict[mod.name] = int(min(max(round(g), 1), max_ranks[mod.name]))
    if n_reset:
        print(
            f"WARNING: {n_reset} non-finite gamma(s) reset to init — rank training "
            "diverged; check learning rate.",
            flush=True,
        )
    return rank_dict


def _ranks_to_ratio_dict(model, rank_dict) -> Dict[str, float]:
    """Integer rank -> per-layer ratio using the bucket MIDPOINT so that
    ``COALA_Linear``'s ``r = ceil(ratio·m·n/(m+n))`` recovers exactly ``r``."""
    named = dict(model.named_modules())
    ratios = {}
    for name, r in rank_dict.items():
        mod = named.get(name)
        if mod is None:
            continue
        m_, n_ = mod.out_features, mod.in_features
        ratios[name] = (r - 0.5) * (m_ + n_) / (m_ * n_)
    return ratios


def _log_rank_diag(
    rank_dict, gamma_inits, max_ranks, n_skipped, n_nonfinite, model=None
):
    moved = sum(1 for k, v in rank_dict.items() if abs(v - gamma_inits[k]) > 1.0)
    distinct = len(set(rank_dict.values()))
    mean_init = sum(gamma_inits.values()) / len(gamma_inits)
    mean_now = sum(rank_dict.values()) / len(rank_dict)
    print(
        "COALA rank-train diagnostic: "
        f"layers={len(rank_dict)} moved(>1)={moved} distinct={distinct} "
        f"mean_rank init={mean_init:.1f}->now={mean_now:.1f} "
        f"skipped_steps={n_skipped} nonfinite_grad_steps={n_nonfinite}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Checkpointing (integer ranks)
# --------------------------------------------------------------------------- #
def _coala_checkpoint_path(cfg: DictConfig):
    return cfg.compress.coala_rank_selection.get("checkpoint", None)


def _save_coala_checkpoint(path, rank_dict, cfg: DictConfig) -> None:
    import json
    from pathlib import Path

    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(cfg.model.id),
        "ratio": float(cfg.compress.ratio),
        "seq_len": int(cfg.compress.coala_rank_selection.seq_len),
        "ranks": rank_dict,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(p)


def _load_coala_checkpoint(path, cfg: DictConfig) -> Dict[str, int]:
    ranks = dobi._load_rank_checkpoint(path, cfg)  # generic JSON loader + warnings
    return {k: int(round(v)) for k, v in ranks.items()}


# --------------------------------------------------------------------------- #
# Single-GPU rank selection
# --------------------------------------------------------------------------- #
# Shared with dobi (same log format for both rank-selection pipelines).
_epoch_log_line = dobi._epoch_log_line


def _uniform_reference_ppl(wrapper, r_unis, val_windows, device) -> float:
    """Validation perplexity of the UNIFORM-rank COALA model.

    Hard-truncate each layer's cached factors at its ``r_uni`` (the soft gate is
    bypassed), measured with the same eval protocol as ``val_perplexity``. This
    anchors the training curve to the uniform-rank baseline (≈ the paper-table
    uniform COALA number), so every per-epoch ``val_ppl`` is read against it.
    """
    if not len(val_windows):
        return float("nan")
    layers = [m for m in wrapper.model.modules() if isinstance(m, COALARankLayer)]
    try:
        for m in layers:
            m.hard_rank = int(r_unis[m.name])
        return wrapper.val_perplexity(val_windows, device)
    finally:
        for m in layers:
            m.hard_rank = None


def _print_reference_ppl(wrapper, rs, gamma_inits, r_unis, val_windows, device):
    """Pre-epoch-1 reference val ppl. Measured at the INIT ranks (labelled
    ``init-ranks``) when ``init_checkpoint`` warm-starts gamma, else at the
    uniform ranks — both labels are recognised by the log parsers (REF_RE in
    scripts/plot_ranktrain.py / collect_ranktrain.py)."""
    if rs.get("init_checkpoint", None):
        label, ranks = "init-ranks", gamma_inits
    else:
        label, ranks = "uniform-rank", r_unis
    ref = _uniform_reference_ppl(wrapper, ranks, val_windows, device)
    print(f"  {label} reference (val): val_ppl={ref:.3f}", flush=True)


def select_ranks(model, tokenizer, cfg: DictConfig, device) -> Dict[str, int]:
    rs = cfg.compress.coala_rank_selection
    # Rank training runs in fp32 by default (fp16 backward -> NaN gammas). In
    # low-mem mode we keep the base model in its loaded dtype (bf16, required)
    # so it never materialises at fp32 — only the COALARankLayer factors stay
    # resident, and the fp32-compute forward keeps gamma's gradient stable.
    if bool(rs.get("low_mem", False)):
        model = model.to(device)
    else:
        model = model.to(dtype=RANK_DTYPE, device=device)
    target_modules = OmegaConf.to_container(cfg.compress.target_modules, resolve=True)
    windows = _rank_windows(cfg, tokenizer)
    val_windows = _rank_val_windows(cfg, tokenizer)

    dic = _collect_rank_activations(model, cfg, target_modules, windows, device)
    gammas, gamma_inits, max_ranks, r_unis = _attach_coala_rank_layers(
        model, cfg, dic, target_modules, device
    )
    trainer = str(rs.get("trainer", "gate"))
    if trainer == "reinforce":
        from pipeline.coala_reinforce import _CoalaReinforceTrainWrapper

        wrapper = _CoalaReinforceTrainWrapper(model, cfg, r_unis).to(device)
        desc = "COALA-REINFORCE rank-train"
        # Learned-sigma parameters join the same optimizer list — and thereby
        # the _EarlyStopper snapshots, so best-restore covers mu AND sigma.
        gammas = gammas + list(wrapper.extra_trainable_params())
    else:
        wrapper = _CoalaRankTrainWrapper(model, cfg).to(device)
        desc = "COALA rank-train"

    def _current_ranks():
        # reinforce extracts via its deterministic policy (budget-projected in
        # project mode) WITHOUT mutating gamma; gate uses the plain extraction.
        fn = getattr(wrapper, "policy_rank_dict", None)
        return fn() if fn is not None else _extract_ranks(model, gamma_inits, max_ranks)

    _print_reference_ppl(wrapper, rs, gamma_inits, r_unis, val_windows, device)

    ckpt = _coala_checkpoint_path(cfg)
    every = int(rs.get("checkpoint_every", 0))
    stopper = _EarlyStopper(
        gammas,
        rs.get("early_stopping_patience", 0),
        rs.get("early_stopping_min_delta", 0.0),
    )
    # reinforce validates at the deterministic policy round(mu) (hard ranks);
    # the gate trainer validates with its soft gates.
    val_fn = getattr(wrapper, "val_perplexity_at_policy", wrapper.val_perplexity)

    def _on_epoch(epoch):
        means = wrapper.epoch_means()
        val_ppl = val_fn(val_windows, device)
        is_best, stop = stopper.update(val_ppl)
        print(_epoch_log_line(epoch, means, val_ppl, is_best), flush=True)
        diag = getattr(wrapper, "epoch_diag", None)
        if diag is not None:
            print(diag(), flush=True)
        getattr(wrapper, "advance_epoch", lambda: None)()
        if ckpt and every > 0 and (epoch + 1) % every == 0:
            _save_coala_checkpoint(ckpt, _current_ranks(), cfg)
        if stop:
            print(
                f"  early stop: val_ppl no better for {stopper.patience} epoch(s); "
                f"best={stopper.best:.3f} — restoring best ranks.",
                flush=True,
            )
        return stop

    n_skipped, n_nonfinite = dobi._train_gammas(
        wrapper,
        gammas,
        windows,
        rs,
        device,
        progress=True,
        on_epoch_end=_on_epoch,
        desc=desc,
    )
    stopper.restore_best()  # keep the best-val ranks, not the last epoch's
    rank_dict = _current_ranks()
    _log_rank_diag(rank_dict, gamma_inits, max_ranks, n_skipped, n_nonfinite)
    gc.collect()
    torch.cuda.empty_cache()
    return rank_dict


# --------------------------------------------------------------------------- #
# Multi-GPU rank selection (DDP; clone of dobi)
# --------------------------------------------------------------------------- #
def _rank_worker(
    rank: int, world_size: int, cfg_container: dict, out_path: str
) -> None:
    from pipeline._paths import add_reference_paths

    add_reference_paths()
    cfg = OmegaConf.create(cfg_container)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", dobi._ddp_port())
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.model.id),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.model.id),
        torch_dtype=RANK_DTYPE,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    ).to(device)

    target_modules = OmegaConf.to_container(cfg.compress.target_modules, resolve=True)
    windows_all = _rank_windows(cfg, tokenizer)
    # All workers build the SAME factors from the FULL calibration set (so the
    # shared gammas gate identical subspaces); only the TRAINING data is sharded.
    dic = _collect_rank_activations(model, cfg, target_modules, windows_all, device)
    gammas, gamma_inits, max_ranks, r_unis = _attach_coala_rank_layers(
        model, cfg, dic, target_modules, device
    )
    from torch.nn.parallel import DistributedDataParallel as DDP

    wrapper = _CoalaRankTrainWrapper(model, cfg).to(device)
    ddp = DDP(wrapper, device_ids=[rank], find_unused_parameters=False)

    windows = windows_all[rank::world_size]
    rs = cfg.compress.coala_rank_selection
    ckpt = _coala_checkpoint_path(cfg)
    every = int(rs.get("checkpoint_every", 0))
    val_windows = _rank_val_windows(cfg, tokenizer) if rank == 0 else []
    if rank == 0:
        _print_reference_ppl(wrapper, rs, gamma_inits, r_unis, val_windows, device)
    stopper = _EarlyStopper(
        gammas,
        rs.get("early_stopping_patience", 0),
        rs.get("early_stopping_min_delta", 0.0),
    )

    def _on_epoch(epoch):
        # Validation + early-stop decision on rank 0 (it holds the full val set and
        # the best-gamma snapshot), broadcast so every rank breaks at the same epoch
        # — a divergent decision would deadlock the next epoch's collective.
        stop_flag = torch.zeros(1, device=device)
        if rank == 0:
            means = wrapper.epoch_means()
            val_ppl = wrapper.val_perplexity(val_windows, device)
            is_best, stop = stopper.update(val_ppl)
            print(_epoch_log_line(epoch, means, val_ppl, is_best), flush=True)
            if ckpt and every > 0 and (epoch + 1) % every == 0:
                _save_coala_checkpoint(
                    ckpt, _extract_ranks(model, gamma_inits, max_ranks), cfg
                )
            if stop:
                print(
                    f"  early stop: val_ppl no better for {stopper.patience} "
                    f"epoch(s); best={stopper.best:.3f} — restoring best ranks.",
                    flush=True,
                )
            stop_flag[0] = 1.0 if stop else 0.0
        dist.broadcast(stop_flag, src=0)
        return bool(stop_flag.item())

    n_skipped, n_nonfinite = dobi._train_gammas(
        ddp,
        gammas,
        windows,
        rs,
        device,
        progress=(rank == 0),
        on_epoch_end=_on_epoch,
        desc="COALA rank-train",
    )
    dist.barrier()
    if rank == 0:
        stopper.restore_best()  # keep best-val ranks, not the last epoch's
        rank_dict = _extract_ranks(model, gamma_inits, max_ranks)
        _log_rank_diag(rank_dict, gamma_inits, max_ranks, n_skipped, n_nonfinite)
        torch.save(rank_dict, out_path)
    dist.barrier()
    dist.destroy_process_group()


def select_ranks_multigpu(cfg: DictConfig, world_size: int) -> Dict[str, int]:
    container = OmegaConf.to_container(cfg, resolve=True)
    out_path = os.path.join(tempfile.gettempdir(), f"coala_ranks_{os.getpid()}.pt")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", dobi._ddp_port())
    try:
        mp.spawn(
            _rank_worker,
            args=(world_size, container, out_path),
            nprocs=world_size,
            join=True,
        )
        return torch.load(out_path)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def _train_world_size(rs, n_gpus: int) -> int:
    """GPUs to train on. The DDP worker trains the soft-gate wrapper, so
    ``trainer=reinforce`` and ``low_mem`` (which fits one big model on a single
    GPU precisely because DDP would just replicate it) both force single-GPU."""
    world_size = min(n_gpus, int(rs.num_samples)) if n_gpus > 1 else 1
    if world_size > 1 and (
        str(rs.get("trainer", "gate")) == "reinforce" or bool(rs.get("low_mem", False))
    ):
        why = (
            "trainer=reinforce"
            if str(rs.get("trainer", "gate")) == "reinforce"
            else "low_mem"
        )
        print(
            f"COALA: {why} runs single-GPU; ignoring {n_gpus - 1} extra GPU(s) "
            "(DDP replicates rather than shards).",
            flush=True,
        )
        world_size = 1
    return world_size


# --------------------------------------------------------------------------- #
# Entry point used by compress.py
# --------------------------------------------------------------------------- #
def resolve_ranks(model, tokenizer, cfg: DictConfig, device):
    """Return the per-layer integer rank dict for COALA, per ``compress.rank_source``.

    ``checkpoint`` loads it; ``train`` runs rank selection (single or DDP) and
    saves a checkpoint. The passed ``model`` is freed when not used for single-GPU
    training (resume / multi-GPU reload their own).
    """
    rank_source = str(cfg.compress.get("rank_source", "uniform"))
    rs = cfg.compress.coala_rank_selection
    ckpt = _coala_checkpoint_path(cfg)

    if rank_source == "checkpoint":
        print(f"COALA: loading ranks from checkpoint {ckpt}", flush=True)
        return _load_coala_checkpoint(ckpt, cfg)

    resume = bool(rs.get("resume", True))
    have_ckpt = bool(ckpt) and resume and os.path.exists(str(ckpt))
    world_size = _train_world_size(rs, torch.cuda.device_count())

    if have_ckpt or world_size > 1:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if have_ckpt:
        print(
            f"COALA: loading ranks from checkpoint {ckpt} (skip training)", flush=True
        )
        rank_dict = _load_coala_checkpoint(ckpt, cfg)
    elif world_size > 1:
        print(f"COALA rank selection: DDP across {world_size} GPUs", flush=True)
        rank_dict = select_ranks_multigpu(cfg, world_size)
    else:
        rank_dict = select_ranks(model, tokenizer, cfg, device)

    if ckpt and not have_ckpt:
        _save_coala_checkpoint(ckpt, rank_dict, cfg)
        print(f"COALA: saved ranks to checkpoint {ckpt}", flush=True)
    return rank_dict
