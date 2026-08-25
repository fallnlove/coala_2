"""coala-reinforce: discrete per-layer rank training via REINFORCE (ablation).

Same pipeline as the gate trainer in ``pipeline/coala_rank.py`` (same cached
COALA factors, data, epochs, val protocol, integer-rank checkpoints), but the
rank is a discrete per-layer variable trained with a score-function gradient:

* policy: per-layer **discretized Gaussian** — trainable mean ``mu`` (stored in
  the existing ``COALARankLayer.gamma`` parameter, init ``r_uni``). Sampling
  ``r = clamp(round(N(mu, sigma^2)), 1, K)`` is exactly the folded discretized
  Gaussian, so the pmf below and the sampler define the same distribution.
  ``sigma`` is either scheduled (cosine anneal, relative to ``r_uni``) or — the
  default — **learnable per layer**: ``sigma = sigma_min +
  softplus(rf_sigma_lr_scale * rho)``. The lr-scale lives INSIDE the
  parameterization because the shared Adam moves every parameter ~lr per step:
  a raw log-sigma at lr 0.5-2.0 would collapse sigma to the floor within a few
  steps; softplus (not a hard clamp) keeps the gradient alive at the floor;
* every model forward uses the existing HARD truncation
  (``COALARankLayer.hard_rank``) under ``torch.inference_mode()`` — no gate and
  no autograd graph through the model; only the tiny log-pmf/entropy graph over
  the ``mu``/``rho`` scalars is differentiated;
* estimator: **REINFORCE with a leave-one-out baseline** over ``rf_samples``
  independent rank vectors per window: ``A_i = R_i - mean_{j!=i} R_j`` (this
  form already contains the N/(N-1) factor — do not add another; any constant
  is absorbed by the normalizer). ``sum_i A_i = 0`` identically; the baseline
  is independent of ``r_i``, so the gradient is unbiased. N=2 degenerates to
  the pairwise difference. Advantages are normalized by a bias-corrected EMA of
  ``mean_i(A_i^2)``, updated once per window AFTER use (scale-only — never
  subtract a cross-window mean; the LOO baseline is already zero-mean within
  the window);
* reward: composite ``R = -(w_ce*CE + w_kl*KL + w_hid*HID)`` in log space —
  CE is the LM loss; KL is the token-mean KL(teacher||student) on logits
  (fp32 log-softmax, chunked over the sequence); HID is the layer-mean relative
  L2 between hidden states (embeddings excluded — they are uncompressed). The
  teacher is a second fp16 copy of the ORIGINAL model (the rank model's target
  weights are freed at attach), loaded only when ``w_kl>0 or w_hid>0``, run
  once per window and shared across the N samples;
* an optional **entropy bonus** ``-w_entropy * sum_l H(pi_l)`` (exact discrete
  entropy over a truncated bin window) drives exploration — meaningful only
  with learnable sigma (validated);
* budget: ``rf_budget=penalty`` applies the gate trainer's
  ``_compression_loss_coala`` verbatim at ``gamma=mu`` (deterministic pathwise
  gradient — the budget never rides the RL channel). ``rf_budget=project``
  enforces the budget HARD: every sampled rank vector is projected DOWN onto
  ``sum((m+n)*r) <= B`` before its reward forward. The projection is a fixed
  function of ``r``, so the score-function gradient with the log-pmf of the
  UNPROJECTED sample is exactly unbiased for ``J(mu)=E[R(proj(r))]``; ``mu``
  itself is projected in place at the start of every forward (projected-Adam
  with a one-step lag), and the deterministic policy (val + extraction) is
  ``proj(round(mu))``.

The training loop itself is ``dobi._train_gammas`` reused verbatim: this module
only provides a wrapper whose ``forward(input_ids)`` returns the surrogate.
"""

from __future__ import annotations

import contextlib
import gc
import math
from typing import Dict, List

import torch
import torch.nn.functional as F

from pipeline import coala_rank

_GEN_SEED_OFFSET = 424243  # sampling stream distinct from data seeds
_KL_CHUNK = 512  # tokens per fp32 log-softmax chunk


# --------------------------------------------------------------------------- #
# Policy pmf / sampling / entropy
# --------------------------------------------------------------------------- #
def _discretized_gaussian_logpmf(r, mu, sigma, K):
    """Log-pmf of the folded discretized Gaussian on ``{1..K}`` (per layer).

    ``P(r) = Phi((r+.5-mu)/sigma) - Phi((r-.5-mu)/sigma)`` for interior bins,
    with the edge bins absorbing the tails (``P(1)=Phi((1.5-mu)/sigma)``,
    ``P(K)=1-Phi((K-.5-mu)/sigma)``) — sums to exactly 1. Computed on the side
    where the CDFs are small (sign flip) to avoid fp cancellation, in fp64.
    Differentiable w.r.t. ``mu`` AND ``sigma`` (the only detach is in the
    branch predicate).
    """
    r = r.to(torch.float64)
    a = (r - 0.5 - mu) / sigma
    b = (r + 0.5 - mu) / sigma
    ndtr = torch.special.ndtr
    lower = ndtr(b) - ndtr(a)  # well-conditioned when the bin sits below mu
    upper = ndtr(-a) - ndtr(-b)  # well-conditioned when the bin sits above mu
    above = (r - mu.detach()) > 0
    p = torch.where(above, upper, lower)
    p = torch.where(r <= 1.0, ndtr(b), p)  # left edge bin: (-inf, 1.5]
    p = torch.where(r >= K, ndtr(-a), p)  # right edge bin: [K-0.5, inf)
    p = torch.where(K <= 1.0, torch.ones_like(p), p)  # degenerate single-bin
    return torch.log(p.clamp_min(1e-30))


def _round_clamp(z, K):
    """Continuous sample -> integer rank in [1, K] (per layer)."""
    return torch.minimum(z.round().clamp_min(1.0), K)


def _sample_gaussian(mu, sigma, K, generator, n):
    """``n`` INDEPENDENT integer rank vectors around ``mu`` (detached) -> [n, L].

    Independence across samples is required by the leave-one-out baseline
    (a mirrored pair would correlate the baseline with the sample and bias it).
    """
    eps = torch.randn(
        (n, mu.shape[0]), generator=generator, device=mu.device, dtype=torch.float64
    )
    z = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    return _round_clamp(z, K.unsqueeze(0))


def _rloo_advantages(rewards: List[float]) -> List[float]:
    """Leave-one-out advantages ``A_i = R_i - mean_{j!=i} R_j``.

    This form already contains the N/(N-1) factor relative to ``R_i - mean(R)``
    — do not add another (any constant scale is absorbed by the EMA
    normalizer). ``sum_i A_i == 0`` identically.
    """
    n = len(rewards)
    total = sum(rewards)
    return [r - (total - r) / (n - 1) for r in rewards]


def _entropy_discretized(mu, sigma, K, half_width=8.0):
    """Exact entropy of the folded discretized Gaussian, per layer -> [L].

    Sums ``-p log p`` over the bins within ``mu +- half_width*sigma`` (mass
    outside is negligible), including the folded edge bins when the window
    reaches them. Window BOUNDS come from detached ``mu``/``sigma``; the bin
    log-probs stay attached — differentiable w.r.t. both. Correctly saturates
    at 0 near the sigma floor and at log K for huge sigma.
    """
    with torch.no_grad():
        mu_d, sig_d = mu.detach(), sigma.detach()
        lo = torch.maximum((mu_d - half_width * sig_d).floor(), torch.ones_like(mu_d))
        lo = torch.minimum(lo, K)
        hi = torch.minimum((mu_d + half_width * sig_d).ceil(), K)
        hi = torch.maximum(hi, lo)
        width = int((hi - lo).max().item()) + 1
    offsets = torch.arange(width, device=mu.device, dtype=torch.float64)
    r = lo.unsqueeze(0) + offsets.unsqueeze(1)  # [W, L]
    valid = r <= hi.unsqueeze(0)
    logp = _discretized_gaussian_logpmf(
        r, mu.unsqueeze(0), sigma.unsqueeze(0), K.unsqueeze(0)
    )
    p = logp.exp() * valid
    return -(p * logp).sum(dim=0)


# --------------------------------------------------------------------------- #
# Budget projection (rf_budget=project)
# --------------------------------------------------------------------------- #
def _scale_down_feasible(x, w, budget, iters=64):
    """Iteratively scale ``x`` down (elementwise >= 1) until ``sum(w*x) <= budget``.

    One proportional pass undershoots: layers clamped at 1 cannot give budget
    back, so the unclamped remainder must be rescaled until feasible.
    """
    x = x.clone()
    free = torch.ones_like(x, dtype=torch.bool)
    for _ in range(iters):
        cost = float((w * x).sum())
        if cost <= budget + 1e-9:
            break
        fixed = float((w[~free] * x[~free]).sum())
        denom = float((w[free] * x[free]).sum())
        if denom <= 0:
            break
        s = (budget - fixed) / denom
        x[free] = x[free] * s
        newly = free & (x < 1.0)
        if not bool(newly.any()):
            break
        x[newly] = 1.0
        free = free & ~newly
    return x.clamp_min(1.0)


def _project_budget_continuous(mu, w, budget, K):
    """Project real-valued ``mu`` DOWN onto ``sum(w*mu) <= budget`` within [1, K].

    Multiplicative scaling is not the Euclidean projection, but it is a
    deterministic, monotone retraction — the RL gradient corrects any
    allocation bias it introduces. No-op when already feasible.
    """
    mu = torch.minimum(mu.clamp_min(1.0), K)
    if float((w * mu).sum()) <= budget:
        return mu
    if budget < float(w.sum()):
        print(
            "WARNING: budget below the all-ones floor; using minimum ranks", flush=True
        )
        return torch.ones_like(mu)
    return torch.minimum(_scale_down_feasible(mu, w, budget), K)


def _project_budget_int(r, w, budget, K):
    """Project an integer rank vector DOWN onto the budget within [1, K].

    Iterative proportional scale (see ``_scale_down_feasible``), floor, then a
    greedy +1 fill by largest fractional remainder that is budget-aware in COST
    units (a +1 in up_proj costs (m+n)=18432 vs 8192 for q_proj — only added if
    it fits the remaining slack). Never raises a layer above its sampled value.
    No-op when already feasible.
    """
    r = torch.minimum(r.to(torch.float64).clamp_min(1.0), K)
    if float((w * r).sum()) <= budget:
        return r
    if budget < float(w.sum()):
        print(
            "WARNING: budget below the all-ones floor; using minimum ranks", flush=True
        )
        return torch.ones_like(r)
    x = _scale_down_feasible(r, w, budget)
    xf = torch.minimum(x.floor().clamp_min(1.0), K)
    slack = budget - float((w * xf).sum())
    frac = x - xf
    for idx in torch.argsort(frac, descending=True).tolist():
        cap = min(float(K[idx]), float(r[idx]))
        if xf[idx] + 1.0 <= cap and float(w[idx]) <= slack:
            xf[idx] += 1.0
            slack -= float(w[idx])
    return xf


# --------------------------------------------------------------------------- #
# Teacher (original model) for KL / hidden-matching rewards
# --------------------------------------------------------------------------- #
def _load_teacher(cfg, device):
    """Second fp16 copy of the ORIGINAL model (the rank model's target weights
    are freed at attach, so its outputs cannot be recovered from the student)."""
    from transformers import AutoModelForCausalLM

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    teacher = (
        AutoModelForCausalLM.from_pretrained(
            str(cfg.model.id),
            torch_dtype=torch.float16,
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
        .to(device)
        .eval()
    )
    teacher.requires_grad_(False)
    if torch.cuda.is_available():
        print(
            f"REINFORCE teacher loaded: allocated "
            f"{torch.cuda.memory_allocated() / 1e9:.1f} GB, reserved "
            f"{torch.cuda.memory_reserved() / 1e9:.1f} GB",
            flush=True,
        )
    return teacher


def _token_kl(teacher_logits, student_logits, chunk=_KL_CHUNK) -> float:
    """Token-mean KL(teacher || student), fp32 log-softmax, chunked over tokens."""
    t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    s = student_logits.reshape(-1, student_logits.shape[-1])
    total, n_tok = 0.0, t.shape[0]
    for start in range(0, n_tok, chunk):
        tl = t[start : start + chunk].float().log_softmax(-1)
        sl = s[start : start + chunk].float().log_softmax(-1)
        total += float((tl.exp() * (tl - sl)).sum())
    return total / max(1, n_tok)


def _hidden_mse(teacher_hiddens, student_hiddens) -> float:
    """Layer-mean relative L2 between hidden states, skipping the embeddings
    (layer 0 is uncompressed — an exact match that would dilute the mean)."""
    total, n = 0.0, 0
    for t_h, s_h in zip(teacher_hiddens[1:], student_hiddens[1:]):
        t32 = t_h.float()
        diff = (s_h.float() - t32).pow(2).sum()
        denom = t32.pow(2).sum().clamp_min(1e-12)
        total += float(diff / denom)
        n += 1
    return total / max(1, n)


@contextlib.contextmanager
def _hard_ranks(layers, ranks):
    """Set every layer's ``hard_rank`` for the duration (restore prior after).

    Guards the silent soft-gate fallback: within the context every target layer
    is guaranteed to run the hard-truncation path.
    """
    prev = [layer.hard_rank for layer in layers]
    try:
        for layer, r in zip(layers, ranks.tolist()):
            layer.hard_rank = int(r)
        yield
    finally:
        for layer, p in zip(layers, prev):
            layer.hard_rank = p


def _sigma_rel_for_epoch(epoch, n_epochs, rs) -> float:
    """Cosine-annealed relative sigma (scheduled mode only)."""
    init = float(rs.get("rf_sigma_init", 0.05))
    final = float(rs.get("rf_sigma_final", 0.005))
    T = max(1, int(n_epochs) - 1)
    t = min(max(int(epoch), 0), T)
    return final + 0.5 * (init - final) * (1.0 + math.cos(math.pi * t / T))


def _inv_softplus(y):
    """softplus^{-1}(y) = log(expm1(y)) for y > 0 (elementwise, fp64)."""
    return torch.log(torch.expm1(y.clamp_min(1e-2)))


class _CoalaReinforceTrainWrapper(coala_rank._CoalaRankTrainWrapper):
    """Surrogate-loss wrapper driven by ``dobi._train_gammas`` unchanged.

    ``forward(input_ids)`` samples rank vectors, measures composite rewards
    with hard truncation under ``inference_mode``, and returns the REINFORCE
    surrogate plus the analytic (pathwise) penalties. Only the per-layer
    ``gamma`` (policy mean) parameters — and ``rho`` when sigma is learned —
    carry gradient.
    """

    def __init__(self, model, cfg, r_unis: Dict[str, int], teacher=None):
        super().__init__(model, cfg)
        rs = cfg.compress.coala_rank_selection
        self._layers = [
            m for m in model.modules() if isinstance(m, coala_rank.COALARankLayer)
        ]
        if not self._layers:
            raise ValueError("no COALARankLayer found — attach layers first")
        self.register_buffer(
            "_rf_r_uni",
            torch.tensor(
                [float(r_unis[layer.name]) for layer in self._layers],
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "_rf_K",
            torch.tensor(
                [float(layer.max_rank) for layer in self._layers],
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "_rf_w",
            torch.tensor(
                [
                    float(layer.in_features + layer.out_features)
                    for layer in self._layers
                ],
                dtype=torch.float64,
            ),
        )
        self._rf_mn_sum = float(
            sum(layer.in_features * layer.out_features for layer in self._layers)
        )
        self._rf_budget_units = float(cfg.compress.ratio) * self._rf_mn_sum

        self._n_epochs = int(rs.n_epochs)
        self._epoch = 0
        self._sigma_min = float(rs.get("rf_sigma_min", 1.0))
        self._rs = rs
        self._n_samples = max(2, int(rs.get("rf_samples", 4)))
        self._budget_mode = str(rs.get("rf_budget", "penalty"))
        self._w_ce = float(rs.get("rf_w_ce", 1.0))
        self._w_kl = float(rs.get("rf_w_kl", 0.0))
        self._w_hid = float(rs.get("rf_w_hidden", 0.0))
        self._w_ent = float(rs.get("rf_w_entropy", 0.0))
        self._learn_sigma = bool(rs.get("rf_learn_sigma", True))
        self._alpha = float(rs.get("rf_sigma_lr_scale", 0.1))
        self._adv_beta = float(rs.get("rf_adv_ema_beta", 0.99))
        self._adv_floor = float(rs.get("rf_adv_std_floor", 1e-3))
        self._ema_sq = 0.0  # EMA of mean_i(A_i^2), updated once/window AFTER use
        self._ema_t = 0
        self._seed = coala_rank._rank_selection_seed(cfg) + _GEN_SEED_OFFSET
        self._gen = None

        if self._budget_mode == "project":
            print(
                "REINFORCE: rf_budget=project — hard budget; lambda_reg ignored.",
                flush=True,
            )
        if self._learn_sigma:
            sigma0 = torch.maximum(
                torch.full_like(self._rf_r_uni, self._sigma_min),
                float(rs.get("rf_sigma_init", 0.05)) * self._rf_r_uni,
            )
            rho0 = _inv_softplus(sigma0 - self._sigma_min) / self._alpha
            self._rf_rho = torch.nn.Parameter(rho0.float())
        else:
            self._rf_rho = None

        # Teacher kept in a plain list, NOT registered as a submodule:
        # _compression_loss_coala/_extract_ranks traverse self.model.modules(),
        # and module registration would also expose it to accidental .to() casts.
        self._teacher = []
        if self._w_kl > 0 or self._w_hid > 0:
            if teacher is not None:
                self._teacher = [teacher]
            else:
                device = next(model.parameters()).device
                self._teacher = [_load_teacher(cfg, device)]

    # ------------------------------------------------------------------ #
    def _reset(self):
        super()._reset()
        # _diag survives epoch_means()->_reset(); epoch_diag() resets it itself
        # (the epoch callback calls epoch_means() before epoch_diag()).
        if not hasattr(self, "_diag"):
            self._diag = self._fresh_diag()

    @staticmethod
    def _fresh_diag():
        return {
            "absadv": 0.0,
            "flip": 0.0,
            "ce": 0.0,
            "kl": 0.0,
            "hid": 0.0,
            "ent": 0.0,
            "slack": 0.0,
            "n": 0,
        }

    def _generator(self, device):
        if self._gen is None or str(self._gen.device) != str(device):
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self._seed)
        return self._gen

    def _sigma_tensor(self):
        """Per-layer sigma. Graph tensor (through rho) when sigma is learned."""
        if self._learn_sigma:
            return self._sigma_min + F.softplus(self._alpha * self._rf_rho.double())
        rel = _sigma_rel_for_epoch(self._epoch, self._n_epochs, self._rs)
        return (rel * self._rf_r_uni).clamp_min(self._sigma_min)

    def advance_epoch(self):
        """Step the sigma schedule (scheduled mode; harmless when learned)."""
        self._epoch += 1

    def extra_trainable_params(self):
        """Extra parameters (rho) to append to the optimizer/early-stopper list."""
        return [self._rf_rho] if self._rf_rho is not None else []

    # ------------------------------------------------------------------ #
    def _policy_ranks(self):
        """Deterministic policy: round(mu) clamped to [1, K]; additionally
        projected onto the budget in project mode (fp64 vector)."""
        mu = torch.stack([layer.gamma.detach() for layer in self._layers]).double()
        mu = torch.where(torch.isfinite(mu), mu, self._rf_r_uni)  # nan -> init
        r = _round_clamp(mu, self._rf_K)
        if self._budget_mode == "project":
            r = _project_budget_int(r, self._rf_w, self._rf_budget_units, self._rf_K)
        return r

    def policy_rank_dict(self) -> Dict[str, int]:
        """Integer ranks of the deterministic policy (projected in project mode).

        Used for checkpointing/extraction instead of mutating ``gamma`` and
        re-reading it: a finalize-style write-back mid-training would snap the
        live means to integers and freeze their fractional progress.
        """
        ranks = self._policy_ranks().tolist()
        return {layer.name: int(r) for layer, r in zip(self._layers, ranks)}

    def _set_policy_hard(self):
        for layer, r in zip(self._layers, self._policy_ranks().tolist()):
            layer.hard_rank = int(r)

    def _project_mu_(self):
        """Project the policy means in place onto the budget (no grad)."""
        with torch.no_grad():
            mu = torch.stack([layer.gamma for layer in self._layers]).double()
            proj = _project_budget_continuous(
                mu, self._rf_w, self._rf_budget_units, self._rf_K
            )
            if not torch.equal(proj, mu):
                for layer, v in zip(self._layers, proj.tolist()):
                    layer.gamma.data.fill_(float(v))

    # ------------------------------------------------------------------ #
    def _reward_components(self, input_ids, ranks, t_logits, t_hiddens):
        """(CE, KL, HID) at the given hard ranks — one student forward."""
        need_hidden = self._w_hid > 0
        with _hard_ranks(self._layers, ranks):
            with torch.inference_mode():
                out = self.model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False,
                    output_hidden_states=need_hidden,
                )
                ce = float(out.loss)
                kl = _token_kl(t_logits, out.logits) if self._w_kl > 0 else 0.0
                hid = _hidden_mse(t_hiddens, out.hidden_states) if need_hidden else 0.0
        return ce, kl, hid

    def forward(self, input_ids):
        device = input_ids.device
        project = self._budget_mode == "project"
        if project:
            self._project_mu_()

        mu32 = torch.stack([layer.gamma for layer in self._layers])  # graph leaves
        mu = mu32.double()
        sigma = self._sigma_tensor()  # graph through rho when learned
        gen = self._generator(device)
        ranks = _sample_gaussian(
            mu.detach(), sigma.detach(), self._rf_K, gen, self._n_samples
        )  # [N, L]

        # Teacher outputs once per window, shared across the N samples.
        t_logits = t_hiddens = None
        if self._teacher:
            with torch.inference_mode():
                t_out = self._teacher[0](
                    input_ids=input_ids,
                    use_cache=False,
                    output_hidden_states=self._w_hid > 0,
                )
            t_logits = t_out.logits
            if self._w_hid > 0:
                t_hiddens = t_out.hidden_states

        rewards, evaluated = [], []
        ce_acc = kl_acc = hid_acc = ppl_acc = 0.0
        for i in range(self._n_samples):
            r_eval = ranks[i]
            if project:
                r_eval = _project_budget_int(
                    r_eval, self._rf_w, self._rf_budget_units, self._rf_K
                )
            ce, kl, hid = self._reward_components(
                input_ids, r_eval, t_logits, t_hiddens
            )
            if not (math.isfinite(ce) and math.isfinite(kl) and math.isfinite(hid)):
                self._set_policy_hard()
                return torch.tensor(float("nan"), device=device)
            rewards.append(-(self._w_ce * ce + self._w_kl * kl + self._w_hid * hid))
            evaluated.append(r_eval)
            ce_acc += ce
            kl_acc += kl
            hid_acc += hid
            ppl_acc += math.exp(min(ce, 80.0))

        advs = _rloo_advantages(rewards)
        mean_sq = sum(a * a for a in advs) / self._n_samples
        if self._ema_t == 0:
            denom = max(math.sqrt(mean_sq), self._adv_floor)
        else:
            corr = self._ema_sq / (1.0 - self._adv_beta**self._ema_t)
            denom = max(math.sqrt(corr), self._adv_floor)

        surr = None
        for i in range(self._n_samples):
            logp = _discretized_gaussian_logpmf(ranks[i], mu, sigma, self._rf_K).sum()
            term = -(advs[i] / denom) * logp
            surr = term if surr is None else surr + term
        surr = surr / self._n_samples

        ent_mean = 0.0
        if self._w_ent > 0:
            H = _entropy_discretized(mu, sigma, self._rf_K)
            surr = surr - self._w_ent * H.sum()
            ent_mean = float(H.detach().mean())

        value = coala_rank._wrong_value_loss_coala(self.model, device)
        if project:
            policy = self._policy_ranks()
            ratio_val = float((self._rf_w * policy).sum()) / self._rf_mn_sum
            reg_val = 0.0
            total = surr.float() + value
        else:
            reg, ratio = coala_rank._compression_loss_coala(
                self.model,
                self.target_ratio,
                self.lambda_reg,
                device,
                objective=self.ratio_objective,
            )
            ratio_val = float(ratio.detach())
            reg_val = float(reg.detach())
            total = surr.float() + reg + value

        # EMA update AFTER use (the current window must not normalize itself).
        self._ema_sq = self._adv_beta * self._ema_sq + (1 - self._adv_beta) * mean_sq
        self._ema_t += 1

        n = self._n_samples
        self._n += 1
        self._sum["ppl"] += ppl_acc / n
        self._sum["reg"] += reg_val
        self._sum["value"] += float(value.detach())
        self._sum["ratio"] += ratio_val
        stacked = torch.stack(evaluated)
        self._diag["absadv"] += sum(abs(a) for a in advs) / (n * denom)
        self._diag["flip"] += float((stacked != stacked[0]).any(dim=0).double().mean())
        self._diag["ce"] += ce_acc / n
        self._diag["kl"] += kl_acc / n
        self._diag["hid"] += hid_acc / n
        self._diag["ent"] += ent_mean
        if project:
            self._diag["slack"] += (
                1.0 - float((self._rf_w * stacked[-1]).sum()) / self._rf_budget_units
            )
        self._diag["n"] += 1

        self._set_policy_hard()  # deterministic post-window state
        return total

    # ------------------------------------------------------------------ #
    def val_perplexity_at_policy(self, windows, device) -> float:
        """Eval-protocol val ppl at the deterministic (projected) policy.

        Always CE-perplexity — even for w_ce=0 reward arms — so best-epoch
        selection stays comparable across arms and with the paper table.
        """
        with _hard_ranks(self._layers, self._policy_ranks()):
            return super().val_perplexity(windows, device)

    def epoch_diag(self) -> str:
        """One extra diagnostics line per epoch (plot parsers ignore it)."""
        sigma = self._sigma_tensor().detach()
        rel = sigma / self._rf_r_uni
        n = max(1, self._diag["n"])
        ema = math.sqrt(
            self._ema_sq / max(1e-12, 1.0 - self._adv_beta ** max(1, self._ema_t))
        )
        d = self._diag
        line = (
            f"    reinforce: sigma_rel={float(rel.mean()):.4f}"
            f"/{float(rel.max()):.4f} mean|adv|={d['absadv'] / n:.3f} "
            f"flip_frac={d['flip'] / n:.3f} ema_dstd={ema:.4f} "
            f"ce={d['ce'] / n:.3f} kl={d['kl'] / n:.4f} hid={d['hid'] / n:.4f} "
            f"ent={d['ent'] / n:.2f} slack={d['slack'] / n:.4f}"
        )
        self._diag = self._fresh_diag()
        return line
