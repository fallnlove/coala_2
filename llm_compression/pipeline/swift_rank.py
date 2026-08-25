from __future__ import annotations

import gc
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from pipeline import data as data_mod

COMPUTE_DTYPE = torch.float32

# Transformer-block index inside a module name (llama/mistral/qwen: model.layers.N.*).
_BLOCK_RE = re.compile(r"\.(?:layers|h|blocks)\.(\d+)\.")


class _LayerInfo(NamedTuple):
    """One target Linear: ``m`` = out_features, ``n`` = in_features."""

    name: str
    module: nn.Module
    block: Optional[int]
    mtype: str
    m: int
    n: int

    @property
    def group(self) -> Tuple[str, int, int]:
        """Budget pool key: same projection in the same shape across blocks."""
        return (self.mtype, self.m, self.n)


# --------------------------------------------------------------------------- #
# Small pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def _kbar(m: int, n: int, ratio: float) -> int:
    """Uniform per-layer rank at ``ratio`` — COALA's formula verbatim
    (``vendor/coala/Linear.py``: ``ceil(ratio*m*n/(m+n))`` clamped to
    ``[1, min(m, n)]``), so a uniform Swift-SVD run and every other uniform
    baseline in the repo share the same per-layer budget."""
    return max(1, min(math.ceil(ratio * m * n / (m + n)), m, n))


def _tail_loss(sv: torch.Tensor, k: int) -> float:
    """``eps*_k = sqrt(sum_{j>k} sigma_j^2)`` — the exact rank-k reconstruction
    loss of the Swift-SVD operator (paper eq. 4). Zero once ``k`` covers the
    spectrum."""
    if k >= sv.numel():
        return 0.0
    tail = sv[k:].to(torch.float64)
    return float(torch.sqrt(torch.sum(tail * tail)))


def _fro_norm(sv: torch.Tensor) -> float:
    """``||Y||_F`` from the spectrum (used by the released-code score mode)."""
    s = sv.to(torch.float64)
    return float(torch.sqrt(torch.sum(s * s)))


def _erank(sv: torch.Tensor) -> float:
    """Effective rank via spectral entropy (paper eq. 11, ``exp(-sum p ln p)``
    with ``p = sigma / sum sigma``) — logged as the layer's compressibility
    diagnostic (NER = erank / len(sigma))."""
    s = sv.to(torch.float64)
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 0.0
    p = s / torch.sum(s)
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    return float(torch.exp(entropy))


def _beta_from_bi(bi: Sequence[float]) -> List[float]:
    """Block-Influence scores -> importance weights in [1, 2] (Algorithm 2 line
    1: min-max normalise, then shift by 1). A constant/degenerate input maps to
    all-ones, which makes the ``beta**alpha`` factor inert."""
    if not len(bi):
        return []
    lo, hi = min(bi), max(bi)
    return [float((b - lo) / (hi - lo + 1e-6) + 1.0) for b in bi]


def _score(
    beta: float, eps: float, ynorm: float, alpha: float, mode: str = "tail"
) -> float:
    """Compressibility score ``beta**alpha * log(arg)**(1-alpha)`` (paper
    eq. 12). ``mode='tail'`` uses ``arg = e + eps*_{kbar}`` (paper: the log is
    then >= 1); ``mode='ynorm'`` uses ``arg = 1 + ||Y||_F`` (released code)."""
    arg = (math.e + eps) if mode == "tail" else (1.0 + ynorm)
    base = max(math.log(max(arg, 1.0 + 1e-12)), 1e-12)
    return float(beta**alpha) * float(base ** (1.0 - alpha))


def _base_rank(kbar: int, m: int, rmax: int, delta: float, mode: str, ratio: float):
    """Guaranteed minimal rank per layer. ``delta_kbar`` (paper): ``delta*kbar``.
    ``released``: ``0.25*ratio*out_features`` — the released code's base, which
    keys off the covariance dimension (= out_features) rather than the budget."""
    if mode == "released":
        base = int(0.25 * ratio * m)
    else:
        base = int(delta * kbar)
    return max(1, min(base, rmax))


def _allocate_type(
    kbar: int,
    rmax: Sequence[int],
    scores: Sequence[float],
    delta: float,
    *,
    out_features: Sequence[int] = (),
    base_mode: str = "delta_kbar",
    ratio: float = 1.0,
) -> List[int]:
    """Algorithm 2 for one budget pool. Returns per-layer integer ranks.

    Exactly conserves the pool budget ``kbar * L`` whenever it is feasible
    (``sum rmax >= kbar*L``): every layer starts at its guaranteed base rank and
    the remaining budget is handed out in proportion to ``scores``, water-filling
    around layers that saturate at ``rmax``. Ties/rounding leftovers go to the
    highest-scoring non-saturated layers, as in the paper's pseudo-code.

    ``delta=1`` reproduces uniform allocation (paper A.2), which is the natural
    sanity anchor for the grid search.
    """
    L = len(scores)
    if L == 0:
        return []
    budget = int(kbar) * L
    outs = list(out_features) if len(out_features) else [int(r) for r in rmax]
    ranks = [
        _base_rank(
            int(kbar), int(outs[i]), int(rmax[i]), float(delta), base_mode, ratio
        )
        for i in range(L)
    ]
    if sum(ranks) > budget:
        # Only reachable with base_mode='released' at ratios where the released
        # base exceeds the uniform budget; fall back to a flat feasible base.
        print(
            f"WARNING: Swift-SVD base ranks ({sum(ranks)}) exceed the pool budget "
            f"({budget}); falling back to a flat base rank.",
            flush=True,
        )
        flat = max(1, budget // L)
        ranks = [min(flat, int(rmax[i])) for i in range(L)]

    remaining = budget - sum(ranks)
    while remaining > 0:
        active = [i for i in range(L) if ranks[i] < rmax[i]]
        if not active:
            print(
                f"WARNING: Swift-SVD allocation saturated with {remaining} rank(s) of "
                "budget unspent (all layers at min(m, n)) — the target ratio is not "
                "reachable for this module type.",
                flush=True,
            )
            break
        total = sum(scores[i] for i in active)
        assigned = 0
        if total > 0:
            for i in active:
                add = min(
                    int(remaining * scores[i] / total),
                    int(rmax[i]) - ranks[i],
                )
                if add > 0:
                    ranks[i] += add
                    assigned += add
        if assigned == 0:
            # Leftover smaller than the smallest proportional share (or all-zero
            # scores): hand out single ranks, highest score first.
            for i in sorted(active, key=lambda j: (-scores[j], j)):
                if assigned >= remaining:
                    break
                ranks[i] += 1
                assigned += 1
        remaining -= assigned
    return ranks


def _param_ratio(infos: Sequence[_LayerInfo], ranks: Dict[str, int]) -> float:
    """Achieved parameter ratio over the target modules: ``sum r(m+n) / sum mn``
    (the quantity ``scripts/collect_compression.py`` reports)."""
    num = sum(ranks[i.name] * (i.m + i.n) for i in infos if i.name in ranks)
    den = sum(i.m * i.n for i in infos if i.name in ranks)
    return float(num) / float(den) if den else float("nan")


# --------------------------------------------------------------------------- #
# Model inspection
# --------------------------------------------------------------------------- #
def _layer_infos(model, target_modules) -> List[_LayerInfo]:
    """Target Linears with their block index and module type (grouping key)."""
    from pipeline.coala_rank import _target_linears

    infos = []
    for name, mod in _target_linears(model, target_modules):
        match = _BLOCK_RE.search(name)
        block = int(match.group(1)) if match else None
        m, n = mod.weight.shape
        infos.append(
            _LayerInfo(
                name=name,
                module=mod,
                block=block,
                mtype=name.rpartition(".")[2],
                m=int(m),
                n=int(n),
            )
        )
    return infos


def _group_infos(
    infos: Sequence[_LayerInfo],
) -> Dict[Tuple[str, int, int], List[_LayerInfo]]:
    groups: Dict[Tuple[str, int, int], List[_LayerInfo]] = {}
    for info in infos:
        groups.setdefault(info.group, []).append(info)
    return groups


# --------------------------------------------------------------------------- #
# Stage B step 1: layer importance (ShortGPT Block Influence)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _block_importance(model, windows, device, angular: bool = False) -> List[float]:
    """Per-block Block Influence: mean over tokens of ``1 - cos(h_l, h_{l+1})``
    (``arccos(cos)/pi`` when ``angular``), averaged over ``windows``.

    Mirrors ShortGPT / the Swift-SVD release (``llm-layer-importance``), including
    its use of HF ``output_hidden_states`` — note the last entry is taken AFTER
    the final norm, so the last block's score is measured against the normed
    state (same convention as the reference implementation).
    """
    if not len(windows):
        return []
    model.eval()
    totals: Optional[torch.Tensor] = None
    n_used = 0
    for w in tqdm(windows, desc="Swift-SVD: layer importance", unit="sample"):
        out = model(
            input_ids=w.unsqueeze(0).to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states
        scores = []
        for lo, hi in zip(hs[:-1], hs[1:]):
            a = lo[0].to(COMPUTE_DTYPE)
            b = hi[0].to(COMPUTE_DTYPE)
            denom = a.norm(dim=-1) * b.norm(dim=-1)
            cos = torch.nan_to_num((a * b).sum(dim=-1) / denom, nan=0.5)
            if angular:
                bi = torch.arccos(cos.clamp(-1.0, 1.0)) / math.pi
            else:
                bi = 1.0 - cos
            scores.append(bi.mean())
        step = torch.stack(scores).detach().float().cpu()
        if not torch.isfinite(step).all():
            continue
        totals = step if totals is None else totals + step
        n_used += 1
        del out, hs
    if totals is None or n_used == 0:
        print(
            "WARNING: Swift-SVD layer importance produced no finite windows — "
            "falling back to uniform importance.",
            flush=True,
        )
        return []
    return [float(v) for v in (totals / n_used)]


# --------------------------------------------------------------------------- #
# Stage B step 2/3: spectra and cached projectors
# --------------------------------------------------------------------------- #
def _layer_x(dic, name: str, device) -> torch.Tensor:
    """The layer's compacted calibration activations as fp32 on ``device``."""
    return torch.cat(dic[name], dim=1).to(device=device, dtype=COMPUTE_DTYPE)


def _spectrum_pass(dic, infos, ratio: float, device) -> Dict[str, torch.Tensor]:
    """Singular values of ``Z = W @ X`` per layer (CPU fp32).

    ``svdvals`` (no singular vectors) keeps this pass cheap; the vectors are
    computed once in ``_cache_pass`` for the ranks the grid can actually request.
    """
    from lib import Solvers

    spectra: Dict[str, torch.Tensor] = {}
    for info in tqdm(infos, desc="Swift-SVD: spectra", unit="layer"):
        X = _layer_x(dic, info.name, device)
        W = info.module.weight.detach().to(device=device, dtype=COMPUTE_DTYPE)
        sv = torch.linalg.svdvals(Solvers.swift_svd_project(W, X)).detach().cpu()
        spectra[info.name] = sv
        k = _kbar(info.m, info.n, ratio)
        print(
            f"  spectrum {info.name}: sv={sv.numel()} erank={_erank(sv):.1f} "
            f"eps@{k}={_tail_loss(sv, k):.4e} ynorm={_fro_norm(sv):.4e}",
            flush=True,
        )
        del X, W
    _free()
    return spectra


def _cache_pass(
    dic, infos, r_need: Dict[str, int], device, cache_dtype
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Cache ``(U[:, :r], P = U[:, :r]ᵀ W)`` on CPU per layer.

    ``W'_r = U[:, :r] @ P[:r]`` for every ``r <= r_need``, so all grid candidates
    are served from these two matrices without another SVD.
    """
    from lib import Solvers

    caches: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    driver = "gesvd" if str(device).startswith("cuda") else None
    for info in tqdm(infos, desc="Swift-SVD: projector cache", unit="layer"):
        r = int(r_need[info.name])
        X = _layer_x(dic, info.name, device)
        W = info.module.weight.detach().to(device=device, dtype=COMPUTE_DTYPE)
        U, _, _ = torch.linalg.svd(
            Solvers.swift_svd_project(W, X), full_matrices=False, driver=driver
        )
        U_r = U[:, :r]
        P_r = U_r.T @ W
        caches[info.name] = (
            U_r.detach().to("cpu", cache_dtype),
            P_r.detach().to("cpu", cache_dtype),
        )
        del X, W, U, U_r, P_r
    _free()
    return caches


# --------------------------------------------------------------------------- #
# Stage B step 4: candidate application / restoration
# --------------------------------------------------------------------------- #
def _snapshot_weights(infos) -> Dict[str, torch.Tensor]:
    """CPU copies of the target weights in the model's own dtype, so restoring
    after the grid search is bit-exact and the final injection sees the original
    model."""
    return {i.name: i.module.weight.detach().to("cpu").clone() for i in infos}


@torch.no_grad()
def _restore_weights(infos, snapshot: Dict[str, torch.Tensor]) -> None:
    for info in infos:
        info.module.weight.data.copy_(snapshot[info.name].to(info.module.weight.device))


@torch.no_grad()
def _apply_allocation(infos, caches, ranks: Dict[str, int], device) -> None:
    """Overwrite each target weight with its rank-``r`` Swift-SVD approximation
    ``W'_r = U_r U_rᵀ W``, reconstructed densely from the cached projectors.

    Dense (rather than factored) so a candidate can be scored by a plain forward
    pass; the winning ranks are materialised as real low-rank factors later, by
    the normal COALA injector.
    """
    for info in infos:
        r = int(ranks[info.name])
        U, P = caches[info.name]
        U_r = U[:, :r].to(device=device, dtype=COMPUTE_DTYPE)
        P_r = P[:r, :].to(device=device, dtype=COMPUTE_DTYPE)
        w = info.module.weight
        w.data.copy_((U_r @ P_r).to(w.dtype))
        del U_r, P_r


# --------------------------------------------------------------------------- #
# Windows / seeds
# --------------------------------------------------------------------------- #
def _swift_seed(cfg: DictConfig) -> int:
    rs = cfg.compress.get("swift_rank_selection", {})
    return int(rs.get("seed", cfg.get("seed", 0)))


def _swift_windows(cfg: DictConfig, tokenizer, block: str, split: str, offset: int):
    rs = cfg.compress.swift_rank_selection
    sub = rs.get(block, {})
    return data_mod.make_token_windows(
        tokenizer=tokenizer,
        name=str(sub.dataset),
        split=split,
        seq_len=int(sub.seq_len),
        num_samples=int(sub.num_samples),
        seed=_swift_seed(cfg) + offset,
        dataset_cache_dir=cfg.calibration.get("dataset_cache_dir", "./datasets"),
    )


def _val_ppl(model, windows) -> float:
    """Held-out perplexity with the eval-stage protocol, so candidate numbers are
    directly comparable to the paper tables."""
    if not len(windows):
        return float("nan")
    from pipeline import evaluate as evaluate_mod

    model.eval()
    return evaluate_mod.evaluate_perplexity(model, data_mod.ppl_input_ids(windows))


# --------------------------------------------------------------------------- #
# Logging / checkpointing
# --------------------------------------------------------------------------- #
def _candidate_log_line(alpha: float, val_ppl: float, is_best: bool) -> str:
    mark = " [best]" if is_best else ""
    return f"  alpha {alpha:.2f}: val_ppl={val_ppl:.3f}{mark}"


def _log_allocation(label: str, infos, ranks: Dict[str, int]) -> None:
    """Per-module-type rank summary + achieved parameter ratio for one candidate."""
    parts = []
    for (mtype, _m, _n), group in sorted(_group_infos(infos).items()):
        rs = [ranks[i.name] for i in group if i.name in ranks]
        if rs:
            parts.append(f"{mtype}={sum(rs) / len(rs):.0f}[{min(rs)}-{max(rs)}]")
    print(
        f"  {label} ranks: param_ratio={_param_ratio(infos, ranks):.4f} "
        + " ".join(parts),
        flush=True,
    )


def _swift_checkpoint_path(cfg: DictConfig):
    return cfg.compress.get("swift_rank_selection", {}).get("checkpoint", None)


def _save_swift_checkpoint(path, rank_dict: Dict[str, int], cfg: DictConfig) -> None:
    """Standard rank-checkpoint payload (same schema as COALA/Dobi), so the
    ranks are loadable by every consumer in the repo (``rank_source=checkpoint``,
    ``scripts/collect_compression.py``, cross-method experiments)."""
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(cfg.model.id),
        "ratio": float(cfg.compress.ratio),
        "seq_len": int(cfg.calibration.seq_len),
        "ranks": {k: int(v) for k, v in rank_dict.items()},
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(p)


def _save_json(path, payload) -> None:
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(p)


def _grid_path(ckpt) -> Optional[str]:
    if not ckpt:
        return None
    p = Path(str(ckpt))
    return str(p.with_name(p.stem + "_grid.json"))


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Stage B orchestration
# --------------------------------------------------------------------------- #
def _candidate_allocations(
    infos,
    spectra,
    beta: Dict[Optional[int], float],
    ratio: float,
    delta: float,
    alphas: Sequence[float],
    score_mode: str,
    base_mode: str,
) -> Tuple[Dict[float, Dict[str, int]], Dict[str, int]]:
    """All grid candidates plus the uniform allocation.

    Returns ``({alpha: {name: rank}}, {name: kbar})``.
    """
    groups = _group_infos(infos)
    candidates: Dict[float, Dict[str, int]] = {float(a): {} for a in alphas}
    uniform: Dict[str, int] = {}

    for (_mtype, m, n), group in groups.items():
        kbar = _kbar(m, n, ratio)
        # Feasible rank: min(m, n) and never past the observed spectrum — the
        # singular values beyond it are exactly zero, so rank spent there buys
        # nothing (and under-sampled calibration makes the spectrum the binding
        # limit).
        rmax = [min(i.m, i.n, int(spectra[i.name].numel())) for i in group]
        outs = [i.m for i in group]
        eps = [_tail_loss(spectra[i.name], kbar) for i in group]
        ynorm = [_fro_norm(spectra[i.name]) for i in group]
        betas = [beta.get(i.block, 1.0) for i in group]
        for info in group:
            uniform[info.name] = kbar
        for alpha in alphas:
            scores = [
                _score(betas[j], eps[j], ynorm[j], float(alpha), score_mode)
                for j in range(len(group))
            ]
            ranks = _allocate_type(
                kbar,
                rmax,
                scores,
                delta,
                out_features=outs,
                base_mode=base_mode,
                ratio=ratio,
            )
            for info, r in zip(group, ranks):
                candidates[float(alpha)][info.name] = int(r)
    return candidates, uniform


def select_ranks(
    model, tokenizer, cfg: DictConfig, coala_config, device
) -> Dict[str, int]:
    """Swift-SVD Algorithm 2 + grid search. Requires ``coala_config.dic``.

    Leaves the model's weights exactly as they were found.
    """
    from pipeline.model import resolve_dtype

    rs = cfg.compress.swift_rank_selection
    ratio = float(cfg.compress.ratio)
    delta = float(rs.get("delta", 0.5))
    alphas = [float(a) for a in rs.get("alphas", [0.6])]
    score_mode = str(rs.get("score", "tail"))
    base_mode = str(rs.get("base_rank", "delta_kbar"))
    cache_dtype = resolve_dtype(str(rs.get("cache_dtype", "float16")))
    angular = bool(rs.get("importance", {}).get("angular", False))

    infos = _layer_infos(model, cfg.compress.target_modules)
    print(
        f"Swift-SVD rank selection: {len(infos)} target layers, ratio={ratio}, "
        f"delta={delta}, score={score_mode}, base_rank={base_mode}, "
        f"alphas={alphas}",
        flush=True,
    )

    # 1. Layer importance on the intact model (must precede any weight edit).
    imp_windows = _swift_windows(cfg, tokenizer, "importance", "train", 1000)
    bi = _block_importance(model, imp_windows, device, angular=angular)
    betas = _beta_from_bi(bi)
    beta_map: Dict[Optional[int], float] = {i: b for i, b in enumerate(betas)}
    for i, (raw, b) in enumerate(zip(bi, betas)):
        print(f"  importance block {i}: bi={raw:.6f} beta={b:.4f}", flush=True)
    if not betas:
        print(
            "WARNING: Swift-SVD has no layer importance — alpha only shifts the "
            "score exponent, so the grid degenerates towards loss-only allocation.",
            flush=True,
        )
    n_missing_beta = sum(1 for i in infos if i.block not in beta_map)
    if n_missing_beta:
        print(
            f"WARNING: {n_missing_beta} target layer(s) have no block index / "
            "importance score; using beta=1.0 for them.",
            flush=True,
        )
    if rs.get("importance_path", None):
        _save_json(
            rs.get("importance_path"),
            {
                "model": str(cfg.model.id),
                "dataset": str(rs.importance.dataset),
                "num_samples": int(rs.importance.num_samples),
                "seq_len": int(rs.importance.seq_len),
                "angular": angular,
                "bi": bi,
                "beta": betas,
            },
        )
        print(
            f"Swift-SVD: saved layer importance to {rs.get('importance_path')}",
            flush=True,
        )
    del imp_windows
    _free()

    # 2. Spectra (loss-vs-rank curve per layer) from the calibration activations.
    spectra = _spectrum_pass(coala_config.dic, infos, ratio, device)

    # 3. Candidate allocations (pure arithmetic) + the uniform anchor.
    candidates, uniform = _candidate_allocations(
        infos, spectra, beta_map, ratio, delta, alphas, score_mode, base_mode
    )
    r_need = {
        i.name: max([uniform[i.name]] + [candidates[a][i.name] for a in candidates])
        for i in infos
    }

    # 4. Cache projectors up to the largest rank the grid can request, snapshot
    #    the original weights, then score every candidate by held-out perplexity.
    val_windows = _swift_windows(cfg, tokenizer, "val", "validation", 104729)
    snapshot = _snapshot_weights(infos)
    caches = _cache_pass(coala_config.dic, infos, r_need, device, cache_dtype)

    grid: Dict[str, dict] = {}
    best_alpha, best_ppl, best_ranks = None, float("inf"), None
    ref_ppl = float("nan")
    try:
        _apply_allocation(infos, caches, uniform, device)
        ref_ppl = _val_ppl(model, val_windows)
        _log_allocation("uniform", infos, uniform)
        print(f"  uniform-rank reference (val): val_ppl={ref_ppl:.3f}", flush=True)
        grid["uniform"] = {
            "val_ppl": ref_ppl,
            "param_ratio": _param_ratio(infos, uniform),
            "ranks": uniform,
        }

        for alpha in alphas:
            ranks = candidates[float(alpha)]
            _apply_allocation(infos, caches, ranks, device)
            ppl = _val_ppl(model, val_windows)
            is_best = math.isfinite(ppl) and ppl < best_ppl
            if is_best:
                best_alpha, best_ppl, best_ranks = float(alpha), ppl, ranks
            _log_allocation(f"alpha {alpha:.2f}", infos, ranks)
            print(_candidate_log_line(float(alpha), ppl, is_best), flush=True)
            grid[f"{float(alpha):.2f}"] = {
                "val_ppl": ppl,
                "param_ratio": _param_ratio(infos, ranks),
                "ranks": ranks,
            }
    finally:
        _restore_weights(infos, snapshot)
        del caches, snapshot
        _free()

    if best_ranks is None:
        print(
            "WARNING: Swift-SVD grid search produced no finite validation "
            "perplexity — falling back to the uniform allocation.",
            flush=True,
        )
        best_alpha, best_ppl, best_ranks = None, float("nan"), uniform

    ckpt = _swift_checkpoint_path(cfg)
    grid_path = _grid_path(ckpt)
    if grid_path:
        _save_json(
            grid_path,
            {
                "model": str(cfg.model.id),
                "ratio": ratio,
                "delta": delta,
                "score": score_mode,
                "base_rank": base_mode,
                "best_alpha": best_alpha,
                "best_val_ppl": best_ppl,
                "candidates": grid,
            },
        )
        print(f"Swift-SVD: saved candidate grid to {grid_path}", flush=True)

    label = "uniform" if best_alpha is None else f"{best_alpha:.2f}"
    print(
        f"Swift-SVD selection: best alpha={label} delta={delta} "
        f"val_ppl={best_ppl:.3f} (uniform ref {ref_ppl:.3f})",
        flush=True,
    )
    all_ranks = list(best_ranks.values())
    mean_kbar = sum(uniform.values()) / max(1, len(uniform))
    print(
        "Swift-SVD rank diagnostic: "
        f"layers={len(best_ranks)} distinct={len(set(all_ranks))} "
        f"mean_rank kbar={mean_kbar:.1f}->best={sum(all_ranks) / len(all_ranks):.1f} "
        f"min={min(all_ranks)} max={max(all_ranks)} "
        f"param_ratio={_param_ratio(infos, best_ranks):.4f}",
        flush=True,
    )
    return dict(best_ranks)


# --------------------------------------------------------------------------- #
# Entry point used by compress.py
# --------------------------------------------------------------------------- #
def run_swift(model, tokenizer, cfg: DictConfig, coala_config):
    """Swift-SVD with non-uniform ranks: allocate (or load) then inject.

    Returns ``(model_or_None, ranks, coala_config)``; the model is ``None`` when
    ``compress.stop_after_ranks`` short-circuits after rank selection.
    """
    from coala.Injector import inject_coala

    from pipeline import coala_rank
    from pipeline import compress as compress_mod

    rank_source = str(cfg.compress.get("rank_source", "uniform"))
    rs = cfg.compress.get("swift_rank_selection", {})
    ckpt = _swift_checkpoint_path(cfg)
    device = next(model.parameters()).device

    if rank_source == "checkpoint":
        if not (ckpt and os.path.exists(str(ckpt))):
            raise FileNotFoundError(
                f"compress.rank_source=checkpoint but the rank checkpoint {ckpt!r} "
                "does not exist."
            )
        print(f"Swift-SVD: loading ranks from checkpoint {ckpt}", flush=True)
        rank_dict = coala_rank._load_coala_checkpoint(ckpt, cfg)
    else:
        resume = bool(rs.get("resume", False))
        have_ckpt = bool(ckpt) and resume and os.path.exists(str(ckpt))
        if have_ckpt:
            print(
                f"Swift-SVD: loading ranks from checkpoint {ckpt} (skip selection)",
                flush=True,
            )
            rank_dict = coala_rank._load_coala_checkpoint(ckpt, cfg)
        else:
            _collect(compress_mod, model, tokenizer, cfg, coala_config, device)
            rank_dict = select_ranks(model, tokenizer, cfg, coala_config, device)
            if ckpt:
                _save_swift_checkpoint(ckpt, rank_dict, cfg)
                print(f"Swift-SVD: saved ranks to checkpoint {ckpt}", flush=True)

    if bool(cfg.compress.get("stop_after_ranks", False)):
        print("Swift-SVD: stop_after_ranks=true — skipping compression.", flush=True)
        return None, rank_dict, coala_config

    ratio_dict = coala_rank._ranks_to_ratio_dict(model, rank_dict)
    # Any target Linear absent from the rank dict (partial/foreign checkpoint)
    # would KeyError inside process_layer; fall back to its uniform rank.
    #
    # CRITICAL: hold no reference to a target module past this point. The injector
    # swaps the modules out one at a time, and anything still pointing at the
    # originals keeps their weights resident alongside the fp32 adapters being
    # built. On Qwen3-32B that is 58GiB of bf16 weights that never get freed: the
    # injection died at 87% holding 138.2GiB of a 139.8GiB card, which is exactly
    # retained originals + adapters-so-far + the per-layer transient. Hence the
    # plain (name, m, n) tuples and the immediate del below.
    ratio = float(cfg.compress.ratio)
    missing = [
        (i.name, i.m, i.n)
        for i in _layer_infos(model, cfg.compress.target_modules)
        if i.name not in ratio_dict
    ]
    for name, m_, n_ in missing:
        k = _kbar(m_, n_, ratio)
        ratio_dict[name] = (k - 0.5) * (m_ + n_) / (m_ * n_)
    if missing:
        print(
            f"WARNING: {len(missing)} target layer(s) missing from the Swift-SVD "
            "rank dict; compressed at the uniform rank instead.",
            flush=True,
        )
    del missing
    _free()

    prev_threads = torch.get_num_threads()
    try:
        _collect(compress_mod, model, tokenizer, cfg, coala_config, device)
        if torch.cuda.is_available():
            print(
                "Swift-SVD: GPU allocated before injection: "
                f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB",
                flush=True,
            )
        ranks, model = inject_coala(coala_config, model, ranks=ratio_dict)
    finally:
        torch.set_num_threads(prev_threads)
    return model, ranks, coala_config


def _collect(compress_mod, model, tokenizer, cfg, coala_config, device) -> None:
    """Collect calibration activations once; reused by selection and injection."""
    if hasattr(coala_config, "dic"):
        return
    compress_mod._collect_activations(model, tokenizer, cfg, coala_config, device)
