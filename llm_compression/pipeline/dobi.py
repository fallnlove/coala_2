"""Dobi-SVD compression: rank selection + IPCA weight update (no remapping).

Ported from Dobi-SVD/svd_trainer.py (rank training) and Dobi-SVD/weight_updater.py
(IPCA weight update), driven by our YAML config instead of argparse + HF Trainer +
accelerate. Two clearly separated stages:

1. ``select_ranks``  — replace each nn.Linear with a differentiable activation-SVD
   layer (``dobi.stable_svd.SVDTransformLayer``) whose per-layer truncation
   position ``gamma`` is trained to minimise perplexity + a compression-ratio
   regulariser. Returns the per-layer ``gamma`` dict (the chosen ranks).
2. ``compress_with_ipca`` — using those ranks, collect activation right-singular
   vectors via GPU IncrementalPCA over the calibration set, project each weight
   onto the top principal components, and factor it into two low-rank Linears
   (``dobi.module.SVDTransformLayer``). The quantization/"remapping" stage is
   intentionally omitted.

Both stages are GPU-only (lots of SVD) and run the model in float16 with float32
SVD, matching Dobi. Vendored math lives in ``vendor/dobi/``.

Guards added vs upstream (which assumed Llama-2-7B, hidden >= seq_len): the
``RANK_RATIO``/``Ngamma`` integer ratios are clamped to >= 1 so grouped-query
models (Mistral, Llama-3.2) with kv projections smaller than ``seq_len`` don't hit
divide-by-zero / empty-IPCA bugs.
"""

from __future__ import annotations

import gc
import math
import os
import tempfile
from typing import Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline import data as data_mod

COMPUTE_DTYPE = torch.float32
MODEL_DTYPE = torch.float16  # IPCA weight-update + final factored model
RANK_DTYPE = torch.float32  # rank training: fp16 backward yields NaN gamma grads


def _no_svd_names(model_id: str):
    """Layer-name substrings to skip (mirrors Dobi's per-arch exclusions)."""
    lower = model_id.split("/")[-1].lower()
    if "opt" in lower:
        return ["project_out", "project_in", "lm_head"]
    return ["lm_head"]


def _is_target(name: str, module: nn.Module, no_svd) -> bool:
    return isinstance(module, nn.Linear) and all(x not in name for x in no_svd)


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, new_module)


def _rank_selection_seed(cfg: DictConfig) -> int:
    base = int(cfg.get("seed", 0))
    rs = cfg.compress.dobi.rank_selection
    calib = cfg.get("calibration", {})
    same = str(rs.dataset) == str(calib.get("dataset", "")) and base == int(
        rs.get("seed", base)
    )
    seed = int(rs.get("seed", base))
    return seed + 1000 if same else seed


# --------------------------------------------------------------------------- #
# Stage 1: rank selection (differentiable truncation training)
# --------------------------------------------------------------------------- #
def _attach_svd_layers(model, cfg: DictConfig, device, progress: bool = True):
    """Replace target Linears with trainable activation-SVD layers (in place).

    Returns ``(gammas, gamma_inits)``: the list of trainable gamma Parameters and
    the per-layer init values (fallback for non-finite gammas).
    """
    from dobi.stable_svd import SVDTransformLayer

    rs = cfg.compress.dobi.rank_selection
    seq_len = int(rs.seq_len)
    target_ratio = float(cfg.compress.ratio)
    beta = float(rs.beta)
    no_svd = _no_svd_names(str(cfg.model.id))
    svd_impl = str(rs.get("svd", "stable"))
    svd_eps = float(rs.get("cans_eps", 1e-8))
    svd_diff = str(rs.get("cans_differentiation", "tikhonov"))

    ori_weight_size = torch.tensor(0.0)
    gamma_inits: Dict[str, float] = {}
    targets = [(n, m) for n, m in model.named_modules() if _is_target(n, m, no_svd)]
    bar = tqdm(
        targets, desc="Dobi: build SVD layers", unit="layer", disable=not progress
    )
    for name, module in bar:
        in_f, out_f = module.in_features, module.out_features
        rank_ratio = max(1, int(min(in_f, out_f) / seq_len))  # guard GQA / small dims
        gamma_init = (1.0 / rank_ratio) * target_ratio * in_f * out_f / (in_f + out_f)
        gamma_inits[name] = gamma_init
        weight_size = torch.tensor(float(in_f * out_f))
        ori_weight_size = ori_weight_size + weight_size
        new_layer = SVDTransformLayer(
            gamma=gamma_init,
            SEQ_LEN=seq_len,
            beta=beta,
            input_size=in_f,
            output_size=out_f,
            weight_size=weight_size,
            weight=module.weight,
            bias=module.bias,
            name=name,
            device=device,
            svd_impl=svd_impl,
            svd_eps=svd_eps,
            svd_differentiation=svd_diff,
        )
        _set_submodule(model, name, new_layer)
    model.ori_weight_size = ori_weight_size.to(device)
    gc.collect()

    for p in model.parameters():
        p.requires_grad = False
    gammas = []
    for m in model.modules():
        if isinstance(m, SVDTransformLayer):
            m.gamma.requires_grad = True
            gammas.append(m.gamma)
    return gammas, gamma_inits


def _rank_windows(cfg: DictConfig, tokenizer):
    """Rank-selection samples (distinct from calibration when datasets coincide)."""
    rs = cfg.compress.dobi.rank_selection
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

    Mirrors ``coala_rank._rank_val_windows``: drawn from the dataset's
    **validation** split (the eval-stage / paper-table distribution) with a
    distinct seed, so they are disjoint from the training windows (train split)
    and the eval set (validation split, ``cfg.seed``). Empty when
    ``val_num_samples<=0`` (default — plain Dobi runs are unchanged).
    """
    rs = cfg.compress.dobi.rank_selection
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


class _DobiTrainWrapper(nn.Module):
    """Computes the full rank-selection loss (ppl + reg + value) in one forward.

    Wrapping the *entire* loss lets DDP wrap this module: every use of the trainable
    gamma params then lives inside a single forward graph, so DDP's reducer marks
    each gamma ready exactly once (computing reg/value outside the DDP forward
    triggers "marked a variable ready only once").
    """

    def __init__(self, model, cfg: DictConfig):
        super().__init__()
        self.model = model
        rs = cfg.compress.dobi.rank_selection
        self.target_ratio = float(cfg.compress.ratio)
        self.lambda_reg = float(rs.lambda_reg)
        self.seq_len = int(rs.seq_len)
        self._reset()

    def _reset(self):
        self._sum = {"ppl": 0.0, "reg": 0.0, "value": 0.0, "ratio": 0.0}
        self._n = 0

    def forward(self, input_ids):
        # Perplexity in float32: fp16 exp(loss) overflows to inf for loss > ~11,
        # which produces NaN gradients -> NaN gamma.
        loss = self.model(input_ids=input_ids, labels=input_ids).loss.float()
        ppl = torch.exp(loss)
        reg_loss, ratio = _compression_loss(
            self.model, self.target_ratio, self.lambda_reg, self.seq_len
        )
        value_loss = _wrong_value_loss(self.model, self.seq_len, input_ids.device)
        self._n += 1
        self._sum["ppl"] += float(ppl.detach())
        self._sum["reg"] += float(reg_loss.detach())
        self._sum["value"] += float(value_loss.detach())
        self._sum["ratio"] += float(ratio.detach())
        return ppl + reg_loss + value_loss

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
        directly comparable to the eval-stage / paper-table perplexity. The model
        forwards run through the soft-truncated SVD layers at the current gammas
        (no_grad, so no SVD backward graph). Empty ``windows`` -> nan."""
        if not len(windows):
            return float("nan")
        from pipeline import evaluate as evaluate_mod

        self.model.eval()
        input_ids = data_mod.ppl_input_ids(windows)
        return evaluate_mod.evaluate_perplexity(self.model, input_ids)


def _train_gammas(
    fwd_module,
    gammas,
    windows,
    rs,
    device,
    progress=True,
    on_epoch_end=None,
    desc="Dobi rank-train",
):
    """Train ``gammas`` on ``windows``. ``fwd_module(input_ids)`` returns the scalar
    total loss (a ``_DobiTrainWrapper`` or its DDP wrapper). ``rs`` is the
    rank-selection sub-config (Dobi or COALA). ``on_epoch_end(epoch)`` is invoked
    after each epoch (periodic checkpointing / validation); if it returns a truthy
    value training stops early. ``desc`` labels the progress bar.

    Returns ``(n_skipped, n_nonfinite_grad)`` diagnostics. Shared by dobi and
    coala_rank.
    """
    n_epochs = int(rs.n_epochs)
    lr = float(rs.lr)
    grad_accum = int(rs.get("gradient_accumulation_steps", 1))
    min_lr = float(rs.get("scheduler_min_lr", 4.0))

    optimizer = torch.optim.Adam(gammas, lr=lr)
    steps_per_epoch = max(1, math.ceil(len(windows) / grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch, eta_min=min_lr
    )

    n_skipped = 0
    n_nonfinite_grad = 0

    def _sanitize_grads():
        nonlocal n_nonfinite_grad
        for g in gammas:
            if g.grad is not None and not torch.isfinite(g.grad).all():
                n_nonfinite_grad += 1
                torch.nan_to_num_(g.grad, nan=0.0, posinf=0.0, neginf=0.0)

    for epoch in tqdm(range(n_epochs), desc=desc, unit="epoch", disable=not progress):
        optimizer.zero_grad()
        bar = tqdm(
            windows,
            desc=f"  epoch {epoch + 1}/{n_epochs}",
            unit="sample",
            leave=False,
            disable=not progress,
        )
        for i, w in enumerate(bar):
            ids = w.unsqueeze(0).to(device)
            total = fwd_module(input_ids=ids) / grad_accum
            if not torch.isfinite(total):
                n_skipped += 1
                optimizer.zero_grad()  # skip a bad step rather than poison gammas
                continue
            total.backward()
            if (i + 1) % grad_accum == 0:
                _sanitize_grads()  # NaN/inf grad -> 0 (no-op step, never NaN gamma)
                torch.nn.utils.clip_grad_norm_(gammas, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        _sanitize_grads()
        torch.nn.utils.clip_grad_norm_(gammas, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if on_epoch_end is not None and on_epoch_end(epoch):
            break  # early stop (e.g. validation perplexity plateaued)

    return n_skipped, n_nonfinite_grad


def _rank_checkpoint_path(cfg: DictConfig):
    return cfg.compress.dobi.rank_selection.get("checkpoint", None)


def _save_rank_checkpoint(path, gamma_dict: Dict[str, float], cfg: DictConfig) -> None:
    """Persist the selected ranks (per-layer gamma) + metadata as JSON."""
    import json
    from pathlib import Path

    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(cfg.model.id),
        "ratio": float(cfg.compress.ratio),
        "seq_len": int(cfg.compress.dobi.rank_selection.seq_len),
        "ranks": gamma_dict,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(p)  # atomic: a crash mid-write never corrupts the checkpoint


def _load_rank_checkpoint(path, cfg: DictConfig) -> Dict[str, float]:
    """Load ranks from a checkpoint; warn on model/ratio mismatch."""
    import json

    with open(str(path)) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "ranks" in payload:
        if str(payload.get("model")) != str(cfg.model.id):
            print(
                f"WARNING: rank checkpoint was made for {payload.get('model')!r}, "
                f"but model.id is {str(cfg.model.id)!r} — layer names may not match.",
                flush=True,
            )
        if float(payload.get("ratio", -1)) != float(cfg.compress.ratio):
            print(
                f"WARNING: rank checkpoint ratio {payload.get('ratio')} != "
                f"compress.ratio {float(cfg.compress.ratio)}.",
                flush=True,
            )
        return payload["ranks"]
    return payload  # legacy bare {name: gamma}


def _extract_gammas(model, gamma_inits, seq_len) -> Dict[str, float]:
    """Collect gammas, sanitising non-finite/out-of-range values."""
    from dobi.stable_svd import SVDTransformLayer

    gamma_dict = {}
    n_reset = 0
    for m in model.modules():
        if isinstance(m, SVDTransformLayer):
            g = float(m.gamma.detach().item())
            if not math.isfinite(g):
                g = gamma_inits[m.name]  # last-resort fallback (should not happen)
                n_reset += 1
            gamma_dict[m.name] = min(max(g, 1.0), float(seq_len))
    if n_reset:
        print(
            f"WARNING: {n_reset} non-finite gamma(s) reset to init — rank training "
            "diverged; check dtype / learning rate.",
            flush=True,
        )
    return gamma_dict


def _log_gamma_diag(gamma_dict, gamma_inits, n_skipped, n_nonfinite_grad):
    """Print a rank-training diagnostic so 'did the gammas actually move?' is visible."""
    moved = sum(1 for k, v in gamma_dict.items() if abs(v - gamma_inits[k]) > 1.0)
    distinct = len({round(v, 3) for v in gamma_dict.values()})
    mean_init = sum(gamma_inits.values()) / len(gamma_inits)
    mean_now = sum(gamma_dict.values()) / len(gamma_dict)
    print(
        "Dobi rank-train diagnostic: "
        f"layers={len(gamma_dict)} moved(>1)={moved} distinct={distinct} "
        f"mean_gamma init={mean_init:.2f}->now={mean_now:.2f} "
        f"skipped_steps={n_skipped} nonfinite_grad_steps={n_nonfinite_grad}",
        flush=True,
    )


class _EarlyStopper:
    """Validation-perplexity early stopping with best-gamma restore.

    ``patience<=0`` disables stopping (training runs all epochs) but the best
    snapshot is still tracked so the best-val ranks are restored at the end.
    Shared by dobi and coala_rank.
    """

    def __init__(self, gammas, patience, min_delta=0.0):
        self.gammas = list(gammas)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf
        self.best_snapshot = None
        self.n_bad = 0

    def update(self, val_ppl):
        """Record an epoch's val ppl. Returns ``(is_best, should_stop)``."""
        if not math.isfinite(val_ppl):
            return False, False
        is_best = val_ppl < self.best - self.min_delta
        if is_best:
            self.best = val_ppl
            self.best_snapshot = [g.detach().clone() for g in self.gammas]
            self.n_bad = 0
        else:
            self.n_bad += 1
        should_stop = self.patience > 0 and self.n_bad >= self.patience
        return is_best, should_stop

    def restore_best(self):
        """Copy the best-val gammas back into the live parameters (in place)."""
        if self.best_snapshot is None:
            return
        with torch.no_grad():
            for g, b in zip(self.gammas, self.best_snapshot):
                g.data.copy_(b)


def _epoch_log_line(epoch, means, val_ppl, is_best) -> str:
    """Per-epoch train+val perplexity line (with a ``[best]`` marker)."""
    mark = " [best]" if is_best else ""
    val = f"{val_ppl:.3f}" if math.isfinite(val_ppl) else "n/a"
    return (
        f"  epoch {epoch + 1}: train_ppl={means['ppl']:.3f} val_ppl={val}{mark} "
        f"reg={means['reg']:.3f} value={means['value']:.3f} ratio={means['ratio']:.3f}"
    )


def select_ranks(model, tokenizer, cfg: DictConfig, device) -> Dict[str, float]:
    """Single-GPU rank selection: train per-layer ``gamma``, return ``{name: gamma}``."""
    rs_cfg = cfg.compress.dobi.rank_selection
    # Rank training is fp32 by default (half backward through the SVD -> NaN
    # gammas). low_mem keeps the base model in its loaded dtype (bf16 required,
    # validated) — the SVDTransformLayer then runs its SVD/gate math in fp32
    # islands and checkpoints the layer forward, so a Qwen3-32B-sized model
    # fits a single GPU (131 GB fp32 -> ~66 GB bf16 + transients).
    if bool(rs_cfg.get("low_mem", False)):
        model = model.to(device)
    else:
        model = model.to(dtype=RANK_DTYPE, device=device)  # rank training is fp32
    gammas, gamma_inits = _attach_svd_layers(model, cfg, device, progress=True)
    rs = cfg.compress.dobi.rank_selection
    windows = _rank_windows(cfg, tokenizer)
    val_windows = _rank_val_windows(cfg, tokenizer)
    wrapper = _DobiTrainWrapper(model, cfg).to(device)

    seq_len = int(rs.seq_len)
    ckpt = _rank_checkpoint_path(cfg)
    every = int(rs.get("checkpoint_every", 0))
    stopper = _EarlyStopper(
        gammas,
        rs.get("early_stopping_patience", 0),
        rs.get("early_stopping_min_delta", 0.0),
    )

    if len(val_windows):
        ref = wrapper.val_perplexity(val_windows, device)
        print(f"  init-ranks reference (val): val_ppl={ref:.3f}", flush=True)

    def _on_epoch(epoch):
        means = wrapper.epoch_means()
        val_ppl = wrapper.val_perplexity(val_windows, device)
        is_best, stop = stopper.update(val_ppl)
        print(_epoch_log_line(epoch, means, val_ppl, is_best), flush=True)
        if ckpt and every > 0 and (epoch + 1) % every == 0:
            _save_rank_checkpoint(
                ckpt, _extract_gammas(model, gamma_inits, seq_len), cfg
            )
        if stop:
            print(
                f"  early stop: val_ppl no better for {stopper.patience} epoch(s); "
                f"best={stopper.best:.3f} — restoring best gammas.",
                flush=True,
            )
        return stop

    n_skipped, n_nonfinite = _train_gammas(
        wrapper,
        gammas,
        windows,
        rs,
        device,
        progress=True,
        on_epoch_end=_on_epoch,
    )
    stopper.restore_best()  # keep the best-val gammas, not the last epoch's
    gamma_dict = _extract_gammas(model, gamma_inits, seq_len)
    _log_gamma_diag(gamma_dict, gamma_inits, n_skipped, n_nonfinite)
    gc.collect()
    torch.cuda.empty_cache()
    return gamma_dict


# --------------------------------------------------------------------------- #
# Multi-GPU rank selection (DDP data parallelism over calibration samples)
# --------------------------------------------------------------------------- #
def _rank_worker(
    rank: int, world_size: int, cfg_container: dict, out_path: str
) -> None:
    """DDP worker: trains the shared gammas on a shard of samples; rank 0 saves."""
    from pipeline._paths import add_reference_paths

    add_reference_paths()  # vendored `dobi.*` must be importable in the child process

    cfg = OmegaConf.create(cfg_container)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _ddp_port())
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.model.id),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Same low_mem rule as the single-GPU path: keep the loaded (bf16) dtype so
    # each DDP replica fits its GPU; default stays the fp32 upcast.
    if bool(cfg.compress.dobi.rank_selection.get("low_mem", False)):
        from pipeline.model import resolve_dtype

        worker_dtype = resolve_dtype(cfg.model.get("load_dtype", "float32"))
    else:
        worker_dtype = RANK_DTYPE  # rank training is fp32
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.model.id),
        torch_dtype=worker_dtype,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    ).to(device)

    gammas, gamma_inits = _attach_svd_layers(model, cfg, device, progress=(rank == 0))
    from torch.nn.parallel import DistributedDataParallel as DDP

    wrapper = _DobiTrainWrapper(model, cfg).to(device)
    ddp = DDP(wrapper, device_ids=[rank], find_unused_parameters=False)

    rs = cfg.compress.dobi.rank_selection
    windows_all = _rank_windows(cfg, tokenizer)
    windows = windows_all[rank::world_size]  # each rank trains a disjoint shard
    # Validation runs on EVERY rank over the SAME windows (a Dobi val pass takes
    # minutes of per-layer SVD forwards; running it on rank 0 only would stall the
    # other ranks' first collective each epoch). Gammas are DDP-synced, so all
    # ranks compute the same value; rank 0's decision is broadcast regardless.
    val_windows = _rank_val_windows(cfg, tokenizer)

    seq_len = int(rs.seq_len)
    ckpt = _rank_checkpoint_path(cfg)
    every = int(rs.get("checkpoint_every", 0))
    stopper = _EarlyStopper(
        gammas,
        rs.get("early_stopping_patience", 0),
        rs.get("early_stopping_min_delta", 0.0),
    )

    if len(val_windows):
        ref = wrapper.val_perplexity(val_windows, device)
        if rank == 0:
            print(f"  init-ranks reference (val): val_ppl={ref:.3f}", flush=True)

    def _on_epoch(epoch):
        means = wrapper.epoch_means()  # rank-local shard means (rank 0 is logged)
        val_ppl = wrapper.val_perplexity(val_windows, device)
        is_best, stop = stopper.update(val_ppl)
        if rank == 0:
            print(_epoch_log_line(epoch, means, val_ppl, is_best), flush=True)
            if ckpt and every > 0 and (epoch + 1) % every == 0:
                _save_rank_checkpoint(
                    ckpt, _extract_gammas(model, gamma_inits, seq_len), cfg
                )
            if stop:
                print(
                    f"  early stop: val_ppl no better for {stopper.patience} "
                    f"epoch(s); best={stopper.best:.3f} — restoring best gammas.",
                    flush=True,
                )
        stop_flag = torch.tensor(
            [1.0 if stop else 0.0], device=device
        )  # rank 0 decides; broadcast so all ranks break at the same epoch
        dist.broadcast(stop_flag, src=0)
        return bool(stop_flag.item())

    n_skipped, n_nonfinite = _train_gammas(
        ddp,
        gammas,
        windows,
        rs,
        device,
        progress=(rank == 0),
        on_epoch_end=_on_epoch,
    )

    dist.barrier()
    if rank == 0:
        stopper.restore_best()  # keep the best-val gammas, not the last epoch's
        gamma_dict = _extract_gammas(model, gamma_inits, seq_len)
        _log_gamma_diag(gamma_dict, gamma_inits, n_skipped, n_nonfinite)
        torch.save(gamma_dict, out_path)
    dist.barrier()
    dist.destroy_process_group()


def _ddp_port() -> str:
    """A rendezvous port unique per SLURM job (fallback: per process).

    A fixed port collides with EADDRINUSE when two multi-GPU dobi jobs land on
    the same node; deriving it from the job id keeps concurrent jobs apart.
    """
    seed = int(os.environ.get("SLURM_JOB_ID", str(os.getpid())))
    return str(29500 + seed % 2000)


def select_ranks_multigpu(cfg: DictConfig, world_size: int) -> Dict[str, float]:
    """Spawn ``world_size`` DDP workers for rank selection; return the gamma dict."""
    container = OmegaConf.to_container(cfg, resolve=True)
    out_path = os.path.join(tempfile.gettempdir(), f"dobi_gamma_{os.getpid()}.pt")
    # Fix the rendezvous address in the parent so every spawned worker inherits
    # the same (job-unique) port.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _ddp_port())
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


def _compression_loss(model, target_ratio, lambda_reg, seq_len):
    """|achieved_ratio - target| * lambda_reg  (no-remapping variant)."""
    from dobi.stable_svd import SVDTransformLayer

    device = model.ori_weight_size.device
    size_new = torch.tensor(0.0, device=device)
    for module in model.modules():
        if isinstance(module, SVDTransformLayer):
            in_f, out_f = module.ori.in_features, module.ori.out_features
            rank_ratio = min(in_f, out_f) / seq_len
            size_now = (in_f + out_f) * module.gamma * rank_ratio
            size_ori = module.ori_weight_size.to(device)
            size_new = torch.where(size_now < size_ori, size_now, size_ori) + size_new
    ratio = size_new / model.ori_weight_size
    loss = torch.abs(ratio - torch.tensor(target_ratio, device=ratio.device))
    return lambda_reg * loss, ratio


def _wrong_value_loss(model, seq_len, device):
    """Penalise gamma outside [0, seq_len]."""
    from dobi.stable_svd import SVDTransformLayer

    penalty = torch.tensor(0.0, device=device)
    for module in model.modules():
        if isinstance(module, SVDTransformLayer):
            lower = torch.relu(-module.gamma) ** 2
            upper = (
                torch.relu(module.gamma - torch.tensor(float(seq_len), device=device))
                ** 2
            )
            penalty = penalty + lower + upper
    return penalty


# --------------------------------------------------------------------------- #
# Stage 2: IPCA weight update (compression)
# --------------------------------------------------------------------------- #
def compress_with_ipca(
    cfg: DictConfig, tokenizer, gamma_dict: Dict[str, float], device
):
    """Apply the chosen ranks via IncrementalPCA weight update; return the model."""
    from dobi.incremental_pca import IncrementalPCAonGPU
    from dobi.module import SVDTransformLayer as FactoredLayer

    rs = cfg.compress.dobi.rank_selection
    seq_len = int(rs.seq_len)
    beta = float(rs.beta)
    no_svd = _no_svd_names(str(cfg.model.id))
    calib = cfg.calibration

    # IPCA-stage model dtype: honor model.load_dtype instead of hardcoding fp16.
    # bf16-native models with large activation outliers (Qwen3) overflow fp16
    # during the IPCA calibration forwards — inf activations poison the PCA and
    # yield garbage weights (observed: ppl ~6.4M on Qwen3-0.6B vs 20.5 val ppl
    # at the same ranks during fp32 rank training). Default stays float16, so
    # existing Mistral/Llama runs are unchanged; pass model.load_dtype=bfloat16
    # for Qwen3.
    from pipeline.model import resolve_dtype

    model_dtype = resolve_dtype(cfg.model.get("load_dtype", "float16"))
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.model.id),
        torch_dtype=model_dtype,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    ).to(device)
    for p in model.parameters():
        p.requires_grad = False

    pca_components: Dict[str, torch.Tensor] = {}
    pca_state: Dict[str, object] = {}
    v_accum: Dict[str, list] = {}

    def svd_forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(0)
        A = x.to(COMPUTE_DTYPE) @ self.weight.to(COMPUTE_DTYPE).T
        gamma_range = max(1, math.ceil(self.gamma))
        real = min(A.shape[0], A.shape[1], gamma_range)
        Uf, Sf, Vf = torch.svd_lowrank(A, q=int(real), niter=2)
        seq = torch.arange(1, len(Sf) + 1, device=A.device, dtype=A.dtype)
        trunc = 0.5 * torch.tanh(self.BETA * (real - seq)) + 0.5
        x_t = (Uf @ torch.diag_embed(Sf * trunc) @ Vf.T).to(model_dtype)

        v_accum[self.name].append(Vf.detach().to("cpu"))
        if len(v_accum[self.name]) == self.Ngamma:
            vf_full = torch.cat(v_accum[self.name][: self.Ngamma], dim=1).to(device)
            pca_state[self.name].partial_fit(vf_full.T)
            pca_components[self.name] = (
                pca_state[self.name].components_.detach().to("cpu")
            )
            v_accum[self.name] = []
            del vf_full
            gc.collect()
            torch.cuda.empty_cache()
        return x_t

    # Patch forwards + set up per-layer IPCA accumulators. Only compress layers we
    # have a rank for — lets external rank sets (e.g. COALA's, which omit down_proj)
    # drive Dobi without KeyError-ing on the layers they don't cover.
    targets = [
        (n, m)
        for n, m in model.named_modules()
        if _is_target(n, m, no_svd) and n in gamma_dict
    ]
    if bool(rs.get("ranks_are_effective", False)):
        # External ranks (e.g. from COALA) are the *effective* low-rank dimension;
        # Dobi's gammas are per-(seq_len)-block (the factored rank is cut*ceil(gamma)),
        # so divide by cut so cut*ceil(gamma) reproduces the requested effective rank.
        for name, module in targets:
            cut = max(1, int(min(module.in_features, module.out_features) / seq_len))
            gamma_dict[name] = float(gamma_dict[name]) / cut
    for name, module in tqdm(targets, desc="Dobi: attach IPCA hooks", unit="layer"):
        module.register_buffer(
            "gamma", torch.tensor(gamma_dict[name], dtype=COMPUTE_DTYPE)
        )
        module.register_buffer("BETA", torch.tensor(beta))
        module.name = name
        m_out, n_in = module.weight.shape
        cut = max(1, int(min(module.in_features, module.out_features) / seq_len))
        module.Ngamma = cut
        gamma_eff = (
            cut * math.ceil(gamma_dict[name])
            if cut > 1
            else math.ceil(gamma_dict[name])
        )
        n_comp = int(min(m_out, n_in, max(1, gamma_eff)))
        pca_state[name] = IncrementalPCAonGPU(device=device, n_components=n_comp)
        v_accum[name] = []
        module.forward = svd_forward.__get__(module, nn.Linear)

    # Collect IPCA stats over the calibration set.
    windows = data_mod.make_token_windows(
        tokenizer=tokenizer,
        name=str(calib.dataset),
        split="train",
        seq_len=seq_len,
        num_samples=int(calib.num_samples),
        seed=int(cfg.get("seed", 0)),
        dataset_cache_dir=calib.get("dataset_cache_dir", "./datasets"),
    )
    with torch.no_grad():
        for w in tqdm(windows, desc="Dobi IPCA: collect activations", unit="sample"):
            model(input_ids=w.unsqueeze(0).to(device))

    def change_weight(gamma, W, name):
        W = W.to(COMPUTE_DTYPE)
        v_pca = pca_components[name].T.to(device).to(COMPUTE_DTYPE)  # [out, n_comp]
        seq = torch.arange(1, v_pca.shape[1] + 1, device=device)
        m_out, n_in = W.shape
        real = min(m_out, n_in, max(1, gamma))
        trunc = 0.5 * torch.tanh(beta * (real - seq)) + 0.5
        G = torch.diag(trunc)
        return (W.T @ v_pca @ G @ v_pca.T).T.to(model_dtype)

    # Project weights and factor each into two low-rank Linears.
    for name, module in tqdm(
        targets, desc="Dobi: update + factor weights", unit="layer"
    ):
        cut = max(1, int(min(module.in_features, module.out_features) / seq_len))
        gamma_w = cut * gamma_dict[name] if cut > 1 else gamma_dict[name]
        new_weight = change_weight(gamma_w, module.weight.data.detach(), name)
        rank = (
            int(cut * math.ceil((cut * gamma_dict[name]) / cut))
            if cut > 1
            else int(math.ceil(gamma_dict[name]))
        )
        rank = max(1, min(rank, *new_weight.shape))
        factored = FactoredLayer(
            gamma=rank,
            weight=new_weight,
            bias=module.bias,
            name=name,
            device=device,
        )
        _set_submodule(model, name, factored)
        del new_weight
        gc.collect()
        torch.cuda.empty_cache()

    return model


def run_dobi(model, tokenizer, cfg: DictConfig):
    """Full Dobi pipeline: select ranks, then compress. Returns (model, gamma_dict).

    Rank checkpointing: if ``compress.dobi.rank_selection.checkpoint`` exists and
    ``resume`` is true (default), the saved per-layer ranks are loaded and the
    (expensive) rank-selection stage is skipped entirely; otherwise the freshly
    selected ranks are written there. ``checkpoint_every`` also saves mid-training.

    Only the rank-selection stage scales across GPUs: with more than one GPU
    visible (``#SBATCH --gpus``) it runs DDP data-parallel over the calibration
    samples. The IPCA compression stage stays single-GPU (cuda:0).
    """
    device = torch.device(str(cfg.model.get("device", "cuda")))
    rs = cfg.compress.dobi.rank_selection
    ckpt = _rank_checkpoint_path(cfg)
    resume = bool(rs.get("resume", True))
    have_ckpt = bool(ckpt) and resume and os.path.exists(str(ckpt))

    n_gpus = torch.cuda.device_count()
    n_samples = int(rs.num_samples)
    world_size = min(n_gpus, n_samples) if n_gpus > 1 else 1

    # The passed (cuda:0) model is only used for single-GPU rank training. When we
    # resume from a checkpoint or train multi-GPU (workers reload), free it so the
    # IPCA reload doesn't double GPU memory.
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
            f"Dobi: loading ranks from checkpoint {ckpt} (skip rank selection)",
            flush=True,
        )
        gamma_dict = _load_rank_checkpoint(ckpt, cfg)
    elif world_size > 1:
        print(f"Dobi rank selection: DDP across {world_size} GPUs", flush=True)
        gamma_dict = select_ranks_multigpu(cfg, world_size)
    else:
        gamma_dict = select_ranks(model, tokenizer, cfg, device)

    if ckpt and not have_ckpt:  # persist freshly selected ranks
        _save_rank_checkpoint(ckpt, gamma_dict, cfg)
        print(f"Dobi: saved ranks to checkpoint {ckpt}", flush=True)

    if bool(cfg.compress.dobi.get("stop_after_ranks", False)):
        print("Dobi: stop_after_ranks=true — skipping IPCA compression.", flush=True)
        return None, gamma_dict

    compressed = compress_with_ipca(cfg, tokenizer, gamma_dict, device)
    return compressed, gamma_dict
