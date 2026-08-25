"""Compression orchestration: drive COALA's hook/inject flow directly.

COALA normally collects calibration activations by running an HF ``Trainer`` over
a pre-generated ``./datasets/*`` dataset (COALA/coala/Config.py:compress_model).
We replace that with an explicit ``torch.no_grad()`` forward loop over tokenized
calibration windows, then reuse COALA's own functions to inject the low-rank
adapters:

    prepare_get_samples -> [forward passes] -> after_get_samples -> inject_coala

Notes baked in from reading the COALA source:

* ``empty``/``svd`` need no activations. ``prepare_get_samples`` would set
  ``config.dic = {}`` which then KeyErrors in ``process_layer`` (it does
  ``torch.cat(dic[name])``). So for those we call ``inject_coala`` directly.
* ``inject_coala`` mutates ``config.ratio`` from a float into a per-module dict
  (the returned ``ranks``) and calls ``torch.set_num_threads(1)``; we restore the
  thread count afterwards.
* The forward loop must feed 2-D ``[batch, seq]`` input_ids so the pre-hook's
  ``input[0].permute(2, 1, 0)`` sees ``[batch, seq, features]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from pipeline import data as data_mod
from pipeline.config import SAMPLE_FREE_STRATEGIES, build_coala_config


def _collect_activations(
    model, tokenizer, cfg: DictConfig, coala_config, device
) -> None:
    """Register COALA hooks, run calibration forward passes, finalize activations."""
    from coala.Injector import after_get_samples, prepare_get_samples

    calib = cfg.calibration
    windows = data_mod.make_token_windows(
        tokenizer=tokenizer,
        name=str(calib.dataset),
        split="train",
        seq_len=int(calib.seq_len),
        num_samples=int(calib.num_samples),
        seed=int(cfg.get("seed", 0)),
        dataset_cache_dir=calib.get("dataset_cache_dir", "./datasets"),
    )

    batch_size = int(calib.batch_size)
    n_batches = (len(windows) + batch_size - 1) // batch_size
    model, hooks = prepare_get_samples(model, coala_config)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            data_mod.calibration_batches(windows, batch_size),
            total=n_batches,
            desc="Calibration forward",
            unit="batch",
        ):
            model(input_ids=batch.to(device))
    after_get_samples(model, coala_config, hooks)


def _learned_coala_ranks(model, tokenizer, cfg, coala_config):
    """COALA with non-uniform learned/loaded ranks.

    Resolve the per-layer integer ranks (train or checkpoint; may consume/mutate
    the passed model), then reload a fresh model and inject COALA at those ranks
    (converted to the midpoint per-layer ratio). Returns
    ``(model_or_None, ranks, coala_config)``; model is ``None`` for stop_after_ranks.
    """
    from coala.Injector import inject_coala

    from pipeline import coala_rank
    from pipeline.model import resolve_dtype

    device = next(model.parameters()).device
    rank_dict = coala_rank.resolve_ranks(model, tokenizer, cfg, device)

    if bool(cfg.compress.get("stop_after_ranks", False)):
        print("COALA: stop_after_ranks=true — skipping compression.", flush=True)
        return None, rank_dict, coala_config

    # Free the rank-selection model (mutated/on-cpu) before reloading for compression.
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.model.id),
        torch_dtype=resolve_dtype(cfg.model.get("load_dtype", "float32")),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    ).to(device)
    ratio_dict = coala_rank._ranks_to_ratio_dict(model, rank_dict)

    prev_threads = torch.get_num_threads()
    try:
        _collect_activations(model, tokenizer, cfg, coala_config, device)
        ranks, model = inject_coala(coala_config, model, ranks=ratio_dict)
    finally:
        torch.set_num_threads(prev_threads)
    return model, ranks, coala_config


def run_compression(
    model, tokenizer, cfg: DictConfig
) -> Tuple[Optional[torch.nn.Module], dict, object]:
    """Compress ``model`` and return ``(model, ranks, coala_config)``.

    ``model`` is ``None`` when ``compress.stop_after_ranks`` short-circuits after
    learned COALA rank selection.
    """
    from coala.Injector import inject_coala

    coala_config = build_coala_config(cfg)
    strategy = coala_config.compress_strategy
    device = next(model.parameters()).device

    rank_source = str(cfg.compress.get("rank_source", "uniform"))

    # Swift-SVD dynamic allocation (swift | checkpoint): training-free, so it runs
    # in a single pass on THIS model and reuses the collected activations for the
    # final injection — no reload (contrast _learned_coala_ranks below).
    if strategy == "swift_svd" and rank_source != "uniform":
        from pipeline import swift_rank

        return swift_rank.run_swift(model, tokenizer, cfg, coala_config)

    # Learned non-uniform COALA ranks (train | checkpoint). Uniform = unchanged.
    if rank_source != "uniform":
        return _learned_coala_ranks(model, tokenizer, cfg, coala_config)

    prev_threads = torch.get_num_threads()
    try:
        if strategy not in SAMPLE_FREE_STRATEGIES:
            _collect_activations(model, tokenizer, cfg, coala_config, device)
        # For empty/svd we deliberately skip activation collection so that
        # process_layer sees X=None (no config.dic attribute).
        ranks, model = inject_coala(coala_config, model)
    finally:
        # inject_coala forces single-thread mode; restore it for later CPU work.
        torch.set_num_threads(prev_threads)

    return model, ranks, coala_config


def maybe_compress(
    model, tokenizer, cfg: DictConfig
) -> Tuple[torch.nn.Module, Optional[dict]]:
    """Compress if enabled, else return the model unchanged (baseline).

    Returns ``(model, ranks)`` where ``ranks`` is ``None`` for the baseline.
    """
    compress = cfg.get("compress", {})
    if not bool(compress.get("enabled", False)):
        return model, None

    if str(compress.get("strategy")) == "dobi":
        from pipeline import dobi

        model, gamma_dict = dobi.run_dobi(model, tokenizer, cfg)
        return model, gamma_dict

    model, ranks, _ = run_compression(model, tokenizer, cfg)
    return model, ranks
