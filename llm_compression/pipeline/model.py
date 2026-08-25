"""HuggingFace model/tokenizer loading and (de)serialization helpers.

Reimplemented (rather than reusing COALA/lib/Models.py) so we control dtype and
device explicitly: COALA hardcodes ``device_map='cuda'``, which we avoid because
accelerate device-map hooks interfere with in-place layer injection. We instead
load in a plain ``torch_dtype`` and ``.to(device)``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.config import to_plain

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[str(name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dtype {name!r}; expected one of {sorted(_DTYPES)}"
        ) from exc


def load_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """Load model + tokenizer per ``cfg.model``.

    The model is loaded in ``model.load_dtype`` (recommended float32 for the
    compression pass) and moved to ``model.device``.
    """
    model_cfg = cfg.model
    model_id = str(model_cfg.id)
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    tokenizer_kwargs = to_plain(model_cfg.get("tokenizer", None), default={}) or {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, **tokenizer_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_dtype = resolve_dtype(model_cfg.get("load_dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=load_dtype, trust_remote_code=trust_remote_code
    )
    model.to(torch.device(str(model_cfg.get("device", "cpu"))))
    model.eval()
    return model, tokenizer


def cast_model(model: torch.nn.Module, dtype, device) -> torch.nn.Module:
    """Cast a model to ``dtype`` (name or ``torch.dtype``) on ``device``."""
    if not isinstance(dtype, torch.dtype):
        dtype = resolve_dtype(dtype)
    model.to(dtype=dtype, device=torch.device(str(device)))
    model.eval()
    return model


def cast_for_inference(model: torch.nn.Module, cfg: DictConfig) -> torch.nn.Module:
    """Cast the (possibly compressed) model to the default inference dtype/device.

    Per-eval-phase dtypes (e.g. fp32 perplexity, fp16 reasoning) are handled in
    ``pipeline.evaluate.run_eval``; this just applies ``model.inference_dtype``.
    """
    return cast_model(
        model,
        cfg.model.get("inference_dtype", "float16"),
        cfg.model.get("device", "cpu"),
    )


def save_compressed(model, tokenizer, ranks, cfg: DictConfig) -> str:
    """Persist the compressed model, tokenizer, and per-layer rank dict.

    Mirrors COALA/compress.py: ``save_pretrained`` + ``ranks.pkl``.
    """
    save_cfg = cfg.compress.get("save", {})
    out_dir = Path(str(save_cfg.get("output_dir", "./outputs/compressed")))
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        str(out_dir), max_shard_size=str(save_cfg.get("max_shard_size", "1GB"))
    )
    tokenizer.save_pretrained(str(out_dir))
    if bool(save_cfg.get("save_ranks", True)):
        with open(out_dir / "ranks.pkl", "wb") as f:
            pickle.dump(ranks, f)
    return str(out_dir)
