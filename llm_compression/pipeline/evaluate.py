"""Evaluation: perplexity (from Dobi-SVD) + reasoning accuracy (from COALA).

* ``evaluate_perplexity`` reimplements Dobi-SVD/evaluate.py:evaluate_perplexity,
  but uses ``model(...).logits`` explicitly (rather than ``[0]``) and guards the
  CUDA cache flush so it also runs on CPU.
* ``evaluate_reasoning`` reuses COALA's approach (COALA/valid.py): wrap the model
  in lm-eval's ``HFLM`` and call ``lm_eval.simple_evaluate``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from pipeline import data as data_mod


@torch.no_grad()
def evaluate_perplexity(
    model, input_ids: torch.Tensor, limit: int | None = None
) -> float:
    """Perplexity over ``input_ids`` of shape ``[num_samples, seq_len]``.

    PPL = exp(sum(token CE losses) / (num_used_samples * seq_len)).
    """
    nsamples, seqlen = input_ids.size()
    nlls = []
    for i in tqdm(range(nsamples), desc="perplexity"):
        if limit is not None and i == limit:
            break
        ids = input_ids[i : i + 1, :-1].to(model.device)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss)

    if not nlls:
        return float("nan")
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl.item()


def evaluate_perplexity_all(model, tokenizer, cfg: DictConfig) -> Dict[str, float]:
    """Compute perplexity per configured dataset.

    Datasets listed in ``eval.perplexity.contiguous`` use the standard
    whole-corpus contiguous protocol on the ``contiguous_split`` (default
    ``test``) — apples-to-apples with published numbers, used for small corpora
    like PTB. Everything else uses the fast sampled-window path on ``validation``.
    """
    ppl_cfg = cfg.eval.perplexity
    results: Dict[str, float] = {}
    cache_dir = Path(str(cfg.calibration.get("dataset_cache_dir", "./datasets")))
    seq_len = int(ppl_cfg.seq_len)
    contiguous = set(ppl_cfg.get("contiguous", []) or [])
    cont_split = str(ppl_cfg.get("contiguous_split", "test"))
    max_chunks = ppl_cfg.get("max_chunks", None)

    for name in ppl_cfg.datasets:
        name = str(name)
        if name in contiguous:
            input_ids = data_mod.contiguous_chunks(
                tokenizer=tokenizer,
                name=name,
                split=cont_split,
                seq_len=seq_len,
                dataset_cache_dir=cache_dir,
                max_chunks=max_chunks,
            )
            limit = None
        else:
            windows = data_mod.make_token_windows(
                tokenizer=tokenizer,
                name=name,
                split="validation",
                seq_len=seq_len,
                num_samples=int(ppl_cfg.num_samples),
                seed=int(cfg.get("seed", 0)),
                dataset_cache_dir=cache_dir,
            )
            input_ids = data_mod.ppl_input_ids(windows)
            limit = int(ppl_cfg.num_samples)
        results[name] = evaluate_perplexity(model, input_ids, limit=limit)
    return results


def evaluate_reasoning(model, tokenizer, cfg: DictConfig) -> Dict:
    """Run lm-eval common-sense-reasoning tasks on the (compressed) model."""
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    reasoning_cfg = cfg.eval.reasoning
    batch_size = int(reasoning_cfg.get("batch_size", 8))
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        hflm,
        tasks=list(reasoning_cfg.tasks),
        num_fewshot=int(reasoning_cfg.get("num_fewshot", 0)),
        batch_size=batch_size,
    )
    return results["results"]


def run_eval(model, tokenizer, cfg: DictConfig) -> Dict:
    """Assemble all enabled evaluations, casting the model per phase.

    Perplexity runs in float32 by default (fp16 log_softmax overflows → inf PPL);
    reasoning runs in float16 (or ``model.inference_dtype``) by default. Both are
    overridable via ``eval.<phase>.dtype``.
    """
    from pipeline.model import cast_model

    out: Dict = {}
    eval_cfg = cfg.get("eval", {})
    device = cfg.model.get("device", "cpu")
    inference_dtype = cfg.model.get("inference_dtype", None)

    ppl_cfg = eval_cfg.get("perplexity", {})
    if ppl_cfg.get("enabled", False):
        # Do NOT fall back to inference_dtype here: fp16 is what poisons PPL.
        cast_model(model, ppl_cfg.get("dtype", None) or "float32", device)
        out["perplexity"] = evaluate_perplexity_all(model, tokenizer, cfg)

    reasoning_cfg = eval_cfg.get("reasoning", {})
    if reasoning_cfg.get("enabled", False):
        dtype = reasoning_cfg.get("dtype", None) or inference_dtype or "float16"
        cast_model(model, dtype, device)
        out["reasoning"] = evaluate_reasoning(model, tokenizer, cfg)

    return out
