"""Baseline (no compression) + perplexity end-to-end on CPU."""

import math

import torch
from omegaconf import OmegaConf

from pipeline import compress as compress_mod
from pipeline import evaluate as evaluate_mod
from pipeline import model as model_mod


def test_baseline_perplexity_finite(cpu_base_config, tmp_path):
    cfg = OmegaConf.create(cpu_base_config(tmp_path))
    model, tokenizer = model_mod.load_model_and_tokenizer(cfg)

    model, ranks = compress_mod.maybe_compress(model, tokenizer, cfg)
    assert ranks is None  # baseline path returns no ranks

    model = model_mod.cast_for_inference(model, cfg)
    results = evaluate_mod.run_eval(model, tokenizer, cfg)

    ppl = results["perplexity"]["wikitext2"]
    assert isinstance(ppl, float)
    assert math.isfinite(ppl) and ppl > 0
    assert "reasoning" not in results  # disabled in the CPU config


def test_perplexity_helper_directly():
    # Deterministic tiny check: random model, ensure shape handling/finiteness.
    from tests.conftest import TINY_MODEL
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    ppl = evaluate_mod.evaluate_perplexity(model, input_ids, limit=2)
    assert math.isfinite(ppl)
