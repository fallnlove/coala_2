"""'empty' strategy exercises inject_coala on CPU (no gesvd / .cuda())."""

import math

import torch
from omegaconf import OmegaConf

from pipeline import compress as compress_mod
from pipeline import model as model_mod

from pipeline._paths import add_reference_paths

add_reference_paths()
from coala.COALA import COALA_Layer  # noqa: E402


def _empty_config(cpu_base_config, tmp_path):
    cfg = cpu_base_config(tmp_path)
    cfg["model"]["device"] = "cpu"
    cfg["compress"] = {
        "enabled": True,
        "strategy": "empty",
        "ratio": 0.5,
        "fp16": False,
        "adaptive_rank": False,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "params": {},
        "save": {"enabled": False},
    }
    cfg["eval"]["perplexity"]["enabled"] = False
    return OmegaConf.create(cfg)


def test_empty_injects_coala_layers(cpu_base_config, tmp_path):
    cfg = _empty_config(cpu_base_config, tmp_path)
    model, tokenizer = model_mod.load_model_and_tokenizer(cfg)
    model, ranks = compress_mod.maybe_compress(model, tokenizer, cfg)

    # ranks is the per-module ratio dict produced by inject_coala.
    assert isinstance(ranks, dict) and ranks

    coala_layers = [
        (name, m) for name, m in model.named_modules() if isinstance(m, COALA_Layer)
    ]
    assert coala_layers, "expected target Linears replaced by COALA_Layer"

    name, layer = coala_layers[0]
    adapter = layer.adapter
    m, n = adapter.adapter_B.shape[1], adapter.adapter_A.shape[0]
    r = adapter.r
    expected_r = max(1, min(m, n, math.ceil(0.5 * m * n / (m + n))))
    assert r == expected_r
    assert adapter.adapter_A.shape == (n, r)
    assert adapter.adapter_B.shape == (r, m)


def test_empty_layer_outputs_zeros(cpu_base_config, tmp_path):
    cfg = _empty_config(cpu_base_config, tmp_path)
    model, tokenizer = model_mod.load_model_and_tokenizer(cfg)
    model, _ = compress_mod.maybe_compress(model, tokenizer, cfg)

    layer = next(m for m in model.modules() if isinstance(m, COALA_Layer))
    n = layer.adapter.adapter_A.shape[0]
    x = torch.randn(1, 4, n, dtype=layer.adapter.adapter_A.dtype)
    out = layer(x)
    assert torch.allclose(out, torch.zeros_like(out))
