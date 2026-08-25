"""GPU-only smoke: real calibration -> inject -> perplexity path."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from pipeline import compress as compress_mod
from pipeline import evaluate as evaluate_mod
from pipeline import model as model_mod

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_svd_llm_compress_and_ppl(tmp_path):
    from tests.conftest import TINY_MODEL

    cfg = OmegaConf.create(
        {
            "model": {
                "id": TINY_MODEL,
                "load_dtype": "float32",
                "inference_dtype": "float16",
                "device": "cuda",
                "trust_remote_code": False,
                "tokenizer": {"use_fast": True},
            },
            "compress": {
                "enabled": True,
                "strategy": "svd_llm",
                "ratio": 0.5,
                "fp16": False,
                "adaptive_rank": False,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "params": {"accumulate": 0},
                "save": {"enabled": False},
            },
            "calibration": {
                "dataset": "wikitext2",
                "seq_len": 32,
                "num_samples": 4,
                "batch_size": 2,
                "dataset_cache_dir": str(tmp_path / "datasets"),
                "data_cache_dir": str(tmp_path / "data_cache"),
            },
            "eval": {
                "perplexity": {
                    "enabled": True,
                    "datasets": ["wikitext2"],
                    "seq_len": 32,
                    "num_samples": 4,
                    "batch_size": 1,
                },
                "reasoning": {"enabled": False, "tasks": ["piqa"]},
            },
            "seed": 0,
            "results_path": str(tmp_path / "results.json"),
        }
    )

    model, tokenizer = model_mod.load_model_and_tokenizer(cfg)
    n_before = sum(p.numel() for p in model.parameters())

    model, ranks = compress_mod.maybe_compress(model, tokenizer, cfg)
    assert isinstance(ranks, dict) and ranks
    n_after = sum(p.numel() for p in model.parameters())
    assert n_after < n_before

    model = model_mod.cast_for_inference(model, cfg)
    results = evaluate_mod.run_eval(model, tokenizer, cfg)
    assert math.isfinite(results["perplexity"]["wikitext2"])


def _tiny_cfg(tmp_path, **compress_overrides):
    from tests.conftest import TINY_MODEL

    compress = {
        "enabled": True,
        "strategy": "swift_svd",
        "ratio": 0.5,
        "fp16": False,
        "adaptive_rank": False,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "params": {"accumulate": 0},
        "save": {"enabled": False},
    }
    compress.update(compress_overrides)
    return OmegaConf.create(
        {
            "model": {
                "id": TINY_MODEL,
                "load_dtype": "float32",
                "inference_dtype": "float16",
                "device": "cuda",
                "trust_remote_code": False,
                "tokenizer": {"use_fast": True},
            },
            "compress": compress,
            "calibration": {
                "dataset": "wikitext2",
                "seq_len": 32,
                "num_samples": 4,
                "batch_size": 2,
                "dataset_cache_dir": str(tmp_path / "datasets"),
                "data_cache_dir": str(tmp_path / "data_cache"),
            },
            "eval": {
                "perplexity": {
                    "enabled": True,
                    "datasets": ["wikitext2"],
                    "seq_len": 32,
                    "num_samples": 4,
                    "batch_size": 1,
                },
                "reasoning": {"enabled": False, "tasks": ["piqa"]},
            },
            "seed": 0,
            "results_path": str(tmp_path / "results.json"),
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_swift_operator_equals_coala_mu0():
    """Correctness oracle: Swift-SVD's closed form is COALA's unregularised
    operator, so the two solvers must agree on identical inputs (both reduce to
    U_r U_rᵀ W with U from W @ whiten(X)). GPU-only: COALA's solvers hardcode the
    cuSOLVER gesvd driver."""
    from pipeline._paths import add_reference_paths

    add_reference_paths()
    from lib import Solvers

    torch.manual_seed(0)
    W = torch.randn(64, 48, device="cuda", dtype=torch.float32)
    X = torch.randn(48, 200, device="cuda", dtype=torch.float32)
    r = 12

    B_s, A_s = Solvers.swift_svd_method(W, X, r)
    B_c, A_c = Solvers.weight_approx(W, X, r)
    # Compare the reconstructions, not the factors: singular vectors are only
    # defined up to sign/rotation within equal singular values.
    torch.testing.assert_close(B_s @ A_s, B_c @ A_c, rtol=1e-4, atol=1e-4)

    err_s = torch.linalg.norm((W - B_s @ A_s) @ X).item()
    sv = torch.linalg.svdvals(W @ X)
    assert err_s == pytest.approx(
        torch.sqrt((sv[r:] ** 2).sum()).item(), rel=1e-4, abs=1e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_swift_svd_uniform_compress_and_ppl(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    model, tokenizer = model_mod.load_model_and_tokenizer(cfg)
    n_before = sum(p.numel() for p in model.parameters())

    model, ranks = compress_mod.maybe_compress(model, tokenizer, cfg)
    assert isinstance(ranks, dict) and ranks
    assert sum(p.numel() for p in model.parameters()) < n_before

    model = model_mod.cast_for_inference(model, cfg)
    results = evaluate_mod.run_eval(model, tokenizer, cfg)
    assert math.isfinite(results["perplexity"]["wikitext2"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_swift_svd_dynamic_allocation_end_to_end(tmp_path):
    """Algorithm 2 + alpha grid on a tiny model: ranks are written, the budget is
    conserved per module type, and the compressed model still evaluates."""
    import json

    from pipeline import swift_rank

    ckpt = tmp_path / "ranks.json"
    cfg = _tiny_cfg(
        tmp_path,
        rank_source="swift",
        swift_rank_selection={
            "delta": 0.5,
            "alphas": [0.0, 0.5, 1.0],
            "score": "tail",
            "base_rank": "delta_kbar",
            "val": {"dataset": "wikitext2", "num_samples": 2, "seq_len": 32},
            "importance": {
                "dataset": "wikitext2",
                "num_samples": 2,
                "seq_len": 32,
                "angular": False,
            },
            "cache_dtype": "float32",
            "checkpoint": str(ckpt),
        },
    )

    model, tokenizer = model_mod.load_model_and_tokenizer(cfg)
    infos = swift_rank._layer_infos(model, cfg.compress.target_modules)
    model, ranks = compress_mod.maybe_compress(model, tokenizer, cfg)
    assert isinstance(ranks, dict) and ranks

    payload = json.loads(ckpt.read_text())
    saved = payload["ranks"]
    assert saved and all(isinstance(v, int) and v >= 1 for v in saved.values())
    for (_mtype, m, n), group in swift_rank._group_infos(infos).items():
        budget = swift_rank._kbar(m, n, 0.5) * len(group)
        assert sum(saved[i.name] for i in group) == budget

    assert (tmp_path / "ranks_grid.json").exists()
    model = model_mod.cast_for_inference(model, cfg)
    results = evaluate_mod.run_eval(model, tokenizer, cfg)
    assert math.isfinite(results["perplexity"]["wikitext2"])
