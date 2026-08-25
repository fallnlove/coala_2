"""Config validation rules (pure, CPU-only)."""

import pytest
from omegaconf import OmegaConf

from pipeline.config import validate_config


def _base():
    return OmegaConf.create(
        {
            "model": {"id": "x"},
            "compress": {
                "enabled": True,
                "strategy": "coala",
                "ratio": 0.5,
                "target_modules": ["q_proj"],
                "params": {"mu": 1.0, "adaptive": True, "log_norms": False},
            },
            "calibration": {"dataset": "wikitext2", "num_samples": 8, "batch_size": 2},
            "eval": {
                "perplexity": {"enabled": True, "datasets": ["wikitext2"]},
                "reasoning": {"enabled": True, "tasks": ["piqa"]},
            },
        }
    )


def test_valid_coala_config_passes():
    validate_config(_base(), cuda_available=True)


def test_unknown_strategy_rejected():
    cfg = _base()
    cfg.compress.strategy = "nope"
    with pytest.raises(ValueError, match="compress.strategy"):
        validate_config(cfg, cuda_available=True)


def test_ratio_out_of_range_rejected():
    cfg = _base()
    cfg.compress.ratio = 1.5
    with pytest.raises(ValueError, match="ratio"):
        validate_config(cfg, cuda_available=True)


def test_asvd_requires_alpha():
    cfg = _base()
    cfg.compress.strategy = "asvd"
    cfg.compress.params = {"accumulate": 0}
    with pytest.raises(ValueError, match="missing keys"):
        validate_config(cfg, cuda_available=True)


def test_adaptive_rank_incompatible_with_svd():
    cfg = _base()
    cfg.compress.strategy = "svd"
    cfg.compress.params = {}
    cfg.compress.adaptive_rank = True
    with pytest.raises(ValueError, match="adaptive_rank"):
        validate_config(cfg, cuda_available=True)


def test_gpu_only_strategy_requires_cuda():
    cfg = _base()
    with pytest.raises(ValueError, match="requires a CUDA device"):
        validate_config(cfg, cuda_available=False)


def test_empty_strategy_allowed_on_cpu():
    cfg = _base()
    cfg.compress.strategy = "empty"
    cfg.compress.params = {}
    validate_config(cfg, cuda_available=False)


def test_unknown_ppl_dataset_rejected():
    cfg = _base()
    cfg.compress.enabled = False
    cfg.eval.perplexity.datasets = ["wikitext2", "bogus"]
    with pytest.raises(ValueError, match="perplexity.datasets"):
        validate_config(cfg, cuda_available=False)


def test_unknown_reasoning_task_rejected():
    cfg = _base()
    cfg.compress.enabled = False
    cfg.eval.reasoning.tasks = ["piqa", "not_a_task"]
    with pytest.raises(ValueError, match="reasoning.tasks"):
        validate_config(cfg, cuda_available=False)


def test_per_phase_eval_dtype_accepted():
    cfg = _base()
    cfg.compress.enabled = False
    cfg.eval.perplexity.dtype = "float32"
    cfg.eval.reasoning.dtype = "float16"
    validate_config(cfg, cuda_available=False)


def test_invalid_eval_dtype_rejected():
    cfg = _base()
    cfg.compress.enabled = False
    cfg.eval.perplexity.dtype = "float8"
    with pytest.raises(ValueError, match="perplexity.dtype"):
        validate_config(cfg, cuda_available=False)


def _dobi_base():
    return OmegaConf.create(
        {
            "model": {"id": "meta-llama/Llama-2-7b-hf", "load_dtype": "float16"},
            "compress": {
                "enabled": True,
                "strategy": "dobi",
                "ratio": 0.8,
                "dobi": {
                    "rank_selection": {
                        "dataset": "wikitext2",
                        "num_samples": 8,
                        "n_epochs": 2,
                        "seq_len": 32,
                        "lr": 5.0,
                        "beta": 10,
                        "lambda_reg": 200,
                    }
                },
            },
            "calibration": {"dataset": "c4", "num_samples": 8},
            "eval": {
                "perplexity": {"enabled": False},
                "reasoning": {"enabled": False},
            },
        }
    )


def test_dobi_config_valid():
    validate_config(_dobi_base(), cuda_available=True)


def test_dobi_allows_any_load_dtype():
    # Rank training forces fp32 internally and IPCA reloads fp16, so load_dtype
    # is unconstrained for dobi (used to require float16).
    cfg = _dobi_base()
    cfg.model.load_dtype = "float32"
    validate_config(cfg, cuda_available=True)


def test_dobi_requires_rank_selection_block():
    cfg = _dobi_base()
    del cfg.compress.dobi
    with pytest.raises(ValueError, match="rank_selection"):
        validate_config(cfg, cuda_available=True)


def test_dobi_requires_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        validate_config(_dobi_base(), cuda_available=False)
