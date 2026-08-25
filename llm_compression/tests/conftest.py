"""Shared test fixtures and reference-path bootstrap."""

import pytest
import torch

from pipeline._paths import add_reference_paths

add_reference_paths()

# COALA's inject_coala calls torch.set_num_interop_threads(1). That can only be
# set once per process before inter-op work begins; in the real pipeline it runs
# exactly once, but across a multi-test session repeated calls raise RuntimeError.
# Make it tolerant for tests only (COALA itself is reference-only and untouched).
_orig_set_interop = torch.set_num_interop_threads


def _safe_set_interop(n):
    try:
        _orig_set_interop(n)
    except RuntimeError:
        pass


torch.set_num_interop_threads = _safe_set_interop

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _cpu_base_config(tmp_path, **overrides):
    """A minimal config dict usable on CPU."""
    cfg = {
        "model": {
            "id": TINY_MODEL,
            "load_dtype": "float32",
            "inference_dtype": "float32",
            "device": "cpu",
            "trust_remote_code": False,
            "tokenizer": {"use_fast": True},
        },
        "compress": {"enabled": False},
        "calibration": {
            "dataset": "wikitext2",
            "seq_len": 16,
            "num_samples": 2,
            "batch_size": 2,
            "dataset_cache_dir": str(tmp_path / "datasets"),
            "data_cache_dir": str(tmp_path / "data_cache"),
        },
        "eval": {
            "perplexity": {
                "enabled": True,
                "datasets": ["wikitext2"],
                "seq_len": 16,
                "num_samples": 2,
                "batch_size": 1,
            },
            "reasoning": {"enabled": False, "tasks": ["piqa"]},
        },
        "seed": 0,
        "results_path": str(tmp_path / "results.json"),
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def cpu_base_config():
    return _cpu_base_config


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
