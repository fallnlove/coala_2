"""CLI entry point for the unified compression pipeline.

Usage:
    python -m pipeline.run --config configs/coala.yaml
    python -m pipeline.run --config configs/coala.yaml model.id=meta-llama/Llama-3.2-1B

Any trailing ``key=value`` arguments are OmegaConf dotlist overrides applied on
top of the YAML config (e.g. ``compress.ratio=0.5``, ``model.device=cpu``).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

# Make the reference codebases importable before anything imports from them.
from pipeline._paths import add_reference_paths

add_reference_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pipeline import compress as compress_mod  # noqa: E402
from pipeline import evaluate as evaluate_mod  # noqa: E402
from pipeline import model as model_mod  # noqa: E402
from pipeline.config import load_config, validate_config  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified LLM compression pipeline")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. model.id=... compress.ratio=0.5",
    )
    return parser.parse_args(argv)


def run(config_path: str, overrides: List[str] | None = None) -> dict:
    cfg = load_config(config_path, overrides)
    validate_config(cfg, cuda_available=torch.cuda.is_available())
    set_seed(int(cfg.get("seed", 0)))

    print("=== Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    # 1-2. Load model + tokenizer (datasets are downloaded lazily on first use).
    model, tokenizer = model_mod.load_model_and_tokenizer(cfg)

    # 3. Optional compression (in place).
    model, ranks = compress_mod.maybe_compress(model, tokenizer, cfg)

    # Rank-selection-only run (compress.dobi.stop_after_ranks=true): ranks are
    # checkpointed; there is no compressed model to save or evaluate.
    if model is None:
        payload = {
            "model": str(cfg.model.id),
            "compress": OmegaConf.to_container(cfg.get("compress", {}), resolve=True),
            "ranks": ranks,
            "results": {},
        }
        results_path = Path(str(cfg.get("results_path", "./outputs/results.json")))
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Stopped after rank selection. Wrote {results_path}")
        return payload

    if ranks is not None and bool(cfg.compress.get("save", {}).get("enabled", False)):
        out_dir = model_mod.save_compressed(model, tokenizer, ranks, cfg)
        print(f"Saved compressed model to {out_dir}")

    # 4. Evaluate. run_eval casts the model per phase (fp32 perplexity,
    #    fp16 reasoning by default; override via eval.<phase>.dtype).
    results = evaluate_mod.run_eval(model, tokenizer, cfg)

    # 5. Persist results.
    payload = {
        "model": str(cfg.model.id),
        "compress": OmegaConf.to_container(cfg.get("compress", {}), resolve=True),
        "results": results,
    }
    results_path = Path(str(cfg.get("results_path", "./outputs/results.json")))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("=== Results ===")
    print(json.dumps(results, indent=2, default=str))
    print(f"Wrote {results_path}")
    return payload


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    run(args.config, args.overrides)


if __name__ == "__main__":
    main()
