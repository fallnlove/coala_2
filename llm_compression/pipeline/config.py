"""Configuration loading, validation, and translation to a ``COALA_Config``.

The pipeline is driven by a single YAML file (see ``configs/``). This module:

* loads it with OmegaConf and applies CLI dotlist overrides (e.g. ``model.id=...``),
* validates the combination of options against the constraints baked into the
  COALA machinery (see the plan / code comments below),
* builds the ``COALA_Config`` dataclass the compression step consumes.
"""

from __future__ import annotations

from typing import List

from omegaconf import DictConfig, OmegaConf

# COALA strategies grouped by whether they need calibration activations. Mirrors
# COALA/coala/Config.py:{METHODS_WITHOUT_SAMPLES, METHODS_WITH_SAMPLES}.
SAMPLE_FREE_STRATEGIES = {"empty", "svd"}
SAMPLE_BASED_STRATEGIES = {"svd_llm", "svd_llm_2", "coala", "asvd", "swift_svd"}
COALA_STRATEGIES = SAMPLE_FREE_STRATEGIES | SAMPLE_BASED_STRATEGIES

# Dobi-SVD is its own two-stage strategy (rank selection + IPCA), not a COALA solver.
DOBI_STRATEGIES = {"dobi"}

# COALA per-layer rank source: uniform (default), learned (train), loaded, or
# Swift-SVD's training-free dynamic allocation (``swift``, strategy=swift_svd only).
COALA_RANK_SOURCES = {"uniform", "train", "checkpoint", "swift"}

# Swift-SVD dynamic-allocation knobs where the paper and the released code differ
# (see pipeline/swift_rank.py). Defaults follow the paper (Algorithm 2).
SWIFT_SCORE_MODES = {"tail", "ynorm"}
SWIFT_BASE_RANK_MODES = {"delta_kbar", "released"}

ALL_STRATEGIES = COALA_STRATEGIES | DOBI_STRATEGIES

# Everything except ``empty`` (zero-init adapters) needs CUDA: COALA solvers use
# the gesvd cuSOLVER driver / .cuda(); Dobi runs SVD + IPCA on GPU.
GPU_ONLY_STRATEGIES = ALL_STRATEGIES - {"empty"}

VALID_DTYPES = {"float32", "float16", "bfloat16", "fp32", "fp16", "bf16"}

PPL_DATASETS = {"wikitext2", "c4", "ptb"}
REASONING_TASKS = {
    "piqa",
    "hellaswag",
    "boolq",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
}

# Required ``params`` keys per strategy (read by COALA/coala/Linear.py).
REQUIRED_PARAMS = {
    "coala": {"mu", "adaptive", "log_norms"},
    "asvd": {"alpha"},
}


def load_config(path: str, overrides: List[str] | None = None) -> DictConfig:
    """Load a YAML config and merge CLI dotlist overrides on top of it."""
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_dtype(value, key: str) -> None:
    _require(
        value is None or str(value).lower() in VALID_DTYPES,
        f"{key} must be one of {sorted(VALID_DTYPES)}, got {value!r}",
    )


def _validate_dobi(cfg: DictConfig) -> None:
    """Validate the Dobi-SVD rank-selection config block."""
    dobi = cfg.compress.get("dobi", {})
    rs = dobi.get("rank_selection", {})
    _require(
        bool(rs), "compress.dobi.rank_selection block is required for strategy=dobi"
    )
    _require(
        rs.get("dataset") in PPL_DATASETS,
        f"compress.dobi.rank_selection.dataset must be one of {sorted(PPL_DATASETS)}",
    )
    for key in ("num_samples", "n_epochs", "seq_len"):
        _require(
            int(rs.get(key, 0)) > 0,
            f"compress.dobi.rank_selection.{key} must be > 0",
        )
    _require(float(rs.get("lr", 0)) > 0, "compress.dobi.rank_selection.lr must be > 0")
    _require(
        str(rs.get("svd", "stable")) in {"stable", "cans"},
        "compress.dobi.rank_selection.svd must be 'stable' or 'cans'",
    )
    _require(
        float(rs.get("cans_eps", 1e-8)) > 0,
        "compress.dobi.rank_selection.cans_eps must be > 0",
    )
    _require(
        str(rs.get("cans_differentiation", "tikhonov")) in {"tikhonov", "pseudo"},
        "compress.dobi.rank_selection.cans_differentiation must be 'tikhonov' or 'pseudo'",
    )
    # Optional validation / early-stopping (off when val_num_samples/patience are 0).
    _require(
        int(rs.get("val_num_samples", 0)) >= 0,
        "compress.dobi.rank_selection.val_num_samples must be >= 0",
    )
    _require(
        int(rs.get("early_stopping_patience", 0)) >= 0,
        "compress.dobi.rank_selection.early_stopping_patience must be >= 0",
    )
    _require(
        float(rs.get("early_stopping_min_delta", 0.0)) >= 0.0,
        "compress.dobi.rank_selection.early_stopping_min_delta must be >= 0",
    )
    if int(rs.get("early_stopping_patience", 0)) > 0:
        _require(
            int(rs.get("val_num_samples", 0)) > 0,
            "compress.dobi.rank_selection.early_stopping_patience>0 requires "
            "val_num_samples>0 (early stopping needs a validation set).",
        )
    # Low-memory rank training: keep the base model in its loaded dtype (never
    # the fp32 upcast) + fp32 SVD islands + per-layer checkpointing. Only sound
    # with a half-precision base; bfloat16 recommended (fp16 risks the SVD-NaN
    # path that motivated fp32 training in the first place).
    if bool(rs.get("low_mem", False)):
        _require(
            str(cfg.model.get("load_dtype", "float32")).lower()
            in {"bfloat16", "bf16", "float16", "fp16"},
            "compress.dobi.rank_selection.low_mem=true requires "
            "model.load_dtype in {bfloat16, float16} (bfloat16 recommended)",
        )
    # Note: outside low_mem, rank training runs in fp32 internally and IPCA
    # reloads at model.load_dtype, so load_dtype is otherwise unconstrained.


def _validate_swift_rank_selection(cfg: DictConfig, cuda_available: bool) -> None:
    """Validate the Swift-SVD dynamic rank-allocation block (Algorithm 2)."""
    rs = cfg.compress.get("swift_rank_selection", {})
    _require(
        cuda_available,
        "compress.rank_source=swift requires a CUDA device (GPU SVD per layer + "
        "candidate validation forwards).",
    )
    delta = float(rs.get("delta", 0.5))
    _require(
        0.0 < delta <= 1.0,
        "compress.swift_rank_selection.delta (retained ratio) must be in (0, 1]",
    )
    alphas = list(to_plain(rs.get("alphas", None), default=[]) or [])
    _require(
        bool(alphas),
        "compress.swift_rank_selection.alphas must be a non-empty list of "
        "scaling factors (paper grid: 0.0..1.0 step 0.1)",
    )
    for a in alphas:
        _require(
            0.0 <= float(a) <= 1.0,
            f"compress.swift_rank_selection.alphas entries must be in [0, 1], got {a!r}",
        )
    score = str(rs.get("score", "tail"))
    _require(
        score in SWIFT_SCORE_MODES,
        f"compress.swift_rank_selection.score must be one of "
        f"{sorted(SWIFT_SCORE_MODES)}, got {score!r}",
    )
    base_rank = str(rs.get("base_rank", "delta_kbar"))
    _require(
        base_rank in SWIFT_BASE_RANK_MODES,
        f"compress.swift_rank_selection.base_rank must be one of "
        f"{sorted(SWIFT_BASE_RANK_MODES)}, got {base_rank!r}",
    )
    _validate_dtype(rs.get("cache_dtype"), "compress.swift_rank_selection.cache_dtype")

    for block in ("val", "importance"):
        sub = rs.get(block, {})
        _require(
            bool(sub),
            f"compress.swift_rank_selection.{block} block is required for rank_source=swift",
        )
        _require(
            sub.get("dataset") in PPL_DATASETS,
            f"compress.swift_rank_selection.{block}.dataset must be one of "
            f"{sorted(PPL_DATASETS)}",
        )
        for key in ("num_samples", "seq_len"):
            _require(
                int(sub.get(key, 0)) > 0,
                f"compress.swift_rank_selection.{block}.{key} must be > 0",
            )


def _validate_coala_rank_source(
    cfg: DictConfig, strategy, cuda_available: bool
) -> None:
    """Validate the COALA learned/loaded per-layer rank config."""
    compress = cfg.compress
    rank_source = str(compress.get("rank_source", "uniform"))
    _require(
        rank_source in COALA_RANK_SOURCES,
        f"compress.rank_source must be one of {sorted(COALA_RANK_SOURCES)}, got {rank_source!r}",
    )
    if rank_source == "uniform":
        return

    # Swift-SVD owns its own rank pipeline (training-free allocation + val grid
    # search) and its own config block; it never touches coala_rank_selection.
    if strategy == "swift_svd":
        _require(
            rank_source in {"swift", "checkpoint"},
            "strategy=swift_svd supports compress.rank_source in "
            f"{{uniform, swift, checkpoint}}, got {rank_source!r}",
        )
        rs = compress.get("swift_rank_selection", {})
        _require(
            bool(rs),
            "compress.swift_rank_selection block is required for rank_source!=uniform "
            "with strategy=swift_svd",
        )
        if rank_source == "checkpoint":
            _require(
                bool(rs.get("checkpoint")),
                "compress.swift_rank_selection.checkpoint path is required for "
                "rank_source=checkpoint",
            )
            return
        _validate_swift_rank_selection(cfg, cuda_available)
        return

    _require(
        rank_source != "swift",
        "compress.rank_source=swift requires compress.strategy=swift_svd "
        "(it is Swift-SVD's own dynamic allocation, Algorithm 2).",
    )
    _require(
        strategy in SAMPLE_BASED_STRATEGIES,
        f"compress.rank_source={rank_source!r} needs calibration activations; "
        f"use a sample-based strategy ({sorted(SAMPLE_BASED_STRATEGIES)}).",
    )
    rs = compress.get("coala_rank_selection", {})
    _require(
        bool(rs),
        "compress.coala_rank_selection block is required for rank_source!=uniform",
    )

    if rank_source == "checkpoint":
        _require(
            bool(rs.get("checkpoint")),
            "compress.coala_rank_selection.checkpoint path is required for rank_source=checkpoint",
        )
        return

    # rank_source == "train"
    _require(
        cuda_available,
        "COALA rank_source=train requires a CUDA device (cans_svd + GPU SVD).",
    )
    _require(
        rs.get("dataset") in PPL_DATASETS,
        f"compress.coala_rank_selection.dataset must be one of {sorted(PPL_DATASETS)}",
    )
    for key in ("num_samples", "n_epochs", "seq_len"):
        _require(
            int(rs.get(key, 0)) > 0,
            f"compress.coala_rank_selection.{key} must be > 0",
        )
    _require(float(rs.get("lr", 0)) > 0, "compress.coala_rank_selection.lr must be > 0")
    _require(
        float(rs.get("beta", 0)) > 0, "compress.coala_rank_selection.beta must be > 0"
    )
    _require(
        float(rs.get("cans_eps", 1e-8)) > 0,
        "compress.coala_rank_selection.cans_eps must be > 0",
    )
    _require(
        str(rs.get("cans_differentiation", "tikhonov")) in {"tikhonov", "pseudo"},
        "compress.coala_rank_selection.cans_differentiation must be 'tikhonov' or 'pseudo'",
    )
    _require(
        float(rs.get("max_rank_ratio", 2.0)) >= 1.0,
        "compress.coala_rank_selection.max_rank_ratio must be >= 1",
    )
    # Low-memory rank training (opt-in): keep the base model in its loaded dtype
    # (never fp32) + store fp16 factors + cap the factor rank at break-even. Only
    # sound when the model is loaded in a half precision — fp32 base defeats the
    # purpose, and the fp16-factor forward needs a half-precision residual stream.
    if bool(rs.get("low_mem", False)):
        _require(
            str(cfg.model.get("load_dtype", "float32")).lower()
            in {"bfloat16", "bf16", "float16", "fp16"},
            "compress.coala_rank_selection.low_mem=true requires "
            "model.load_dtype in {bfloat16, float16} (bfloat16 recommended)",
        )
    # Compression objective for the rank-training regularizer (and the uniform
    # init/reference derived from it): params = sum (m+n)r / sum mn (default);
    # ranks = sum r / sum out_features (aggregate rank fraction).
    ratio_objective = str(rs.get("ratio_objective", "params"))
    _require(
        ratio_objective in {"params", "ranks"},
        f"compress.coala_rank_selection.ratio_objective must be 'params' or "
        f"'ranks', got {ratio_objective!r}",
    )
    # Rank trainer: continuous gate (default) or discrete REINFORCE ablation.
    trainer = str(rs.get("trainer", "gate"))
    _require(
        trainer in {"gate", "reinforce"},
        f"compress.coala_rank_selection.trainer must be 'gate' or 'reinforce', "
        f"got {trainer!r}",
    )
    _require(
        not (
            ratio_objective == "ranks"
            and trainer == "reinforce"
            and str(rs.get("rf_budget", "penalty")) == "project"
        ),
        "compress.coala_rank_selection.ratio_objective=ranks is not supported "
        "with trainer=reinforce + rf_budget=project (the hard budget projection "
        "is denominated in params); use rf_budget=penalty",
    )
    if trainer == "reinforce":
        _require(
            int(rs.get("rf_samples", 4)) >= 2,
            "compress.coala_rank_selection.rf_samples must be >= 2 "
            "(the leave-one-out baseline needs at least two samples)",
        )
        _require(
            str(rs.get("rf_budget", "penalty")) in {"penalty", "project"},
            "compress.coala_rank_selection.rf_budget must be 'penalty' or 'project'",
        )
        w_ce = float(rs.get("rf_w_ce", 1.0))
        w_kl = float(rs.get("rf_w_kl", 0.0))
        w_hid = float(rs.get("rf_w_hidden", 0.0))
        w_ent = float(rs.get("rf_w_entropy", 0.0))
        for name, w in (
            ("rf_w_ce", w_ce),
            ("rf_w_kl", w_kl),
            ("rf_w_hidden", w_hid),
            ("rf_w_entropy", w_ent),
        ):
            _require(w >= 0.0, f"compress.coala_rank_selection.{name} must be >= 0")
        _require(
            w_ce + w_kl + w_hid > 0.0,
            "compress.coala_rank_selection: at least one of rf_w_ce/rf_w_kl/"
            "rf_w_hidden must be > 0 (an all-zero reward is a pure noise walk)",
        )
        learn_sigma = bool(rs.get("rf_learn_sigma", True))
        _require(
            not (w_ent > 0.0 and not learn_sigma),
            "compress.coala_rank_selection.rf_w_entropy > 0 requires "
            "rf_learn_sigma=true (with a fixed sigma schedule the entropy term "
            "is a silent no-op)",
        )
        _require(
            float(rs.get("rf_sigma_lr_scale", 0.1)) > 0.0,
            "compress.coala_rank_selection.rf_sigma_lr_scale must be > 0",
        )
        sigma_init = float(rs.get("rf_sigma_init", 0.05))
        _require(
            sigma_init > 0.0,
            "compress.coala_rank_selection.rf_sigma_init must be > 0",
        )
        if not learn_sigma:
            _require(
                sigma_init > float(rs.get("rf_sigma_final", 0.005)) > 0.0,
                "compress.coala_rank_selection.rf_sigma_init must be > "
                "rf_sigma_final > 0 (rf_sigma_final is inert when "
                "rf_learn_sigma=true)",
            )
        _require(
            float(rs.get("rf_sigma_min", 1.0)) > 0.0,
            "compress.coala_rank_selection.rf_sigma_min must be > 0",
        )
        beta_ema = float(rs.get("rf_adv_ema_beta", 0.99))
        _require(
            0.0 < beta_ema < 1.0,
            "compress.coala_rank_selection.rf_adv_ema_beta must be in (0, 1)",
        )
        _require(
            float(rs.get("rf_adv_std_floor", 1e-3)) > 0.0,
            "compress.coala_rank_selection.rf_adv_std_floor must be > 0",
        )
    # Optional validation / early-stopping (off when val_num_samples/patience are 0).
    _require(
        int(rs.get("val_num_samples", 0)) >= 0,
        "compress.coala_rank_selection.val_num_samples must be >= 0",
    )
    _require(
        int(rs.get("early_stopping_patience", 0)) >= 0,
        "compress.coala_rank_selection.early_stopping_patience must be >= 0",
    )
    _require(
        float(rs.get("early_stopping_min_delta", 0.0)) >= 0.0,
        "compress.coala_rank_selection.early_stopping_min_delta must be >= 0",
    )
    if int(rs.get("early_stopping_patience", 0)) > 0:
        _require(
            int(rs.get("val_num_samples", 0)) > 0,
            "compress.coala_rank_selection.early_stopping_patience>0 requires "
            "val_num_samples>0 (early stopping needs a validation set).",
        )


def validate_config(cfg: DictConfig, *, cuda_available: bool) -> None:
    """Validate a loaded config. Raises ``ValueError`` on any inconsistency.

    ``cuda_available`` is injected so the rules are testable on a CPU-only host.
    """
    compress = cfg.get("compress", {})
    enabled = bool(compress.get("enabled", False))

    if enabled:
        strategy = compress.get("strategy")
        _require(
            strategy in ALL_STRATEGIES,
            f"compress.strategy must be one of {sorted(ALL_STRATEGIES)}, got {strategy!r}",
        )

        ratio = compress.get("ratio")
        _require(
            ratio is not None and 0.0 < float(ratio) <= 1.0,
            f"compress.ratio must be in (0, 1], got {ratio!r}",
        )

        if strategy in GPU_ONLY_STRATEGIES:
            _require(
                cuda_available,
                f"compress.strategy={strategy!r} requires a CUDA device "
                "(SVD uses the gesvd cuSOLVER driver / GPU SVD+IPCA). "
                "Use strategy=empty for a CPU dry-run, or run on the GPU.",
            )

        # Strategies that consume the calibration set (COALA data-aware + Dobi IPCA).
        if strategy in SAMPLE_BASED_STRATEGIES or strategy in DOBI_STRATEGIES:
            calib = cfg.get("calibration", {})
            _require(
                calib.get("dataset") in PPL_DATASETS,
                f"calibration.dataset must be one of {sorted(PPL_DATASETS)} "
                f"for strategy {strategy!r}",
            )
            _require(
                int(calib.get("num_samples", 0)) > 0,
                "calibration.num_samples must be > 0 for this strategy",
            )

        if strategy in COALA_STRATEGIES:
            params = dict(compress.get("params", {}) or {})
            missing = REQUIRED_PARAMS.get(strategy, set()) - params.keys()
            _require(
                not missing,
                f"compress.params for strategy {strategy!r} is missing keys: {sorted(missing)}",
            )
            _require(
                bool(compress.get("target_modules")),
                "compress.target_modules must be a non-empty list",
            )
            if bool(compress.get("adaptive_rank", False)):
                _require(
                    strategy in SAMPLE_BASED_STRATEGIES,
                    "compress.adaptive_rank=true requires a sample-based strategy "
                    f"({sorted(SAMPLE_BASED_STRATEGIES)}); it needs calibration activations.",
                )
                # COALA's heterogeneous_compression would silently override /
                # compete with Swift-SVD's own allocation (uniform or Algorithm 2).
                _require(
                    strategy != "swift_svd",
                    "compress.adaptive_rank=true is incompatible with "
                    "strategy=swift_svd: Swift-SVD allocates ranks itself "
                    "(uniform, or rank_source=swift for Algorithm 2).",
                )
            _require(
                int(cfg.get("calibration", {}).get("batch_size", 1)) > 0,
                "calibration.batch_size must be > 0",
            )
            _validate_coala_rank_source(cfg, strategy, cuda_available)

        if strategy in DOBI_STRATEGIES:
            _validate_dobi(cfg)

    # Evaluation sanity checks (independent of compression).
    eval_cfg = cfg.get("eval", {})
    ppl_cfg = eval_cfg.get("perplexity", {})
    if ppl_cfg.get("enabled", False):
        datasets = list(ppl_cfg.get("datasets", []) or [])
        unknown = set(datasets) - PPL_DATASETS
        _require(
            datasets and not unknown,
            f"eval.perplexity.datasets must be a non-empty subset of "
            f"{sorted(PPL_DATASETS)}; unknown: {sorted(unknown)}",
        )
        _validate_dtype(ppl_cfg.get("dtype"), "eval.perplexity.dtype")

        contiguous = set(ppl_cfg.get("contiguous", []) or [])
        unknown_c = contiguous - set(datasets)
        _require(
            not unknown_c,
            f"eval.perplexity.contiguous must be a subset of datasets "
            f"{sorted(datasets)}; unknown: {sorted(unknown_c)}",
        )

    reasoning_cfg = eval_cfg.get("reasoning", {})
    if reasoning_cfg.get("enabled", False):
        tasks = list(reasoning_cfg.get("tasks", []) or [])
        unknown = set(tasks) - REASONING_TASKS
        _require(
            tasks and not unknown,
            f"eval.reasoning.tasks must be a non-empty subset of "
            f"{sorted(REASONING_TASKS)}; unknown: {sorted(unknown)}",
        )
        _validate_dtype(reasoning_cfg.get("dtype"), "eval.reasoning.dtype")


def to_plain(node, default=None):
    """Convert an OmegaConf node to a plain Python container, robust to empties.

    An empty ``DictConfig``/``ListConfig`` is falsy, so ``node or {}`` would
    wrongly collapse it to a bare ``dict`` that ``OmegaConf.to_container`` rejects.
    """
    if node is None:
        return default
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


def build_coala_config(cfg: DictConfig):
    """Translate ``cfg.compress`` into a COALA ``COALA_Config`` dataclass.

    ``params`` and ``target_modules`` are converted to plain Python containers
    (COALA indexes ``params['mu']`` and ThreadPool-pickles the config).
    """
    from coala.Config import COALA_Config  # imported lazily after sys.path setup

    compress = cfg.compress
    params = dict(to_plain(compress.get("params", None), default={}) or {})
    target_modules = list(to_plain(compress.target_modules, default=[]))

    return COALA_Config(
        ratio=float(compress.ratio),
        params=dict(params),
        target_modules=target_modules,
        compress_strategy=str(compress.strategy),
        samples=None,  # unused: we feed calibration via a manual forward loop
        fp16=bool(compress.get("fp16", False)),
        adaptive_rank=bool(compress.get("adaptive_rank", False)),
    )
