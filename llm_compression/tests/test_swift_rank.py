"""Swift-SVD: solver optimality, Algorithm 2 allocation, plumbing — pure CPU."""

import json
import math

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pipeline._paths import add_reference_paths

add_reference_paths()

import pipeline.swift_rank as sr  # noqa: E402
from coala.Linear import COALA_Linear  # noqa: E402
from lib import Solvers  # noqa: E402
from pipeline.config import validate_config  # noqa: E402


# --------------------------------------------------------------------------- #
# Stage A: closed-form operator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m,n,s", [(8, 6, 40), (6, 8, 40), (8, 6, 3), (10, 10, 25)])
@pytest.mark.parametrize("r", [1, 2, 4])
def test_swift_solver_hits_theoretical_optimum(m, n, s, r):
    """||(W - W')X||_F must equal the singular tail sqrt(sum_{j>r} sigma_j^2),
    i.e. the paper's eps*_r (Theorem 3.1 / eq. 4)."""
    if r > min(m, n, s):
        pytest.skip("rank exceeds the achievable rank of W @ X")
    torch.manual_seed(0)
    W = torch.randn(m, n, dtype=torch.float64)
    X = torch.randn(n, s, dtype=torch.float64)
    B, A = Solvers.swift_svd_method(W, X, r)
    assert B.shape == (m, r) and A.shape == (r, n)

    achieved = torch.linalg.norm((W - B @ A) @ X)
    sv = torch.linalg.svdvals(W @ X)
    expected = (
        torch.sqrt(torch.sum(sv[r:] ** 2)) if r < sv.numel() else torch.tensor(0.0)
    )
    assert achieved.item() == pytest.approx(expected.item(), rel=1e-8, abs=1e-8)


def test_swift_solver_is_output_projection():
    """W' = V_r V_rᵀ W: the operator is a projection applied on the OUTPUT side,
    so B's columns are orthonormal and B @ Bᵀ W reproduces W'."""
    torch.manual_seed(1)
    W = torch.randn(9, 7, dtype=torch.float64)
    X = torch.randn(7, 30, dtype=torch.float64)
    B, A = Solvers.swift_svd_method(W, X, 3)
    torch.testing.assert_close(B.T @ B, torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(B @ A, B @ (B.T @ W))


def test_swift_solver_no_worse_than_plain_svd():
    """Activation awareness must pay off: the Swift operator's data-weighted error
    cannot exceed that of a plain (activation-blind) truncated SVD of W."""
    torch.manual_seed(2)
    W = torch.randn(12, 10, dtype=torch.float64)
    X = torch.randn(10, 50, dtype=torch.float64) @ torch.diag(
        torch.linspace(1.0, 40.0, 50, dtype=torch.float64)
    )
    r = 4
    B, A = Solvers.swift_svd_method(W, X, r)
    # Plain truncated SVD of W, inline: Solvers.svd_low_rank_approx hardcodes the
    # CUDA-only gesvd driver.
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W_svd = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
    swift_err = torch.linalg.norm((W - B @ A) @ X).item()
    svd_err = torch.linalg.norm((W - W_svd) @ X).item()
    assert swift_err <= svd_err + 1e-9


def test_swift_solver_finite_guard():
    torch.manual_seed(3)
    W = torch.randn(6, 5)
    X = torch.randn(5, 20)
    X[0, 0] = float("inf")
    X[1, 1] = float("nan")
    B, A = Solvers.swift_svd_method(W, X, 2)
    assert torch.isfinite(B).all() and torch.isfinite(A).all()


def test_swift_factored_forward_equals_dense():
    """COALA_Linear's (X @ adapter_A) @ adapter_B must reproduce x @ W'ᵀ, i.e. the
    (B, A) convention of the solver matches the injector."""
    torch.manual_seed(4)
    m, n = 8, 6
    W = torch.randn(m, n, dtype=torch.float64)
    X = torch.randn(n, 40, dtype=torch.float64)
    B, A = Solvers.swift_svd_method(W, X, 3)
    W_prime = B @ A
    x = torch.randn(5, n, dtype=torch.float64)
    adapter_A, adapter_B = A.T, B.T  # what set_parameters stores
    torch.testing.assert_close((x @ adapter_A) @ adapter_B, x @ W_prime.T)


def test_project_shares_gram_with_raw_product():
    """The QR pre-step must not change the spectrum (same Gram matrix)."""
    torch.manual_seed(5)
    W = torch.randn(8, 6, dtype=torch.float64)
    X = torch.randn(6, 40, dtype=torch.float64)
    sv_qr = torch.linalg.svdvals(Solvers.swift_svd_project(W, X))
    sv_raw = torch.linalg.svdvals(W @ X)
    k = min(sv_qr.numel(), sv_raw.numel())
    torch.testing.assert_close(sv_qr[:k], sv_raw[:k], rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# Spectrum helpers
# --------------------------------------------------------------------------- #
def test_tail_loss_matches_measured_residual_for_every_rank():
    torch.manual_seed(6)
    W = torch.randn(7, 5, dtype=torch.float64)
    X = torch.randn(5, 30, dtype=torch.float64)
    sv = torch.linalg.svdvals(Solvers.swift_svd_project(W, X))
    for k in range(1, min(W.shape) + 1):
        B, A = Solvers.swift_svd_method(W, X, k)
        measured = torch.linalg.norm((W - B @ A) @ X).item()
        assert sr._tail_loss(sv, k) == pytest.approx(measured, rel=1e-7, abs=1e-9)


def test_tail_loss_zero_past_spectrum():
    sv = torch.tensor([3.0, 2.0, 1.0])
    assert sr._tail_loss(sv, 3) == 0.0
    assert sr._tail_loss(sv, 99) == 0.0
    assert sr._tail_loss(sv, 0) == pytest.approx(math.sqrt(9 + 4 + 1))


def test_fro_norm_and_erank():
    sv = torch.tensor([2.0, 0.0, 0.0, 0.0])
    assert sr._fro_norm(sv) == pytest.approx(2.0)
    # One direction carries everything -> effective rank 1.
    assert sr._erank(sv) == pytest.approx(1.0, abs=1e-6)
    flat = torch.ones(8)
    assert sr._erank(flat) == pytest.approx(8.0, rel=1e-5)
    assert sr._erank(torch.zeros(4)) == 0.0


@pytest.mark.parametrize(
    "m,n,ratio",
    [(4096, 4096, 0.6), (14336, 4096, 0.6), (1024, 4096, 0.8), (10, 10, 1.0)],
)
def test_kbar_matches_coala_linear(m, n, ratio):
    """The budget must be COALA's own per-layer rank, or Swift-SVD* would not be
    budget-matched to the uniform baselines."""
    layer = COALA_Linear(
        ratio=ratio, compress_strategy="empty", base_tensor=torch.zeros(m, n)
    )
    assert sr._kbar(m, n, ratio) == layer.r


# --------------------------------------------------------------------------- #
# Importance -> beta
# --------------------------------------------------------------------------- #
def test_beta_from_bi_range_and_order():
    beta = sr._beta_from_bi([0.1, 0.5, 0.3])
    assert beta[0] == pytest.approx(1.0, abs=1e-4)
    assert beta[1] == pytest.approx(2.0, abs=1e-4)
    assert 1.0 < beta[2] < 2.0
    assert all(1.0 <= b <= 2.0 for b in beta)


def test_beta_from_bi_constant_input_is_inert():
    beta = sr._beta_from_bi([0.4, 0.4, 0.4])
    assert beta == pytest.approx([1.0, 1.0, 1.0])
    assert sr._beta_from_bi([]) == []


class _FakeOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeModel:
    """Minimal stand-in exposing HF's ``output_hidden_states`` contract."""

    def __init__(self, hidden_states):
        self._hs = hidden_states

    def eval(self):
        return self

    def __call__(self, **_kwargs):
        return _FakeOutput(self._hs)


def _hs_stack(vectors, tokens=3):
    return [v.view(1, 1, -1).expand(1, tokens, v.numel()).clone() for v in vectors]


def test_block_importance_identical_and_orthogonal_states():
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    model = _FakeModel(_hs_stack([e1, e1, e2]))
    bi = sr._block_importance(model, [torch.zeros(3, dtype=torch.long)], "cpu")
    assert len(bi) == 2
    assert bi[0] == pytest.approx(0.0, abs=1e-6)  # unchanged hidden state
    assert bi[1] == pytest.approx(1.0, abs=1e-6)  # orthogonal rotation


def test_block_importance_angular_variant():
    e1 = torch.tensor([1.0, 0.0])
    e2 = torch.tensor([0.0, 1.0])
    model = _FakeModel(_hs_stack([e1, e2]))
    bi = sr._block_importance(
        model, [torch.zeros(2, dtype=torch.long)], "cpu", angular=True
    )
    assert bi[0] == pytest.approx(0.5, abs=1e-6)  # arccos(0)/pi


def test_block_importance_no_windows():
    assert sr._block_importance(_FakeModel([]), [], "cpu") == []


# --------------------------------------------------------------------------- #
# Algorithm 2: scores + allocation
# --------------------------------------------------------------------------- #
def test_score_modes_and_alpha_extremes():
    # alpha=1 -> importance only; alpha=0 -> compressibility only.
    assert sr._score(1.7, 5.0, 900.0, 1.0, "tail") == pytest.approx(1.7)
    # alpha=0 leaves log(e + eps) (paper) / log(1 + ||Y||_F) (released code).
    assert sr._score(1.7, 5.0, 900.0, 0.0, "tail") == pytest.approx(
        math.log(math.e + 5.0)
    )
    assert sr._score(1.7, 5.0, 900.0, 0.0, "ynorm") == pytest.approx(math.log(901.0))
    # Mid-grid: the full beta^alpha * log(...)^(1-alpha) product.
    assert sr._score(1.6, 4.0, 900.0, 0.5, "tail") == pytest.approx(
        (1.6**0.5) * math.log(math.e + 4.0) ** 0.5
    )
    # tail mode is rank-aware, ynorm mode is not
    assert sr._score(1.0, 1.0, 10.0, 0.5, "tail") != sr._score(
        1.0, 2.0, 10.0, 0.5, "tail"
    )
    assert sr._score(1.0, 1.0, 10.0, 0.5, "ynorm") == sr._score(
        1.0, 2.0, 10.0, 0.5, "ynorm"
    )


def test_score_monotone_in_both_signals():
    assert sr._score(1.9, 3.0, 1.0, 0.5) > sr._score(1.1, 3.0, 1.0, 0.5)
    assert sr._score(1.5, 9.0, 1.0, 0.5) > sr._score(1.5, 3.0, 1.0, 0.5)


@pytest.mark.parametrize("delta", [0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_allocate_type_conserves_budget(delta, seed):
    torch.manual_seed(seed)
    L, kbar, rmax = 12, 700, [4096] * 12
    scores = [float(v) for v in torch.rand(L) * 5 + 0.1]
    ranks = sr._allocate_type(kbar, rmax, scores, delta)
    assert sum(ranks) == kbar * L
    assert all(1 <= r <= 4096 for r in ranks)


def test_allocate_type_delta_one_is_uniform():
    ranks = sr._allocate_type(
        500, [4096] * 8, [1.0, 9.0, 3.0, 2.0, 7.0, 4.0, 5.0, 6.0], 1.0
    )
    assert ranks == [500] * 8


def test_allocate_type_follows_scores():
    """Higher score (more important / less compressible) must get more rank."""
    ranks = sr._allocate_type(500, [4096] * 4, [1.0, 2.0, 3.0, 10.0], 0.5)
    assert ranks == sorted(ranks)
    assert ranks[-1] > ranks[0]
    assert sum(ranks) == 2000


def test_allocate_type_clamps_and_redistributes():
    """A layer whose min(m,n) is small saturates; its share must go to the others
    and the pool budget must still be met exactly."""
    rmax = [64, 4096, 4096]
    ranks = sr._allocate_type(500, rmax, [100.0, 1.0, 1.0], 0.5)
    assert ranks[0] == 64
    assert sum(ranks) == 1500
    assert all(r <= rm for r, rm in zip(ranks, rmax))


def test_allocate_type_infeasible_budget_warns_and_caps():
    ranks = sr._allocate_type(100, [10, 10], [1.0, 1.0], 0.5)
    assert ranks == [10, 10]  # capped, cannot reach 200


def test_allocate_type_zero_scores_still_conserves():
    ranks = sr._allocate_type(300, [4096] * 5, [0.0] * 5, 0.5)
    assert sum(ranks) == 1500


def test_allocate_type_released_base_mode():
    """base_rank=released uses 0.25*ratio*out_features instead of delta*kbar."""
    ranks = sr._allocate_type(
        700,
        [4096] * 4,
        [1.0, 1.0, 1.0, 1.0],
        0.5,
        out_features=[4096] * 4,
        base_mode="released",
        ratio=0.6,
    )
    assert sum(ranks) == 2800
    # equal scores -> equal ranks (base 614 each, pool split evenly)
    assert len(set(ranks)) == 1


# --------------------------------------------------------------------------- #
# Candidate grid
# --------------------------------------------------------------------------- #
def _toy_infos():
    """Two module types x 3 blocks, Mistral-like shapes."""
    infos = []
    for block in range(3):
        for mtype, m, n in (("q_proj", 64, 64), ("gate_proj", 224, 64)):
            mod = nn.Linear(n, m, bias=False)
            infos.append(
                sr._LayerInfo(
                    name=f"model.layers.{block}.{mtype}",
                    module=mod,
                    block=block,
                    mtype=mtype,
                    m=m,
                    n=n,
                )
            )
    return infos


def test_candidate_allocations_budget_exact_per_type():
    infos = _toy_infos()
    torch.manual_seed(7)
    spectra = {i.name: torch.linspace(10.0, 0.1, min(i.m, i.n)) for i in infos}
    beta = {0: 1.0, 1: 1.5, 2: 2.0}
    alphas = [0.0, 0.5, 1.0]
    candidates, uniform = sr._candidate_allocations(
        infos, spectra, beta, 0.6, 0.5, alphas, "tail", "delta_kbar"
    )
    assert set(candidates) == {0.0, 0.5, 1.0}
    for ranks in list(candidates.values()) + [uniform]:
        for (_mtype, m, n), group in sr._group_infos(infos).items():
            budget = sr._kbar(m, n, 0.6) * len(group)
            assert sum(ranks[i.name] for i in group) == budget


def test_candidate_allocations_alpha_one_follows_importance():
    infos = _toy_infos()
    spectra = {i.name: torch.linspace(10.0, 0.1, min(i.m, i.n)) for i in infos}
    beta = {0: 1.0, 1: 1.5, 2: 2.0}  # block 2 most important
    candidates, _ = sr._candidate_allocations(
        infos, spectra, beta, 0.6, 0.5, [1.0], "tail", "delta_kbar"
    )
    ranks = candidates[1.0]
    q = [ranks[f"model.layers.{b}.q_proj"] for b in range(3)]
    assert q == sorted(q) and q[2] > q[0]


def test_param_ratio_matches_uniform_target():
    infos = _toy_infos()
    uniform = {i.name: sr._kbar(i.m, i.n, 0.6) for i in infos}
    # sum r(m+n) / sum mn at the uniform rank is the configured ratio (up to the
    # per-layer ceil), which is what collect_compression.py reports.
    assert sr._param_ratio(infos, uniform) == pytest.approx(0.6, abs=0.02)


# --------------------------------------------------------------------------- #
# Apply / restore
# --------------------------------------------------------------------------- #
def test_apply_and_restore_roundtrip():
    torch.manual_seed(8)
    infos = _toy_infos()[:2]
    snapshot = sr._snapshot_weights(infos)
    caches = {}
    for info in infos:
        rmax = min(info.m, info.n)
        U, _, _ = torch.linalg.svd(torch.randn(info.m, rmax), full_matrices=False)
        P = U.T @ info.module.weight.detach()
        caches[info.name] = (U, P)

    ranks = {info.name: 4 for info in infos}
    sr._apply_allocation(infos, caches, ranks, "cpu")
    for info in infos:
        U, P = caches[info.name]
        expected = U[:, :4] @ P[:4, :]
        torch.testing.assert_close(info.module.weight.data, expected)
        assert torch.linalg.matrix_rank(info.module.weight.data).item() <= 4

    sr._restore_weights(infos, snapshot)
    for info in infos:
        assert torch.equal(info.module.weight.data, snapshot[info.name])


def test_snapshot_is_a_copy_not_a_view():
    infos = _toy_infos()[:1]
    snapshot = sr._snapshot_weights(infos)
    infos[0].module.weight.data.zero_()
    assert snapshot[infos[0].name].abs().sum() > 0


# --------------------------------------------------------------------------- #
# Logging / checkpointing
# --------------------------------------------------------------------------- #
def test_candidate_log_line_format():
    assert (
        sr._candidate_log_line(0.3, 12.0456, True)
        == "  alpha 0.30: val_ppl=12.046 [best]"
    )
    assert sr._candidate_log_line(1.0, 9.5, False) == "  alpha 1.00: val_ppl=9.500"


def test_reference_line_matches_collect_regex():
    """The uniform reference line must stay parseable by the existing log tools."""
    import re

    ref_re = re.compile(
        r"(uniform-rank|init-ranks) reference \(val\): val_ppl=([\d.]+)"
    )
    line = f"  uniform-rank reference (val): val_ppl={12.345:.3f}"
    assert ref_re.search(line).group(2) == "12.345"


def _swift_cfg(tmp_path=None):
    cfg = OmegaConf.create(
        {
            "model": {"id": "mistralai/Mistral-7B-v0.1"},
            "compress": {
                "enabled": True,
                "strategy": "swift_svd",
                "ratio": 0.6,
                "target_modules": ["q_proj"],
                "params": {"accumulate": 8},
                "rank_source": "swift",
                "swift_rank_selection": {
                    "delta": 0.5,
                    "alphas": [0.0, 0.5, 1.0],
                    "score": "tail",
                    "base_rank": "delta_kbar",
                    "val": {"dataset": "c4", "num_samples": 8, "seq_len": 128},
                    "importance": {
                        "dataset": "c4",
                        "num_samples": 4,
                        "seq_len": 64,
                        "angular": False,
                    },
                    "cache_dtype": "float16",
                },
            },
            "calibration": {
                "dataset": "c4",
                "seq_len": 2048,
                "num_samples": 8,
                "batch_size": 2,
            },
            "eval": {
                "perplexity": {"enabled": False, "datasets": ["c4"]},
                "reasoning": {"enabled": False, "tasks": ["piqa"]},
            },
            "seed": 0,
        }
    )
    if tmp_path is not None:
        cfg.compress.swift_rank_selection.checkpoint = str(tmp_path / "ranks.json")
    return cfg


def test_swift_checkpoint_roundtrip(tmp_path):
    from pipeline import coala_rank

    cfg = _swift_cfg(tmp_path)
    ranks = {"model.layers.0.q_proj": 1229, "model.layers.1.q_proj": 900}
    path = tmp_path / "ranks.json"
    sr._save_swift_checkpoint(path, ranks, cfg)

    payload = json.loads(path.read_text())
    assert payload["model"] == "mistralai/Mistral-7B-v0.1"
    assert payload["ratio"] == 0.6
    assert payload["seq_len"] == 2048
    assert payload["ranks"] == ranks
    assert coala_rank._load_coala_checkpoint(str(path), cfg) == ranks


def test_grid_path_next_to_checkpoint():
    assert sr._grid_path("/a/b/ranks.json") == "/a/b/ranks_grid.json"
    assert sr._grid_path(None) is None


def test_ranks_survive_ratio_dict_roundtrip():
    """Swift ranks -> per-layer ratio -> COALA_Linear must recover them exactly."""
    from pipeline import coala_rank

    infos = _toy_infos()
    model = nn.Module()
    for info in infos:
        parts = info.name.split(".")
        parent = model
        for p in parts[:-1]:
            if not hasattr(parent, p):
                setattr(parent, p, nn.Module())
            parent = getattr(parent, p)
        setattr(parent, parts[-1], info.module)

    ranks = {i.name: sr._kbar(i.m, i.n, 0.6) for i in infos}
    ratios = coala_rank._ranks_to_ratio_dict(model, ranks)
    for info in infos:
        layer = COALA_Linear(
            ratio=float(ratios[info.name]),
            compress_strategy="empty",
            base_tensor=torch.zeros(info.m, info.n),
        )
        assert layer.r == ranks[info.name]


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_swift_uniform_config_needs_no_block():
    cfg = _swift_cfg()
    cfg.compress.rank_source = "uniform"
    del cfg.compress.swift_rank_selection
    validate_config(cfg, cuda_available=True)


def test_swift_dynamic_config_passes():
    validate_config(_swift_cfg(), cuda_available=True)


def test_swift_requires_cuda():
    with pytest.raises(ValueError, match="requires a CUDA device"):
        validate_config(_swift_cfg(), cuda_available=False)


def test_swift_rank_source_requires_swift_strategy():
    cfg = _swift_cfg()
    cfg.compress.strategy = "asvd"
    cfg.compress.params = {"accumulate": 8, "alpha": 0.5}
    with pytest.raises(ValueError, match="rank_source=swift requires"):
        validate_config(cfg, cuda_available=True)


def test_swift_rejects_train_rank_source():
    cfg = _swift_cfg()
    cfg.compress.rank_source = "train"
    with pytest.raises(ValueError, match="strategy=swift_svd supports"):
        validate_config(cfg, cuda_available=True)


def test_swift_checkpoint_source_requires_path():
    cfg = _swift_cfg()
    cfg.compress.rank_source = "checkpoint"
    with pytest.raises(ValueError, match="checkpoint path is required"):
        validate_config(cfg, cuda_available=True)


def test_swift_missing_block_rejected():
    cfg = _swift_cfg()
    del cfg.compress.swift_rank_selection
    with pytest.raises(ValueError, match="swift_rank_selection block is required"):
        validate_config(cfg, cuda_available=True)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("delta", 0.0, "delta"),
        ("delta", 1.5, "delta"),
        ("alphas", [], "alphas"),
        ("alphas", [1.5], "alphas entries"),
        ("score", "bogus", "score must be one of"),
        ("base_rank", "bogus", "base_rank must be one of"),
        ("cache_dtype", "float8", "cache_dtype"),
    ],
)
def test_swift_bad_knobs_rejected(key, value, match):
    cfg = _swift_cfg()
    cfg.compress.swift_rank_selection[key] = value
    with pytest.raises(ValueError, match=match):
        validate_config(cfg, cuda_available=True)


@pytest.mark.parametrize("block", ["val", "importance"])
def test_swift_sub_block_validated(block):
    cfg = _swift_cfg()
    cfg.compress.swift_rank_selection[block].num_samples = 0
    with pytest.raises(ValueError, match=f"{block}.num_samples"):
        validate_config(cfg, cuda_available=True)

    cfg = _swift_cfg()
    cfg.compress.swift_rank_selection[block].dataset = "nope"
    with pytest.raises(ValueError, match=f"{block}.dataset"):
        validate_config(cfg, cuda_available=True)


def test_swift_rejects_adaptive_rank():
    cfg = _swift_cfg()
    cfg.compress.adaptive_rank = True
    with pytest.raises(ValueError, match="adaptive_rank"):
        validate_config(cfg, cuda_available=True)


def test_shipped_swift_configs_validate():
    for path in ("configs/swift_svd.yaml", "configs/swift_svd_dynamic.yaml"):
        cfg = OmegaConf.load(path)
        validate_config(cfg, cuda_available=True)
