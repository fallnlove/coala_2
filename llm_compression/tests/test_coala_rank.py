"""COALA learned-rank logic — pure CPU (no CUDA)."""

import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from pipeline._paths import add_reference_paths

add_reference_paths()

import pipeline.coala_rank as cr  # noqa: E402
from coala.Linear import COALA_Linear  # noqa: E402
from pipeline.config import validate_config  # noqa: E402


# --------------------------------------------------------------------------- #
# Midpoint ratio round-trip: COALA_Linear must recover exactly the learned rank.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "m,n", [(4096, 4096), (14336, 4096), (4096, 14336), (1024, 4096)]
)
@pytest.mark.parametrize("r", [1, 2, 50, 100, 819, 1598])
def test_midpoint_ratio_roundtrip(m, n, r):
    if r > min(m, n):
        pytest.skip("rank exceeds matrix dims")
    ratio = (r - 0.5) * (m + n) / (m * n)
    # store/reload through float32 like the JSON checkpoint does
    ratio = float(torch.tensor(ratio, dtype=torch.float32).item())
    layer = COALA_Linear(
        ratio=ratio, compress_strategy="empty", base_tensor=torch.zeros(m, n)
    )
    assert layer.r == r


def test_ranks_to_ratio_dict_covers_targets():
    from transformers import AutoModelForCausalLM

    from tests.conftest import TINY_MODEL

    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    targets = cr._target_linears(model, ["q_proj", "k_proj", "v_proj", "o_proj"])
    rank_dict = {name: 3 for name, _ in targets}
    ratios = cr._ranks_to_ratio_dict(model, rank_dict)
    assert set(ratios) == set(rank_dict)
    # every emitted ratio recovers its rank
    named = dict(model.named_modules())
    for name, ratio in ratios.items():
        mod = named[name]
        layer = COALA_Linear(
            ratio=ratio,
            compress_strategy="empty",
            base_tensor=torch.zeros(mod.out_features, mod.in_features),
        )
        assert layer.r == rank_dict[name]


# --------------------------------------------------------------------------- #
# Differentiable gate layer (cans_svd runs on CPU).
# --------------------------------------------------------------------------- #
def _factors(use_reg, adaptive, K=5):
    torch.manual_seed(0)
    W = torch.randn(12, 8, dtype=torch.float32)  # [out, in]
    X = torch.randn(8, 16, dtype=torch.float32)  # [in, samples]
    U_K, P_K = cr._build_factors(
        W,
        X,
        K=K,
        r_uni=3,
        use_reg=use_reg,
        adaptive=adaptive,
        mu=1.0,
        cans_eps=1e-8,
        cans_diff="tikhonov",
    )
    return W, X, U_K, P_K


def test_layer_forward_dtype_and_grad():
    _, _, U_K, P_K = _factors(use_reg=True, adaptive=False)
    layer = cr.COALARankLayer(
        U_K,
        P_K,
        gamma_init=3.0,
        beta=10,
        max_rank=5,
        bias=None,
        name="t",
        device="cpu",
        in_features=8,
        out_features=12,
    )
    for p in layer.parameters():
        p.requires_grad = False
    layer.gamma.requires_grad = True
    x = torch.randn(1, 4, 8, dtype=torch.float32)
    out = layer(x)
    assert out.dtype == torch.float32 and out.shape == (1, 4, 12)
    (out**2).sum().backward()
    assert torch.isfinite(layer.gamma.grad).all() and float(layer.gamma.grad) != 0.0


def test_soft_rank_differentiable():
    _, _, U_K, P_K = _factors(use_reg=False, adaptive=False)
    layer = cr.COALARankLayer(
        U_K,
        P_K,
        gamma_init=2.5,
        beta=10,
        max_rank=5,
        bias=None,
        name="t",
        device="cpu",
        in_features=8,
        out_features=12,
    )
    layer.gamma.requires_grad = True
    layer.gate().sum().backward()
    assert torch.isfinite(layer.gamma.grad).all() and float(layer.gamma.grad) != 0.0


def test_full_gate_matches_coala_projection():
    # gamma >> K => gate ~ all ones => W' ~ U_K U_Kᵀ W (the rank-K COALA approx).
    K = 5
    W, X, U_K, P_K = _factors(use_reg=False, adaptive=False, K=K)
    layer = cr.COALARankLayer(
        U_K,
        P_K,
        gamma_init=float(K + 50),
        beta=10,
        max_rank=K,
        bias=None,
        name="t",
        device="cpu",
        in_features=8,
        out_features=12,
    )
    with torch.no_grad():
        wp = layer.U_K @ (layer.gate().unsqueeze(1) * layer.P_K)
    # reference: top-K left singular subspace of W@S.T (plain torch SVD, no gesvd)
    _, S = torch.linalg.qr(X.T, mode="r")
    Uref, _, _ = torch.linalg.svd(W @ S.T, full_matrices=False)
    Uref_K = Uref[:, :K]
    proj = Uref_K @ (Uref_K.T @ W)
    rel = torch.linalg.norm(wp - proj) / torch.linalg.norm(proj)
    assert rel < 0.05  # cans_svd is approximate; subspace projection matches closely


# --------------------------------------------------------------------------- #
# Low-memory rank training: break-even rank cap + fp16 factor storage.
# --------------------------------------------------------------------------- #
def _rank_layer(K, factor_dtype=None, gamma_init=3.0, bias=None):
    _, _, U_K, P_K = _factors(use_reg=False, adaptive=False, K=K)
    return cr.COALARankLayer(
        U_K,
        P_K,
        gamma_init=gamma_init,
        beta=10,
        max_rank=K,
        bias=bias,
        name="t",
        device="cpu",
        in_features=8,
        out_features=12,
        factor_dtype=factor_dtype,
    )


@pytest.mark.parametrize(
    "m,n", [(4096, 4096), (14336, 4096), (4096, 14336), (1024, 4096), (5120, 25600)]
)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.6, 0.7, 0.8, 0.95])
def test_break_even_rank_cap(m, n, ratio):
    r_uni = max(1, math.ceil(ratio * m * n / (m + n)))
    r_cap = math.floor(m * n / (m + n))
    K_off = cr._factor_max_rank(m, n, r_uni, 2.0, low_mem=False)
    K_on = cr._factor_max_rank(m, n, r_uni, 2.0, low_mem=True)
    # low-mem never exceeds break-even, never exceeds the uncapped K, stays >= 1
    assert K_on <= r_cap
    assert K_on <= K_off
    assert K_on >= 1
    # break-even factors never exceed the original weight params
    assert (m + n) * K_on <= m * n
    # gamma_init = min(r_uni, K) is always a valid gate index (<= K), even when
    # r_uni > r_cap (ratio near 1)
    assert min(r_uni, K_on) <= K_on
    # for ratio > 0.5 the cap actively binds (uncapped K ~ 2*r_uni > r_cap)
    if ratio > 0.5:
        assert K_on < K_off


def test_default_factor_dtype_is_fp32():
    layer = _rank_layer(K=5)  # factor_dtype=None -> COMPUTE_DTYPE
    assert layer.U_K.dtype == torch.float32 and layer.P_K.dtype == torch.float32


def test_fp16_factors_stored_half_forward_matches_fp32():
    fp32 = _rank_layer(K=5, factor_dtype=None)
    fp16 = _rank_layer(K=5, factor_dtype=torch.float16)
    assert fp16.U_K.dtype == torch.float16 and fp16.P_K.dtype == torch.float16
    x = torch.randn(1, 4, 8, dtype=torch.float32)
    with torch.no_grad():
        y32, y16 = fp32(x), fp16(x)
    assert y16.dtype == torch.float32  # fp32 input -> fp32 output
    rel = torch.linalg.norm(y16 - y32) / torch.linalg.norm(y32)
    assert rel < 1e-2  # fp16 storage rounding only


def test_fp16_factors_bf16_input_keeps_output_dtype_and_fp32_grad():
    layer = _rank_layer(K=5, factor_dtype=torch.float16)
    for p in layer.parameters():
        p.requires_grad = False
    layer.gamma.requires_grad = True
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)  # bf16 residual stream
    out = layer(x)
    assert out.dtype == torch.bfloat16 and out.shape == (1, 4, 12)
    (out.float() ** 2).sum().backward()
    # gamma gradient stays fp32 and finite despite fp16 factors + bf16 activations
    assert layer.gamma.grad.dtype == torch.float32
    assert torch.isfinite(layer.gamma.grad).all() and float(layer.gamma.grad) != 0.0


def test_fp16_factors_checkpointed_grad_matches_direct():
    # low_mem forward routes through torch.utils.checkpoint when grads are on
    # (so backward recomputes the fp32 factor upcasts instead of retaining
    # them). The checkpointed gamma gradient must match the direct path.
    torch.manual_seed(0)
    a = _rank_layer(K=5, factor_dtype=torch.float16)
    b = _rank_layer(K=5, factor_dtype=torch.float16)  # same seed -> same factors
    for layer in (a, b):
        for p in layer.parameters():
            p.requires_grad = False
        layer.gamma.requires_grad = True
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    out_ckpt = a(x)  # grad enabled -> checkpoint path
    (out_ckpt.float() ** 2).sum().backward()
    out_direct = b._forward_impl(x)  # bypasses checkpointing
    (out_direct.float() ** 2).sum().backward()
    assert torch.equal(out_ckpt, out_direct)
    assert torch.allclose(a.gamma.grad, b.gamma.grad, rtol=1e-6, atol=1e-8)
    # no-grad path (val/reference) takes the direct route and stays finite
    with torch.no_grad():
        y = a(x)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()


def test_fp16_factors_hard_rank_forward():
    layer = _rank_layer(K=5, factor_dtype=torch.float16)
    layer.hard_rank = 3
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    out = layer(x)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# ratio_objective=ranks + init_checkpoint warm start.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "m,n", [(4096, 4096), (14336, 4096), (4096, 14336), (1024, 4096)]
)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7, 0.95])
def test_uniform_rank_both_objectives(m, n, ratio):
    r_params = cr._uniform_rank(m, n, ratio, "params")
    assert r_params == max(1, math.ceil(ratio * m * n / (m + n)))  # current formula
    r_ranks = cr._uniform_rank(m, n, ratio, "ranks")
    assert r_ranks == max(1, min(math.ceil(ratio * m), m, n))
    assert 1 <= r_ranks <= min(m, n)
    # the infeasible-wide-layer clamp: e.g. up/gate-shaped m=14336,n=4096 @0.7
    if ratio * m > min(m, n):
        assert r_ranks == min(m, n)


def test_compression_loss_ranks_objective_exact_and_differentiable():
    # half-integer gammas: at high beta the gate saturates to exactly 1 for
    # positions <= floor(gamma) and 0 above (an integer gamma would leave its
    # own position's tanh at 0.5).
    layers = [_rank_layer(K=5, gamma_init=3.5), _rank_layer(K=5, gamma_init=2.5)]
    model = torch.nn.Sequential(*layers)
    for la in layers:
        la.beta = 1000.0
        la.gamma.requires_grad = True
    # sum r / sum out_features = (3 + 2) / (12 + 12)
    want = (3.0 + 2.0) / 24.0
    loss, ratio = cr._compression_loss_coala(model, want, 100.0, "cpu", "ranks")
    assert abs(float(ratio) - want) < 1e-4
    assert float(loss) < 1e-2  # ~0 at target
    loss2, _ = cr._compression_loss_coala(model, 0.9, 100.0, "cpu", "ranks")
    loss2.backward()
    assert all(torch.isfinite(la.gamma.grad).all() for la in layers)


def test_compression_loss_params_default_unchanged():
    layers = [_rank_layer(K=5, gamma_init=3.0)]
    model = torch.nn.Sequential(*layers)
    # explicit params == no-arg default, and matches the hand formula
    l_def, r_def = cr._compression_loss_coala(model, 0.5, 200.0, "cpu")
    l_par, r_par = cr._compression_loss_coala(model, 0.5, 200.0, "cpu", "params")
    assert float(l_def) == float(l_par) and float(r_def) == float(r_par)
    soft = float(layers[0].gate().sum())
    want = (8 + 12) * soft / (8 * 12)
    assert abs(float(r_def) - want) < 1e-6


def test_gamma_init_from_checkpoint():
    loaded = {"a": 4, "b": 2.6, "c": 99, "d": 0}
    assert cr._gamma_init_from(loaded, "a", 3, 8) == (4.0, True)  # exact hit
    assert cr._gamma_init_from(loaded, "b", 3, 8) == (3.0, True)  # float rounds
    assert cr._gamma_init_from(loaded, "c", 3, 8) == (8.0, True)  # clamps to K
    assert cr._gamma_init_from(loaded, "d", 3, 8) == (1.0, True)  # clamps to 1
    assert cr._gamma_init_from(loaded, "zz", 3, 8) == (3.0, False)  # fallback
    assert cr._gamma_init_from(None, "a", 3, 8) == (3.0, False)  # no file


def test_ratio_objective_config_validation():
    from pipeline.config import load_config

    base = ["compress.rank_source=train"]
    for good in (
        base + ["compress.coala_rank_selection.ratio_objective=ranks"],
        base + ["compress.coala_rank_selection.init_checkpoint=./ranks.json"],
        base
        + [
            "compress.coala_rank_selection.ratio_objective=ranks",
            "compress.coala_rank_selection.trainer=reinforce",
            "compress.coala_rank_selection.rf_budget=penalty",
        ],
    ):
        validate_config(
            load_config("configs/coala_ranktrain.yaml", good), cuda_available=True
        )
    for bad in (
        base + ["compress.coala_rank_selection.ratio_objective=bogus"],
        base
        + [
            "compress.coala_rank_selection.ratio_objective=ranks",
            "compress.coala_rank_selection.trainer=reinforce",
            "compress.coala_rank_selection.rf_budget=project",
        ],
    ):
        with pytest.raises(ValueError):
            validate_config(
                load_config("configs/coala_ranktrain.yaml", bad), cuda_available=True
            )


def test_low_mem_requires_half_precision_load_dtype():
    from pipeline.config import load_config

    base = [
        "compress.rank_source=train",
        "compress.coala_rank_selection.low_mem=true",
    ]
    # bf16 / fp16 load dtype is accepted
    for dt in ("bfloat16", "float16"):
        cfg = load_config(
            "configs/coala_ranktrain.yaml", base + [f"model.load_dtype={dt}"]
        )
        validate_config(cfg, cuda_available=True)
    # fp32 load dtype (the config default) is rejected in low-mem mode
    cfg = load_config(
        "configs/coala_ranktrain.yaml", base + ["model.load_dtype=float32"]
    )
    with pytest.raises(ValueError):
        validate_config(cfg, cuda_available=True)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def _coala_ranks_base():
    return OmegaConf.create(
        {
            "model": {"id": "x"},
            "compress": {
                "enabled": True,
                "strategy": "coala",
                "ratio": 0.5,
                "target_modules": ["q_proj"],
                "params": {"mu": 1.0, "adaptive": True, "log_norms": False},
                "rank_source": "train",
                "coala_rank_selection": {
                    "dataset": "c4",
                    "num_samples": 8,
                    "n_epochs": 2,
                    "seq_len": 32,
                    "lr": 5.0,
                    "beta": 10,
                    "lambda_reg": 200,
                    "cans_eps": 1e-8,
                    "max_rank_ratio": 2.0,
                    "checkpoint": "./outputs/coala_ranks.json",
                },
            },
            "calibration": {"dataset": "c4", "num_samples": 8, "batch_size": 4},
            "eval": {"perplexity": {"enabled": False}, "reasoning": {"enabled": False}},
        }
    )


def test_coala_train_config_valid():
    validate_config(_coala_ranks_base(), cuda_available=True)


def test_coala_train_requires_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        validate_config(_coala_ranks_base(), cuda_available=False)


def test_coala_unknown_rank_source_rejected():
    cfg = _coala_ranks_base()
    cfg.compress.rank_source = "bogus"
    with pytest.raises(ValueError, match="rank_source"):
        validate_config(cfg, cuda_available=True)


def test_coala_checkpoint_source_requires_path():
    cfg = _coala_ranks_base()
    cfg.compress.rank_source = "checkpoint"
    cfg.compress.coala_rank_selection.checkpoint = None
    with pytest.raises(ValueError, match="checkpoint path"):
        validate_config(cfg, cuda_available=True)


def test_coala_rank_source_uniform_unconstrained():
    cfg = _coala_ranks_base()
    cfg.compress.rank_source = "uniform"
    validate_config(cfg, cuda_available=True)  # no rank-selection block needed


# --------------------------------------------------------------------------- #
# Checkpoint stores integer ranks and round-trips
# --------------------------------------------------------------------------- #
def test_coala_checkpoint_integer_roundtrip(tmp_path):
    cfg = OmegaConf.create(
        {
            "model": {"id": "m"},
            "compress": {"ratio": 0.5, "coala_rank_selection": {"seq_len": 2048}},
        }
    )
    path = tmp_path / "ranks.json"
    ranks = {"model.layers.0.self_attn.q_proj": 123, "model.layers.0.mlp.up_proj": 45}
    cr._save_coala_checkpoint(str(path), ranks, cfg)
    loaded = cr._load_coala_checkpoint(str(path), cfg)
    assert loaded == ranks
    assert all(isinstance(v, int) for v in loaded.values())


# --------------------------------------------------------------------------- #
# Validation perplexity + early stopping
# --------------------------------------------------------------------------- #
def test_early_stopper_snapshot_stop_restore():
    import torch.nn as nn

    g = [nn.Parameter(torch.tensor(3.0))]
    es = cr._EarlyStopper(g, patience=2, min_delta=0.0)
    is_best, stop = es.update(100.0)
    assert is_best and not stop  # first val -> best; snapshot taken at gamma=3
    g[0].data.fill_(9.0)  # gamma drifts away from the best snapshot
    is_best, stop = es.update(101.0)
    assert (not is_best) and not stop  # bad epoch #1
    is_best, stop = es.update(102.0)
    assert (not is_best) and stop  # bad epoch #2 -> patience hit -> stop
    es.restore_best()
    assert float(g[0]) == 3.0  # best-val gammas restored


def test_early_stopper_disabled_and_nonfinite():
    import torch.nn as nn

    g = [nn.Parameter(torch.tensor(3.0))]
    es = cr._EarlyStopper(g, patience=0, min_delta=0.0)  # disabled
    for v in (100.0, 200.0, 300.0):
        _, stop = es.update(v)
        assert not stop
    es2 = cr._EarlyStopper(g, patience=1)
    is_best, stop = es2.update(float("nan"))  # non-finite val ignored
    assert (not is_best) and (not stop) and es2.n_bad == 0


def test_min_delta_requires_real_improvement():
    import torch.nn as nn

    g = [nn.Parameter(torch.tensor(1.0))]
    es = cr._EarlyStopper(g, patience=1, min_delta=1.0)
    es.update(100.0)
    is_best, stop = es.update(99.5)  # improves by 0.5 < min_delta=1.0 -> not best
    assert (not is_best) and stop


def test_epoch_log_line_format():
    means = {"ppl": 10.0, "reg": 1.0, "value": 0.0, "ratio": 0.5}
    line = cr._epoch_log_line(0, means, 12.5, is_best=True)
    assert "epoch 1" in line
    assert "train_ppl=10.000" in line and "val_ppl=12.500" in line
    assert "[best]" in line
    line2 = cr._epoch_log_line(2, means, float("nan"), is_best=False)
    assert "epoch 3" in line2 and "val_ppl=n/a" in line2 and "[best]" not in line2


def test_train_gammas_early_stop_and_desc():
    """The shared trainer breaks when on_epoch_end returns truthy; desc is accepted."""
    import torch.nn as nn

    from pipeline import dobi

    g = nn.Parameter(torch.tensor(5.0))

    class _Fake(nn.Module):
        def forward(self, input_ids):
            return (g**2).sum()  # scalar with grad wrt g; ignores input

    rs = OmegaConf.create({"n_epochs": 10, "lr": 0.1, "scheduler_min_lr": 0.0})
    seen = []

    def cb(epoch):
        seen.append(epoch)
        return epoch >= 1  # request stop after the 2nd epoch (index 1)

    dobi._train_gammas(
        _Fake(),
        [g],
        [torch.tensor([1, 2, 3])],
        rs,
        "cpu",
        progress=False,
        on_epoch_end=cb,
        desc="COALA rank-train",
    )
    assert seen == [0, 1]  # loop broke right after cb returned True


# --------------------------------------------------------------------------- #
# Hard-rank truncation (uniform-rank reference) + eval-protocol validation
# --------------------------------------------------------------------------- #
def _layer(K=5, gamma_init=3.0):
    _, _, U_K, P_K = _factors(use_reg=False, adaptive=False, K=K)
    return cr.COALARankLayer(
        U_K,
        P_K,
        gamma_init=gamma_init,
        beta=10,
        max_rank=K,
        bias=None,
        name="t",
        device="cpu",
        in_features=8,
        out_features=12,
    )


def test_hard_rank_truncation_matches_manual():
    layer = _layer(K=5)
    r = 3
    layer.hard_rank = r
    x = torch.randn(1, 4, 8, dtype=torch.float32)
    out = layer(x)
    w_ref = layer.U_K[:, :r] @ layer.P_K[:r, :]  # hard rank-r COALA weight
    assert torch.allclose(out, x @ w_ref.T, atol=1e-5)
    # bypasses the gate: differs from the soft-gated forward
    layer.hard_rank = None
    assert not torch.allclose(out, layer(x), atol=1e-4)


def test_hard_rank_clamped_to_bounds():
    layer = _layer(K=5)
    x = torch.randn(1, 8, dtype=torch.float32)
    layer.hard_rank = 99  # > max_rank -> clamp to K
    hi = layer(x)
    layer.hard_rank = 5
    assert torch.allclose(hi, layer(x), atol=1e-6)
    layer.hard_rank = 0  # < 1 -> clamp to 1
    lo = layer(x)
    layer.hard_rank = 1
    assert torch.allclose(lo, layer(x), atol=1e-6)


def test_uniform_reference_sets_then_resets_hard_rank():
    l1 = _layer(K=5)
    l1.name = "a"
    l2 = _layer(K=5)
    l2.name = "b"
    seen = {}

    class _Wrap:
        model = SimpleNamespace(modules=lambda: [l1, l2])

        def val_perplexity(self, windows, device):
            seen["a"], seen["b"] = l1.hard_rank, l2.hard_rank
            return 42.0

    out = cr._uniform_reference_ppl(_Wrap(), {"a": 2, "b": 4}, [object()], "cpu")
    assert out == 42.0
    assert seen == {"a": 2, "b": 4}  # hard_rank = r_uni during the measurement
    assert l1.hard_rank is None and l2.hard_rank is None  # reset afterwards


def test_uniform_reference_empty_windows_is_nan():
    out = cr._uniform_reference_ppl(object(), {}, [], "cpu")
    assert math.isnan(out)


def test_val_perplexity_uses_eval_protocol():
    from transformers import AutoModelForCausalLM

    from pipeline import data as data_mod
    from pipeline import evaluate as evaluate_mod
    from tests.conftest import TINY_MODEL

    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    cfg = OmegaConf.create(
        {"compress": {"ratio": 0.5, "coala_rank_selection": {"lambda_reg": 200}}}
    )
    wrap = cr._CoalaRankTrainWrapper(model, cfg)
    torch.manual_seed(0)
    vocab = int(model.config.vocab_size)
    windows = [torch.randint(0, vocab, (16,)) for _ in range(3)]
    got = wrap.val_perplexity(windows, "cpu")
    # routes through evaluate.evaluate_perplexity on the stacked windows
    expect = evaluate_mod.evaluate_perplexity(model, data_mod.ppl_input_ids(windows))
    assert math.isfinite(got) and abs(got - expect) < 1e-3


def test_dobi_val_perplexity_uses_eval_protocol():
    from transformers import AutoModelForCausalLM

    from pipeline import data as data_mod
    from pipeline import dobi
    from pipeline import evaluate as evaluate_mod
    from tests.conftest import TINY_MODEL

    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    cfg = OmegaConf.create(
        {
            "compress": {
                "ratio": 0.8,
                "dobi": {"rank_selection": {"lambda_reg": 200, "seq_len": 16}},
            }
        }
    )
    wrap = dobi._DobiTrainWrapper(model, cfg)
    assert math.isnan(wrap.val_perplexity([], "cpu"))  # val disabled -> nan
    torch.manual_seed(0)
    vocab = int(model.config.vocab_size)
    windows = [torch.randint(0, vocab, (16,)) for _ in range(3)]
    got = wrap.val_perplexity(windows, "cpu")
    expect = evaluate_mod.evaluate_perplexity(model, data_mod.ppl_input_ids(windows))
    assert math.isfinite(got) and abs(got - expect) < 1e-3


def test_early_stopper_shared_between_pipelines():
    from pipeline import dobi

    # coala_rank aliases dobi's implementations (single shared copy).
    assert cr._EarlyStopper is dobi._EarlyStopper
    assert cr._epoch_log_line is dobi._epoch_log_line
