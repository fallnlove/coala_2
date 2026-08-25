"""coala-reinforce (discrete-rank REINFORCE trainer, v2) — pure CPU (no CUDA)."""

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pipeline._paths import add_reference_paths

add_reference_paths()

import pipeline.coala_rank as cr  # noqa: E402
import pipeline.coala_reinforce as crf  # noqa: E402
from pipeline import dobi  # noqa: E402
from pipeline.config import load_config, validate_config  # noqa: E402
from tests.test_coala_rank import _factors  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _rank_layer(name, K=8, gamma_init=4.0):
    _, _, U_K, P_K = _factors(use_reg=False, adaptive=False, K=K)
    return cr.COALARankLayer(
        U_K,
        P_K,
        gamma_init=gamma_init,
        beta=10,
        max_rank=K,
        bias=None,
        name=name,
        device="cpu",
        in_features=8,
        out_features=12,
    )


class _FakeRankModel(nn.Module):
    """Stand-in LM: loss/logits/hiddens are DETERMINISTIC functions of the
    layers' hard ranks (random outputs would make the KL/HID tests flaky)."""

    def __init__(self, layers, loss_fn, vocab=16, logits_fn=None, hidden_fn=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self._loss_fn = loss_fn
        self._vocab = vocab
        self._logits_fn = logits_fn
        self._hidden_fn = hidden_fn
        self.device = torch.device("cpu")

    def _ranks(self):
        return [
            layer.hard_rank if layer.hard_rank is not None else -1
            for layer in self.layers
        ]

    def forward(
        self, input_ids=None, labels=None, use_cache=None, output_hidden_states=False
    ):
        ranks = self._ranks()
        loss = torch.tensor(float(self._loss_fn(ranks)), dtype=torch.float32)
        T = input_ids.shape[1]
        if self._logits_fn is not None:
            logits = self._logits_fn(ranks, T, self._vocab)
        else:
            logits = torch.zeros(1, T, self._vocab)
            logits[..., 0] = 0.1 * sum(ranks)
        hiddens = None
        if output_hidden_states:
            if self._hidden_fn is not None:
                hiddens = self._hidden_fn(ranks, T)
            else:
                hiddens = (
                    torch.zeros(1, T, 4),  # "embeddings" (excluded from HID)
                    torch.full((1, T, 4), float(sum(ranks))),
                )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hiddens)


class _FakeTeacher(nn.Module):
    """Fixed-output teacher that counts its forward calls."""

    def __init__(self, logits, hiddens=None):
        super().__init__()
        self._logits = logits
        self._hiddens = hiddens
        self.calls = 0

    def forward(self, input_ids=None, use_cache=None, output_hidden_states=False):
        self.calls += 1
        return SimpleNamespace(
            logits=self._logits,
            hidden_states=self._hiddens if output_hidden_states else None,
        )


def _rf_cfg(ratio=0.5, **overrides):
    rs = {
        "dataset": "c4",
        "lambda_reg": 0.0,
        "n_epochs": 30,
        "trainer": "reinforce",
        "rf_samples": 2,
        "rf_budget": "penalty",
        "rf_w_ce": 1.0,
        "rf_w_kl": 0.0,
        "rf_w_hidden": 0.0,
        "rf_w_entropy": 0.0,
        "rf_learn_sigma": False,
        "rf_sigma_lr_scale": 0.1,
        "rf_sigma_init": 0.05,
        "rf_sigma_final": 0.005,
        "rf_sigma_min": 1.0,
        "rf_adv_ema_beta": 0.99,
        "rf_adv_std_floor": 1e-3,
    }
    rs.update(overrides)
    return OmegaConf.create(
        {
            "seed": 0,
            "compress": {"ratio": ratio, "coala_rank_selection": rs},
            "calibration": {"dataset": "wikitext2"},
        }
    )


def _make_wrapper(
    loss_fn,
    names=("a", "b"),
    K=8,
    gamma_init=4.0,
    ratio=0.5,
    teacher=None,
    model_kwargs=None,
    **cfg_overrides,
):
    layers = [_rank_layer(n, K=K, gamma_init=gamma_init) for n in names]
    model = _FakeRankModel(layers, loss_fn, **(model_kwargs or {}))
    for p in model.parameters():
        p.requires_grad = False
    for layer in layers:
        layer.gamma.requires_grad = True
    cfg = _rf_cfg(ratio=ratio, **cfg_overrides)
    r_unis = {n: gamma_init for n in names}
    wrapper = crf._CoalaReinforceTrainWrapper(model, cfg, r_unis, teacher=teacher)
    return wrapper, layers


IDS = torch.tensor([[1, 2, 3, 4]])


# --------------------------------------------------------------------------- #
# The pmf and its gradients (unchanged math from v1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [1, 5, 7, 50])
@pytest.mark.parametrize("sigma", [1.0, 2.5, 30.0])
@pytest.mark.parametrize("mu_off", [-2.0, 0.6, 3.3, 0.5, 3.0])
def test_pmf_sums_to_one(K, sigma, mu_off):
    mu = torch.tensor([mu_off if mu_off < K else K / 2.0], dtype=torch.float64)
    r = torch.arange(1, K + 1, dtype=torch.float64).unsqueeze(1)
    logp = crf._discretized_gaussian_logpmf(
        r,
        mu.expand(K, 1),
        torch.tensor([sigma], dtype=torch.float64),
        torch.tensor([float(K)], dtype=torch.float64),
    )
    assert abs(logp.exp().sum().item() - 1.0) < 1e-9


def test_sampler_matches_pmf():
    K, mu_v, sigma_v, n = 7, 3.3, 1.2, 20000
    mu = torch.tensor([mu_v], dtype=torch.float64)
    sigma = torch.tensor([sigma_v], dtype=torch.float64)
    Kt = torch.tensor([float(K)], dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    samples = crf._sample_gaussian(mu, sigma, Kt, gen, n)  # [n, 1]
    counts = torch.bincount(samples.long().flatten() - 1, minlength=K).float()
    freq = counts / n
    pmf = torch.tensor(
        [
            crf._discretized_gaussian_logpmf(
                torch.tensor([float(r)], dtype=torch.float64), mu, sigma, Kt
            )
            .exp()
            .item()
            for r in range(1, K + 1)
        ]
    )
    assert (freq - pmf).abs().max().item() < 0.02


@pytest.mark.parametrize(
    "r_val,mu_v,sigma_v,K",
    [
        (4.0, 4.2, 1.5, 9),
        (1.0, 3.0, 1.5, 9),
        (9.0, 6.5, 1.5, 9),
        (13.0, 4.0, 1.5, 20),  # |r-mu| = 6 sigma
    ],
)
def test_logpmf_gradient_matches_finite_difference(r_val, mu_v, sigma_v, K):
    sigma = torch.tensor([sigma_v], dtype=torch.float64)
    Kt = torch.tensor([float(K)], dtype=torch.float64)
    r = torch.tensor([r_val], dtype=torch.float64)
    mu = torch.tensor([mu_v], dtype=torch.float64, requires_grad=True)
    crf._discretized_gaussian_logpmf(r, mu, sigma, Kt).sum().backward()
    grad = float(mu.grad)
    h = 1e-4  # large enough that far-tail fp64 CDF rounding doesn't dominate
    lo = crf._discretized_gaussian_logpmf(
        r, torch.tensor([mu_v - h], dtype=torch.float64), sigma, Kt
    ).item()
    hi = crf._discretized_gaussian_logpmf(
        r, torch.tensor([mu_v + h], dtype=torch.float64), sigma, Kt
    ).item()
    fd = (hi - lo) / (2 * h)
    assert grad != 0.0
    assert abs(grad - fd) < 1e-4 * max(1.0, abs(fd))


def test_logpmf_gradient_flows_to_sigma():
    Kt = torch.tensor([9.0], dtype=torch.float64)
    mu = torch.tensor([4.0], dtype=torch.float64)
    sigma = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    r = torch.tensor([6.0], dtype=torch.float64)
    crf._discretized_gaussian_logpmf(r, mu, sigma, Kt).sum().backward()
    assert sigma.grad is not None and float(sigma.grad) != 0.0


def test_score_function_zero_mean():
    K, mu_v, sigma_v = 9, 4.2, 1.5
    sigma = torch.tensor([sigma_v], dtype=torch.float64)
    Kt = torch.tensor([float(K)], dtype=torch.float64)
    total = 0.0
    for r_val in range(1, K + 1):
        mu = torch.tensor([mu_v], dtype=torch.float64, requires_grad=True)
        logp = crf._discretized_gaussian_logpmf(
            torch.tensor([float(r_val)], dtype=torch.float64), mu, sigma, Kt
        ).sum()
        logp.backward()
        total += logp.exp().item() * float(mu.grad)
    assert abs(total) < 1e-8


# --------------------------------------------------------------------------- #
# LOO estimator
# --------------------------------------------------------------------------- #
def test_rloo_advantages_sum_to_zero_and_n2_pairwise():
    advs = crf._rloo_advantages([1.0, 3.0, 5.0, 7.0])
    assert abs(sum(advs)) < 1e-12
    a2 = crf._rloo_advantages([2.0, 5.0])  # N=2 degenerates to pairwise diff
    assert a2 == [-3.0, 3.0]


def test_constant_reward_gives_zero_gradient():
    wrapper, layers = _make_wrapper(lambda ranks: 3.0)
    total = wrapper(input_ids=IDS)
    assert torch.isfinite(total)
    total.backward()
    for layer in layers:
        assert layer.gamma.grad is not None
        assert float(layer.gamma.grad) == 0.0  # all LOO advantages exactly 0


def test_gradient_direction_pushes_mu_toward_better_ranks():
    # loss decreases in rank => reward increases in rank => mean surrogate
    # gradient on mu must be negative (minimizing the surrogate raises mu).
    torch.manual_seed(0)
    wrapper, layers = _make_wrapper(
        lambda ranks: 5.0 - 0.5 * (ranks[0] - 7),
        names=("solo",),
        K=15,
        gamma_init=7.0,
        rf_samples=4,
    )
    grads = []
    for _ in range(200):
        total = wrapper(input_ids=IDS)
        if not torch.isfinite(total):
            continue
        total.backward()
        grads.append(float(layers[0].gamma.grad))
        layers[0].gamma.grad = None
    assert sum(grads) / len(grads) < 0.0


def test_toy_convergence_recovers_targets():
    torch.manual_seed(0)
    targets = {"a": 6, "b": 2}
    wrapper, layers = _make_wrapper(
        lambda ranks: 0.05 * sum((r - t) ** 2 for r, t in zip(ranks, targets.values())),
        names=tuple(targets),
        K=8,
        gamma_init=4.0,
        n_epochs=50,
    )
    gammas = [layer.gamma for layer in layers]
    rs = OmegaConf.create({"n_epochs": 50, "lr": 0.3, "scheduler_min_lr": 0.24})
    windows = [torch.tensor([1, 2, 3, 4]) for _ in range(40)]

    def _on_epoch(epoch):
        wrapper.advance_epoch()
        return False

    n_skipped, n_nonfinite = dobi._train_gammas(
        wrapper,
        gammas,
        windows,
        rs,
        "cpu",
        progress=False,
        on_epoch_end=_on_epoch,
        desc="toy reinforce",
    )
    assert n_skipped == 0 and n_nonfinite == 0
    ranks = wrapper.policy_rank_dict()
    for name, target in targets.items():
        assert abs(ranks[name] - target) <= 1, (name, ranks[name], target)


def test_ema_normalizer_first_step_and_update_after_use():
    torch.manual_seed(1)
    wrapper, layers = _make_wrapper(lambda ranks: 3.0 + 0.2 * ranks[0], names=("a",))
    assert wrapper._ema_t == 0
    wrapper(input_ids=IDS)
    assert wrapper._diag["n"] == 1
    # first window self-normalizes -> mean |adv| == 1 (unless the pair tied)
    assert wrapper._diag["absadv"] in (0.0, pytest.approx(1.0))
    assert wrapper._ema_t == 1  # EMA updated once per window, after use


# --------------------------------------------------------------------------- #
# Budget projection
# --------------------------------------------------------------------------- #
def _pw(vals):
    return torch.tensor(vals, dtype=torch.float64)


def test_project_int_feasible_monotone_idempotent():
    w, K = _pw([10.0, 1.0, 5.0]), _pw([10.0, 10.0, 10.0])
    r = _pw([8.0, 9.0, 6.0])
    budget = 60.0
    p = crf._project_budget_int(r, w, budget, K)
    assert float((w * p).sum()) <= budget + 1e-9
    assert bool((p <= r + 1e-9).all()) and bool((p >= 1.0).all())
    assert bool((p <= K).all())
    p2 = crf._project_budget_int(p, w, budget, K)
    assert torch.allclose(p, p2)  # idempotent


def test_project_int_noop_when_feasible_and_all_ones_guard():
    w, K = _pw([2.0, 2.0]), _pw([8.0, 8.0])
    r = _pw([3.0, 4.0])
    assert torch.allclose(crf._project_budget_int(r, w, 100.0, K), r)
    ones = crf._project_budget_int(r, w, 1.0, K)  # budget below all-ones floor
    assert torch.allclose(ones, torch.ones_like(r))


def test_project_int_fill_is_budget_aware():
    # After scaling, slack fits only the cheap layer's +1 — the expensive
    # layer must be skipped even if its fractional remainder is larger.
    w, K = _pw([10.0, 1.0]), _pw([20.0, 20.0])
    r = _pw([10.0, 10.0])
    budget = 55.0  # scale s=0.5 -> x=[5,5], cost 55 -> slack 0 -> no fill
    p = crf._project_budget_int(r, w, budget, K)
    assert float((w * p).sum()) <= budget + 1e-9
    budget = 58.0  # x=[5.27,5.27] floor [5,5] cost 55, slack 3 -> only w=1 fills
    p = crf._project_budget_int(r, w, budget, K)
    assert float((w * p).sum()) <= budget + 1e-9
    assert float(p[1]) >= float(p[0])  # cheap layer got the fill


def test_project_continuous_feasible_and_noop():
    w, K = _pw([10.0, 1.0]), _pw([20.0, 20.0])
    mu = _pw([10.0, 10.0])
    p = crf._project_budget_continuous(mu, w, 55.0, K)
    assert float((w * p).sum()) <= 55.0 + 1e-6
    assert bool((p >= 1.0).all())
    assert torch.allclose(crf._project_budget_continuous(mu, w, 1000.0, K), mu)


def test_project_mode_wrapper_keeps_everything_feasible():
    # ratio 0.15 -> budget = 0.15 * 2 * 96 = 28.8 weighted units; w = 20/layer
    # -> r_a + r_b <= 1.44... in rank units budget/20 = 1.44 -> forces ~1 each.
    wrapper, layers = _make_wrapper(
        lambda ranks: 3.0 + 0.01 * sum(ranks),
        ratio=0.3,
        rf_budget="project",
        lambda_reg=200.0,  # must be ignored in project mode
    )
    budget = wrapper._rf_budget_units
    total = wrapper(input_ids=IDS)
    assert torch.isfinite(total)
    # mu projected in place -> feasible up to fp32 parameter storage
    # quantization (the projection is exact in fp64; writing back into the
    # fp32 gamma params can round up by ~budget * 2^-24). The DEPLOYED ranks
    # go through the exact integer projection below regardless.
    mu_cost = sum(
        float(layer.gamma) * float(w)
        for layer, w in zip(layers, wrapper._rf_w.tolist())
    )
    assert mu_cost <= budget * (1 + 1e-6)
    # deterministic policy + extraction within budget
    ranks = wrapper.policy_rank_dict()
    cost = sum(
        ranks[layer.name] * float(w) for layer, w in zip(layers, wrapper._rf_w.tolist())
    )
    assert cost <= budget + 1e-6
    # post-window hard state == projected policy
    for layer, r in zip(layers, wrapper._policy_ranks().tolist()):
        assert layer.hard_rank == int(r)


def test_toy_budget_allocation_convergence():
    # reward favors layer a 3:1; budget allows ~4 total ranks -> a should end
    # above b, and every extracted allocation must respect the budget.
    torch.manual_seed(0)
    wrapper, layers = _make_wrapper(
        lambda ranks: 5.0 - 0.3 * ranks[0] - 0.1 * ranks[1],
        ratio=0.5,  # budget = 0.5 * 192 = 96 units; w = 20/layer -> sum r <= 4.8
        rf_budget="project",
        n_epochs=40,
    )
    gammas = [layer.gamma for layer in layers]
    rs = OmegaConf.create({"n_epochs": 40, "lr": 0.3, "scheduler_min_lr": 0.24})
    windows = [torch.tensor([1, 2, 3, 4]) for _ in range(40)]
    dobi._train_gammas(
        wrapper,
        gammas,
        windows,
        rs,
        "cpu",
        progress=False,
        on_epoch_end=lambda e: (wrapper.advance_epoch(), False)[1],
        desc="toy budget",
    )
    ranks = wrapper.policy_rank_dict()
    cost = sum(r * 20.0 for r in ranks.values())
    assert cost <= wrapper._rf_budget_units + 1e-6
    assert ranks["a"] > ranks["b"]


# --------------------------------------------------------------------------- #
# Learnable sigma + entropy
# --------------------------------------------------------------------------- #
def test_learned_sigma_param_init_and_floor():
    wrapper, _ = _make_wrapper(lambda ranks: 3.0, rf_learn_sigma=True)
    extra = wrapper.extra_trainable_params()
    assert len(extra) == 1 and extra[0].shape == (2,)
    sigma = wrapper._sigma_tensor()
    assert bool((sigma >= wrapper._sigma_min).all())
    # r_uni=4 -> 0.05*4=0.2 < floor 1.0 -> init just above the floor
    assert float(sigma.max()) < wrapper._sigma_min + 0.1


def test_learned_sigma_receives_gradient():
    wrapper, layers = _make_wrapper(
        lambda ranks: 3.0 + 0.3 * ranks[0], names=("a",), rf_learn_sigma=True
    )
    torch.manual_seed(3)
    for _ in range(10):
        total = wrapper(input_ids=IDS)
        if torch.isfinite(total):
            total.backward()
    rho = wrapper.extra_trainable_params()[0]
    assert rho.grad is not None and torch.isfinite(rho.grad).all()


def test_entropy_monotone_in_sigma_and_saturates():
    K = torch.tensor([15.0], dtype=torch.float64)
    mu = torch.tensor([8.0], dtype=torch.float64)
    h1 = crf._entropy_discretized(mu, torch.tensor([1.0], dtype=torch.float64), K)
    h2 = crf._entropy_discretized(mu, torch.tensor([2.0], dtype=torch.float64), K)
    h_huge = crf._entropy_discretized(mu, torch.tensor([1e4], dtype=torch.float64), K)
    assert float(h2) > float(h1) > 0.0
    assert float(h_huge) <= math.log(15.0) + 1e-6  # saturates at log K


def test_entropy_bonus_raises_sigma():
    def run(w_ent):
        torch.manual_seed(0)
        wrapper, _ = _make_wrapper(
            lambda ranks: 3.0,  # flat reward: only the entropy term moves rho
            names=("a",),
            rf_learn_sigma=True,
            rf_w_entropy=w_ent,
        )
        rho = wrapper.extra_trainable_params()[0]
        opt = torch.optim.Adam([rho], lr=0.5)
        for _ in range(60):
            opt.zero_grad()
            total = wrapper(input_ids=IDS)
            total.backward()
            opt.step()
        return float(wrapper._sigma_tensor().max())

    assert run(1e-2) > run(0.0) + 0.05


def test_early_stopper_snapshots_include_rho():
    wrapper, layers = _make_wrapper(lambda ranks: 3.0, rf_learn_sigma=True)
    params = [layer.gamma for layer in layers] + wrapper.extra_trainable_params()
    stopper = dobi._EarlyStopper(params, patience=0)
    stopper.update(10.0)
    rho = wrapper.extra_trainable_params()[0]
    before = rho.detach().clone()
    with torch.no_grad():
        rho += 5.0
    stopper.restore_best()
    assert torch.allclose(rho.detach(), before)


# --------------------------------------------------------------------------- #
# Composite reward + teacher
# --------------------------------------------------------------------------- #
def test_token_kl_matches_manual_and_zero_for_identical():
    torch.manual_seed(0)
    t = torch.randn(1, 6, 5)
    s = torch.randn(1, 6, 5)
    got = crf._token_kl(t, s, chunk=2)
    tl = t.reshape(-1, 5).float().log_softmax(-1)
    sl = s.reshape(-1, 5).float().log_softmax(-1)
    want = float((tl.exp() * (tl - sl)).sum(-1).mean())
    assert abs(got - want) < 1e-6
    assert abs(crf._token_kl(t, t.clone())) < 1e-7


def test_hidden_mse_skips_embeddings_and_zero_for_identical():
    t = (torch.ones(1, 3, 4), torch.full((1, 3, 4), 2.0))
    s_same = (torch.zeros(1, 3, 4), torch.full((1, 3, 4), 2.0))
    assert crf._hidden_mse(t, s_same) == 0.0  # layer 0 mismatch ignored
    s_off = (torch.ones(1, 3, 4), torch.full((1, 3, 4), 4.0))
    want = float((4.0 - 2.0) ** 2 * 12) / float(2.0**2 * 12)
    assert abs(crf._hidden_mse(t, s_off) - want) < 1e-6


def test_teacher_called_once_per_window_and_absent_when_unused():
    teacher = _FakeTeacher(torch.zeros(1, 4, 16))
    wrapper, _ = _make_wrapper(
        lambda ranks: 3.0 + 0.1 * ranks[0],
        rf_w_kl=1.0,
        rf_samples=4,
        teacher=teacher,
    )
    torch.manual_seed(0)
    wrapper(input_ids=IDS)
    assert teacher.calls == 1  # shared across the 4 samples
    wrapper2, _ = _make_wrapper(lambda ranks: 3.0)
    assert wrapper2._teacher == []  # no teacher when w_kl = w_hid = 0


def test_kl_only_reward_zero_grad_when_student_matches_teacher():
    # student logits identical to the teacher's regardless of ranks -> KL == 0
    # for every sample -> constant reward -> exact zero gradient on mu.
    fixed = torch.zeros(1, 4, 16)
    teacher = _FakeTeacher(fixed.clone())
    wrapper, layers = _make_wrapper(
        lambda ranks: 999.0,  # CE ignored (w_ce = 0)
        rf_w_ce=0.0,
        rf_w_kl=1.0,
        teacher=teacher,
        model_kwargs={"logits_fn": lambda ranks, T, V: fixed.clone()},
    )
    total = wrapper(input_ids=IDS)
    total.backward()
    for layer in layers:
        assert float(layer.gamma.grad) == 0.0


# --------------------------------------------------------------------------- #
# Plumbing: hard-rank discipline, diag/log format, config, world size
# --------------------------------------------------------------------------- #
def test_hard_rank_discipline():
    wrapper, layers = _make_wrapper(lambda ranks: 3.0 + 0.01 * sum(ranks))
    assert all(layer.hard_rank is None for layer in layers)
    wrapper(input_ids=IDS)
    for layer in layers:
        assert layer.hard_rank == int(round(float(layer.gamma)))
    windows = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8])]
    val = wrapper.val_perplexity_at_policy(windows, "cpu")
    assert math.isfinite(val)
    for layer in layers:
        assert layer.hard_rank == int(round(float(layer.gamma)))


def _plot_epoch_regex():
    import re

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "plot_ranktrain.py")
    with open(path) as f:
        src = f.read()
    m = re.search(r'EPOCH_RE = re\.compile\(r"(.+)"\)', src)
    assert m, "EPOCH_RE definition not found in plot_ranktrain.py"
    return re.compile(m.group(1))


def test_epoch_line_parses_and_diag_line_does_not():
    epoch_re = _plot_epoch_regex()
    means = {"ppl": 12.0, "reg": 0.5, "value": 0.0, "ratio": 0.6}
    line = cr._epoch_log_line(3, means, 8.25, True)
    assert epoch_re.search(line)
    wrapper, _ = _make_wrapper(lambda ranks: 3.0)
    wrapper(input_ids=IDS)
    diag = wrapper.epoch_diag()
    for field in ("sigma_rel=", "ce=", "kl=", "hid=", "ent=", "slack="):
        assert field in diag
    assert not epoch_re.search(diag)


def test_config_validation_accepts_and_rejects():
    base = ["compress.coala_rank_selection.trainer=reinforce"]
    for good in (
        base,
        base + ["compress.coala_rank_selection.rf_samples=3"],  # odd N now valid
        base + ["compress.coala_rank_selection.rf_budget=project"],
    ):
        validate_config(
            load_config("configs/coala_ranktrain.yaml", good), cuda_available=True
        )
    for bad in (
        ["compress.coala_rank_selection.trainer=bogus"],
        base + ["compress.coala_rank_selection.rf_samples=1"],
        base + ["compress.coala_rank_selection.rf_budget=hardish"],
        base + ["compress.coala_rank_selection.rf_w_kl=-0.5"],
        base
        + [
            "compress.coala_rank_selection.rf_w_ce=0.0"
        ],  # all-zero reward (kl/hid default 0)
        base
        + [
            "compress.coala_rank_selection.rf_w_entropy=0.001",
            "compress.coala_rank_selection.rf_learn_sigma=false",
        ],
        base + ["compress.coala_rank_selection.rf_sigma_lr_scale=0"],
        base + ["compress.coala_rank_selection.rf_adv_ema_beta=1.5"],
    ):
        with pytest.raises(ValueError):
            validate_config(
                load_config("configs/coala_ranktrain.yaml", bad), cuda_available=True
            )


def test_sigma_schedule_endpoints_scheduled_mode():
    rs = _rf_cfg().compress.coala_rank_selection
    assert crf._sigma_rel_for_epoch(0, 20, rs) == pytest.approx(0.05)
    assert crf._sigma_rel_for_epoch(19, 20, rs) == pytest.approx(0.005)
    wrapper, _ = _make_wrapper(lambda ranks: 3.0)  # scheduled (rf_learn_sigma=False)
    sigma = wrapper._sigma_tensor()
    assert float(sigma.min()) == 1.0  # 0.05*4 = 0.2 -> floored


def test_world_size_forced_single_gpu_for_reinforce():
    rs_rf = OmegaConf.create({"trainer": "reinforce", "num_samples": 256})
    rs_gate = OmegaConf.create({"trainer": "gate", "num_samples": 256})
    assert cr._train_world_size(rs_rf, 4) == 1
    assert cr._train_world_size(rs_gate, 4) == 4
    assert cr._train_world_size(rs_rf, 1) == 1
