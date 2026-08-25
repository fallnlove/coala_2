"""Dobi low_mem rank training: bf16 base + fp32 SVD islands + checkpointing.

Pure CPU. Mirrors the COALA2 low_mem tests: the checkpointed path must produce
identical outputs/gradients to the direct path, dtypes must round-trip, and the
default fp32 path must stay byte-identical to the old behavior.
"""

import pytest
import torch

from pipeline._paths import add_reference_paths

add_reference_paths()

from dobi.stable_svd import SVDTransformLayer  # noqa: E402
from pipeline.config import load_config, validate_config  # noqa: E402


def _layer(dtype, seed=0, in_f=8, out_f=12, gamma=3.0):
    torch.manual_seed(seed)
    w = torch.randn(out_f, in_f, dtype=dtype)
    return SVDTransformLayer(
        gamma=gamma,
        SEQ_LEN=4,
        beta=10.0,
        input_size=in_f,
        output_size=out_f,
        weight_size=torch.tensor(float(in_f * out_f)),
        weight=w,
        bias=None,
        name="t",
        device="cpu",
        svd_impl="stable",
    )


def test_fp32_default_path_unchanged():
    # fp32 weights -> direct path (no checkpoint), fp32 in -> fp32 out, finite
    # gamma grad — the pre-low_mem contract.
    layer = _layer(torch.float32)
    for p in layer.parameters():
        p.requires_grad = False
    layer.gamma.requires_grad = True
    x = torch.randn(1, 4, 8, dtype=torch.float32)
    out = layer(x)
    assert out.dtype == torch.float32 and out.shape == (1, 4, 12)
    (out**2).sum().backward()
    assert torch.isfinite(layer.gamma.grad).all() and float(layer.gamma.grad) != 0.0


def test_bf16_low_mem_output_dtype_and_fp32_grad():
    layer = _layer(torch.bfloat16)
    for p in layer.parameters():
        p.requires_grad = False
    layer.gamma.requires_grad = True
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    out = layer(x)  # grad enabled + bf16 weights -> checkpoint path
    assert out.dtype == torch.bfloat16
    (out.float() ** 2).sum().backward()
    assert layer.gamma.grad.dtype == torch.float32
    assert torch.isfinite(layer.gamma.grad).all() and float(layer.gamma.grad) != 0.0


def test_bf16_checkpointed_grad_matches_direct():
    # svd_lowrank draws a random projection, so RNG state must line up: seed
    # before each forward; checkpoint's use_reentrant=False preserves/restores
    # RNG in its backward recompute, so grads must match the direct path.
    a = _layer(torch.bfloat16, seed=1)
    b = _layer(torch.bfloat16, seed=1)  # identical weights
    for layer in (a, b):
        for p in layer.parameters():
            p.requires_grad = False
        layer.gamma.requires_grad = True
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)

    torch.manual_seed(7)
    out_ckpt = a(x)  # checkpoint path
    (out_ckpt.float() ** 2).sum().backward()

    torch.manual_seed(7)
    out_direct = b._forward_impl(x)  # bypasses checkpointing
    (out_direct.float() ** 2).sum().backward()

    assert torch.equal(out_ckpt, out_direct)
    assert torch.allclose(a.gamma.grad, b.gamma.grad, rtol=1e-6, atol=1e-8)


def test_bf16_no_grad_takes_direct_path():
    layer = _layer(torch.bfloat16)
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        y = layer(x)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()


def test_dobi_low_mem_config_validation():
    base = ["compress.dobi.rank_selection.low_mem=true"]
    for dt in ("bfloat16", "float16"):
        cfg = load_config("configs/dobi.yaml", base + [f"model.load_dtype={dt}"])
        validate_config(cfg, cuda_available=True)
    with pytest.raises(ValueError):
        validate_config(
            load_config("configs/dobi.yaml", base + ["model.load_dtype=float32"]),
            cuda_available=True,
        )
    # default config (low_mem absent/false) still validates with any dtype
    validate_config(load_config("configs/dobi.yaml"), cuda_available=True)
