#!/usr/bin/env python3
"""Whole-model compression percentages from rank checkpoints and/or run logs.

Sources (auto-detected per argument):

* ``*.log``          — parse COALA's printed ``Compress Ratio (%):`` line
                       (real param count from ``vendor/coala/Injector.py``);
                       used for the uniform baselines (no ranks JSON exists).
* ``*_ranks.json``   — a rank checkpoint ``{model, ratio, seq_len, ranks}``:
    - COALA2 (integer ranks): per-layer cost ``r * (in+out)`` — exactly what
      ``COALA_Linear`` materialises (uncapped: a rank above break-even really
      does store more params than the original weight);
    - Dobi (float gammas): per-layer cost
      ``min(gamma * (min(in,out)/seq_len) * (in+out), in*out)`` — mirrors
      ``pipeline/dobi.py:_compression_loss`` (the ratio Dobi itself optimises;
      its IPCA reconstruction is dense, so this is the nominal rank cost).

Layer dims come from a built-in architecture table keyed by the checkpoint's
``model`` field. Reported per source: whole-model params-after %, target-only
ratio, and absolute param counts.

Usage:
  python scripts/collect_compression.py PATH [PATH ...] [--latex]
"""

from __future__ import annotations

import json
import os
import re
import sys

# suffix -> (in_features, out_features); other = embed + lm_head + norms
ARCH = {
    "mistralai/Mistral-7B-v0.1": {
        "modules": {
            "self_attn.q_proj": (4096, 4096),
            "self_attn.k_proj": (4096, 1024),
            "self_attn.v_proj": (4096, 1024),
            "self_attn.o_proj": (4096, 4096),
            "mlp.gate_proj": (4096, 14336),
            "mlp.up_proj": (4096, 14336),
            "mlp.down_proj": (14336, 4096),
        },
        "other_params": 2 * 32000 * 4096 + (2 * 32 + 1) * 4096,
    },
    "Qwen/Qwen3-0.6B": {
        "modules": {
            "self_attn.q_proj": (1024, 2048),
            "self_attn.k_proj": (1024, 1024),
            "self_attn.v_proj": (1024, 1024),
            "self_attn.o_proj": (2048, 1024),
            "mlp.gate_proj": (1024, 3072),
            "mlp.up_proj": (1024, 3072),
            "mlp.down_proj": (3072, 1024),
        },
        # tied embeddings: one 151936x1024 table + norms (28 layers x 2 + final)
        "other_params": 151936 * 1024 + (2 * 28 + 1) * 1024,
    },
    "Qwen/Qwen3-32B": {
        "modules": {
            "self_attn.q_proj": (5120, 8192),
            "self_attn.k_proj": (5120, 1024),
            "self_attn.v_proj": (5120, 1024),
            "self_attn.o_proj": (8192, 5120),
            "mlp.gate_proj": (5120, 25600),
            "mlp.up_proj": (5120, 25600),
            "mlp.down_proj": (25600, 5120),
        },
        "other_params": 2 * 151936 * 5120 + (2 * 64 + 1) * 5120,
    },
}

LOG_RE = re.compile(r"Compress Ratio \(%\): ([0-9.]+)")


def _dims(arch, layer_name):
    for suffix, dims in arch["modules"].items():
        if layer_name.endswith(suffix):
            return dims
    raise KeyError(f"no dims for layer {layer_name!r} in the arch table")


def from_checkpoint(path):
    payload = json.load(open(path))
    model_id = str(payload.get("model"))
    arch = ARCH.get(model_id)
    if arch is None:
        raise SystemExit(f"{path}: model {model_id!r} not in the ARCH table")
    ranks = payload["ranks"]
    seq_len = int(payload.get("seq_len", 2048))
    # integer ranks -> a rank-per-layer method (COALA2, Swift-SVD); float gammas
    # -> Dobi, whose gamma is a truncation position per seq_len block.
    is_dobi = any(float(v) != int(v) for v in ranks.values())

    target_before = target_after = 0.0
    for name, val in ranks.items():
        in_f, out_f = _dims(arch, name)
        original = in_f * out_f
        if is_dobi:
            eff_rank = float(val) * (min(in_f, out_f) / seq_len)
            cost = min(eff_rank * (in_f + out_f), original)  # dobi caps at orig
        else:
            cost = int(val) * (in_f + out_f)  # COALA adapters store this, uncapped
        target_before += original
        target_after += cost

    other = arch["other_params"]
    total_before = target_before + other
    total_after = target_after + other
    return {
        "label": os.path.splitext(os.path.basename(path))[0],
        "kind": "dobi(gamma)" if is_dobi else "int(rank)",
        "model": model_id,
        "target_ratio_cfg": payload.get("ratio"),
        "target_ratio_achieved": target_after / target_before,
        "whole_pct": total_after / total_before * 100,
        "before_b": total_before / 1e9,
        "after_b": total_after / 1e9,
    }


def from_log(path):
    pct = None
    with open(path, errors="ignore") as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                pct = float(m.group(1))  # keep the LAST occurrence
    if pct is None:
        return None
    return {
        "label": os.path.splitext(os.path.basename(path))[0],
        "kind": "log(injector)",
        "model": "-",
        "target_ratio_cfg": None,
        "target_ratio_achieved": None,
        "whole_pct": pct,
        "before_b": None,
        "after_b": None,
    }


def main(argv):
    latex = "--latex" in argv
    paths = [a for a in argv if a != "--latex"]
    if not paths:
        print(__doc__)
        return 1
    rows = []
    for p in paths:
        row = from_checkpoint(p) if p.endswith(".json") else from_log(p)
        if row is None:
            print(f"WARNING: no 'Compress Ratio (%)' line in {p}", file=sys.stderr)
            continue
        rows.append(row)

    head = (
        f"{'source':28}{'kind':>14}{'cfg_ratio':>10}{'target_ratio':>13}"
        f"{'whole_model%':>13}{'params_after':>13}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        cfg = f"{r['target_ratio_cfg']:.2f}" if r["target_ratio_cfg"] else "-"
        tgt = (
            f"{r['target_ratio_achieved']:.4f}"
            if r["target_ratio_achieved"] is not None
            else "-"
        )
        after = f"{r['after_b']:.2f}B" if r["after_b"] else "-"
        print(
            f"{r['label']:28}{r['kind']:>14}{cfg:>10}{tgt:>13}"
            f"{r['whole_pct']:>12.2f}%{after:>13}"
        )

    if latex:
        print("\n% source & whole-model params (%) & target-modules ratio")
        for r in rows:
            tgt = (
                f"{r['target_ratio_achieved']:.3f}"
                if r["target_ratio_achieved"] is not None
                else "-"
            )
            print(f"{r['label']} & {r['whole_pct']:.2f} & {tgt} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
