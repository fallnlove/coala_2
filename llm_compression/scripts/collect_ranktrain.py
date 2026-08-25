#!/usr/bin/env python3
"""Collect COALA2 (gate) rank-training logs and report the best hyperparams.

Parses the per-epoch lines written by pipeline/coala_rank.py's rank-training
loop (``epoch N: train_ppl=... val_ppl=... ratio=...``) plus the pre-epoch-1
uniform-rank reference (``uniform-rank reference (val): val_ppl=...``), finds
each config's best (lowest) validation perplexity and the epoch it occurred
at, and ranks configs by improvement over their own uniform reference.

Usage:
  python scripts/collect_ranktrain.py [LOG_OR_DIR ...] [--latex]

Each positional arg is a ``.log`` file or a directory (expanded to
``*.log``); configs are grouped by ratio (parsed from the log's config dump)
so a multi-ratio sweep collected in one call still reports per-ratio winners.
Defaults to ``outputs/coala_ranktrain_r06 outputs/coala_ranktrain_r08``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# pipeline/dobi.py:_epoch_log_line (shared by coala_rank.py):
#   "  epoch {n}: train_ppl={p:.3f} val_ppl={v}[ [best]] reg={r:.3f} value={x:.3f} ratio={a:.3f}"
# val_ppl is "n/a" when the eval-protocol perplexity was non-finite that epoch.
REF_RE = re.compile(r"(?:init-ranks|uniform-rank) reference \(val\): val_ppl=([0-9.]+)")
EPOCH_RE = re.compile(
    r"epoch (\d+): train_ppl=([0-9.]+) val_ppl=([0-9.]+|n/a)(?: \[best\])? "
    r"reg=([0-9.]+) value=([0-9.]+) ratio=([0-9.]+)"
)
LR_RE = re.compile(r"^\s*lr:\s*([0-9.]+)\s*$")
LREG_RE = re.compile(r"^\s*lambda_reg:\s*([0-9.]+)\s*$")
RATIO_RE = re.compile(r"^\s*ratio:\s*([0-9.]+)\s*$")


def parse_log(path: str) -> dict:
    ref = None
    epochs = []
    lr = lreg = ratio = None
    with open(path, errors="ignore") as f:
        for line in f:
            if lr is None:
                m = LR_RE.match(line)
                if m:
                    lr = float(m.group(1))
            if lreg is None:
                m = LREG_RE.match(line)
                if m:
                    lreg = float(m.group(1))
            if ratio is None:
                m = RATIO_RE.match(line)
                if m:
                    ratio = float(m.group(1))
            m = REF_RE.search(line)
            if m:
                ref = float(m.group(1))
            m = EPOCH_RE.search(line)
            if m:
                val_str = m.group(3)
                epochs.append(
                    {
                        "epoch": int(m.group(1)),
                        "train_ppl": float(m.group(2)),
                        "val_ppl": float(val_str) if val_str != "n/a" else None,
                        "ratio_achieved": float(m.group(6)),
                    }
                )
    return {
        "label": os.path.splitext(os.path.basename(path))[0],
        "path": path,
        "ref": ref,
        "lr": lr,
        "lambda_reg": lreg,
        "ratio": ratio,
        "epochs": epochs,
    }


def expand(patterns):
    files = []
    for p in patterns:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.log")))
        else:
            files += sorted(glob.glob(p))
    return list(dict.fromkeys(files))


def summarize(run: dict):
    finite = [e for e in run["epochs"] if e["val_ppl"] is not None]
    if not finite:
        return None
    best = min(finite, key=lambda e: e["val_ppl"])
    last = run["epochs"][-1]
    improvement = None
    if run["ref"] is not None:
        improvement = (run["ref"] - best["val_ppl"]) / run["ref"] * 100
    out = dict(run)
    out["best"] = best
    out["last_epoch"] = last
    out["improvement_pct"] = improvement
    return out


def _sort_key(r):
    # runs without a reference sort last; otherwise biggest improvement first
    return (r["improvement_pct"] is None, -(r["improvement_pct"] or -1e9))


def _fmt(x, spec):
    return format(x, spec) if x is not None else "-"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        default=["outputs/coala_ranktrain_r06", "outputs/coala_ranktrain_r08"],
    )
    ap.add_argument("--latex", action="store_true")
    ns = ap.parse_args(argv)

    runs = [summarize(parse_log(p)) for p in expand(ns.paths)]
    runs = [r for r in runs if r is not None]
    if not runs:
        print("No epoch lines found in the given logs.", file=sys.stderr)
        return 1

    by_ratio = {}
    for r in runs:
        by_ratio.setdefault(r["ratio"], []).append(r)

    for ratio in sorted(by_ratio, key=lambda x: (x is None, x)):
        group = sorted(by_ratio[ratio], key=_sort_key)
        ratio_str = f"{ratio:g}" if ratio is not None else "unknown"
        print(f"\n=== ratio={ratio_str} ===")
        head = (
            f"{'config':12}{'lr':>6}{'lreg':>7}{'ref_val':>10}{'best_val':>10}"
            f"{'@ep':>5}{'ratio@best':>11}{'improve%':>10}{'last_val':>10}"
        )
        print(head)
        print("-" * len(head))
        for r in group:
            row = (
                f"{r['label']:12}"
                f"{_fmt(r['lr'], '6g')}"
                f"{_fmt(r['lambda_reg'], '7g')}"
                f"{_fmt(r['ref'], '10.3f')}"
                f"{r['best']['val_ppl']:10.3f}"
                f"{r['best']['epoch']:5d}"
                f"{r['best']['ratio_achieved']:11.4f}"
                f"{_fmt(r['improvement_pct'], '10.2f')}"
                f"{_fmt(r['last_epoch']['val_ppl'], '10.3f')}"
            )
            print(row)
        winner = group[0]
        if winner["ref"] is not None:
            sign = "+" if winner["improvement_pct"] >= 0 else ""
            print(
                f"\nBest params @ ratio={ratio_str}: {winner['label']} "
                f"(lr={winner['lr']}, lambda_reg={winner['lambda_reg']}) -> "
                f"val_ppl={winner['best']['val_ppl']:.3f} at epoch {winner['best']['epoch']} "
                f"(uniform ref={winner['ref']:.3f}, {sign}{winner['improvement_pct']:.2f}% improvement)"
            )
        else:
            print(
                f"\nBest params @ ratio={ratio_str}: {winner['label']} -> "
                f"val_ppl={winner['best']['val_ppl']:.3f} at epoch {winner['best']['epoch']} "
                f"(no uniform reference found in log)"
            )

    if ns.latex:
        print("\n% --- config & lr & lambda_reg & ref & best_val & improve% ---")
        for ratio in sorted(by_ratio, key=lambda x: (x is None, x)):
            group = sorted(by_ratio[ratio], key=_sort_key)
            for r in group:
                imp = _fmt(r["improvement_pct"], ".2f")
                ref = _fmt(r["ref"], ".3f")
                print(
                    f"{r['label']} & {r['lr']} & {r['lambda_reg']} & {ref} & "
                    f"{r['best']['val_ppl']:.3f} & {imp} \\\\"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
