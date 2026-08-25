#!/usr/bin/env python3
"""Plot rank-training train/val perplexity curves from run logs (matplotlib).

Parses the per-epoch lines written by the dobi/coala rank-selection loops
(``epoch N: train_ppl=... val_ppl=... ratio=...``) plus the pre-epoch-1
reference (``init-ranks reference (val)`` or ``uniform-rank reference (val)``),
and renders two panels: validation perplexity vs epoch (epoch 0 = the
reference) and train perplexity vs epoch (log scale — Dobi's train ppl can
oscillate by an order of magnitude).

Usage:
  python scripts/plot_ranktrain.py [LOG_OR_DIR ...] [--out PREFIX]

Each positional arg is a ``.log`` file or a directory (expanded to ``*.log``);
series are labelled by file stem. Writes ``PREFIX.png`` and ``PREFIX.pdf``
(default ``ranktrain_curves``).
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_RE = re.compile(r"(?:init-ranks|uniform-rank) reference \(val\): val_ppl=([0-9.]+)")
EPOCH_RE = re.compile(r"epoch (\d+): train_ppl=([0-9.]+) val_ppl=([0-9.]+)")

COLORS = [
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#e87ba4",
    "#eb6834",
]
DASHES = ["-", ":", "--", (0, (6, 2, 1, 2)), (0, (2, 2)), "-", "--", ":"]


def parse_log(path):
    ref, epochs, train, val = None, [], [], []
    with open(path, errors="ignore") as f:
        for line in f:
            m = REF_RE.search(line)
            if m:
                ref = float(m.group(1))
            e = EPOCH_RE.search(line)
            if e:
                epochs.append(int(e.group(1)))
                train.append(float(e.group(2)))
                val.append(float(e.group(3)))
    return {"ref": ref, "epochs": epochs, "train": train, "val": val}


def expand(args):
    files = []
    for a in args:
        if os.path.isdir(a):
            files += sorted(glob.glob(os.path.join(a, "*.log")))
        else:
            files += sorted(glob.glob(a))
    return list(dict.fromkeys(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["outputs/dobi_ranktrain"])
    ap.add_argument("--out", default="ranktrain_curves")
    ns = ap.parse_args()

    runs = []
    for p in expand(ns.paths or ["outputs/dobi_ranktrain"]):
        d = parse_log(p)
        if d["epochs"]:
            runs.append((os.path.splitext(os.path.basename(p))[0], d))
    if not runs:
        raise SystemExit("no epoch lines found in the given logs")

    fig, (ax_val, ax_tr) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for i, (name, d) in enumerate(runs):
        c, ls = COLORS[i % len(COLORS)], DASHES[i % len(DASHES)]
        # validation: epoch 0 = the init/uniform reference (open marker)
        if d["ref"] is not None:
            ax_val.plot(
                [0] + d["epochs"],
                [d["ref"]] + d["val"],
                ls=ls,
                color=c,
                marker="o",
                ms=3.5,
                lw=1.8,
                label=name,
            )
            ax_val.plot([0], [d["ref"]], marker="o", ms=7, mfc="none", mec=c, lw=0)
        else:
            ax_val.plot(
                d["epochs"],
                d["val"],
                ls=ls,
                color=c,
                marker="o",
                ms=3.5,
                lw=1.8,
                label=name,
            )
        ax_tr.plot(d["epochs"], d["train"], ls=ls, color=c, marker="o", ms=3.5, lw=1.8)

    ax_val.set_title("validation perplexity (held-out, eval protocol)", fontsize=11)
    ax_val.set_xlabel("epoch  (0 = init-ranks reference)")
    ax_val.set_ylabel("perplexity")
    ax_val.legend(fontsize=8.5, frameon=False)
    ax_tr.set_title("train perplexity (epoch mean)", fontsize=11)
    ax_tr.set_xlabel("epoch")
    ax_tr.set_yscale("log")
    ax_tr.set_ylabel("perplexity (log scale)")
    ax_tr.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for ax in (ax_val, ax_tr):
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{ns.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {ns.out}.png and {ns.out}.pdf ({len(runs)} runs)")


if __name__ == "__main__":
    main()
