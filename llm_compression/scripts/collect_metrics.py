#!/usr/bin/env python3
"""Collect pipeline eval metrics from results JSONs into one comparison table.

Each ``*_results.json`` written by ``pipeline.run`` holds ``results.perplexity``
(per dataset) and ``results.reasoning`` (lm-eval per-task dicts). This prints a
table of perplexity (lower=better) and reasoning accuracy (plain ``acc,none`` %,
higher=better) — matching the paper table's convention — plus the across-task mean.

Usage:
  python scripts/collect_metrics.py [--latex] [PATH ...]

PATH may be a glob or a directory (expanded to ``<dir>/*_results.json``).
Defaults to ``outputs/coala_compress``. ``--latex`` emits a paper-ready LaTeX
tabular (paper column order, best per column in bold) instead of the text table.
"""

from __future__ import annotations

import glob
import json
import os
import sys

PPL = ["wikitext2", "ptb", "c4"]
REASON = [
    "piqa",
    "hellaswag",
    "boolq",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
]
ABBR = {
    "piqa": "piqa",
    "hellaswag": "hella",
    "boolq": "boolq",
    "winogrande": "wino",
    "arc_easy": "arcE",
    "arc_challenge": "arcC",
    "openbookqa": "obqa",
}


def _acc(task_metrics: dict):
    """Plain accuracy ``acc,none`` for every task (NOT ``acc_norm``), to match the
    paper table's convention. All 7 reasoning tasks report ``acc,none``."""
    return task_metrics.get("acc,none")


def _label(path: str) -> str:
    b = os.path.basename(path)
    return b[: -len("_results.json")] if b.endswith("_results.json") else b


def _expand(patterns):
    files = []
    for p in patterns:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "*_results.json"))
        else:
            files += glob.glob(p)
    return sorted(set(files))


def load_row(path: str) -> dict:
    d = json.load(open(path))
    res = d.get("results", {}) or {}
    ppl = res.get("perplexity", {}) or {}
    rea = res.get("reasoning", {}) or {}
    compress = d.get("compress", {}) or {}
    params = compress.get("params", {}) or {}
    # baseline runs (compress.enabled=false) are the uncompressed reference —
    # they have no strategy and must not fall through to the "coala" default.
    if not compress.get("enabled", True):
        strategy = "none"
    else:
        strategy = str(compress.get("strategy", "coala"))
    if strategy == "dobi":
        # Dobi has no compress.rank_source (that's a COALA-only key) — it always
        # rank-selects (trains or loads a checkpoint), never "uniform".
        rank_source = (
            "checkpoint"
            if compress.get("dobi", {}).get("rank_selection", {}).get("resume")
            else "train"
        )
    else:
        rank_source = str(compress.get("rank_source", "uniform"))
    accs = {t: _acc(rea.get(t, {}) or {}) for t in REASON}
    have = [v for v in accs.values() if v is not None]
    return {
        "label": _label(path),
        "strategy": strategy,
        "ratio": compress.get("ratio"),
        "rank_source": rank_source,
        "mu": params.get("mu", "-"),
        "adaptive": params.get("adaptive", "-"),
        "ppl": ppl,
        "accs": accs,
        "mean": (sum(have) / len(have)) if have else None,
    }


def _num(x, pct=False):
    if x is None:
        return "-"
    return f"{x * 100:.1f}" if pct else f"{x:.2f}"


# Paper-table column order (perplexity then the 7 reasoning tasks, then Average).
_LATEX_PPL = [("wikitext2", "WikiText-2"), ("c4", "C4"), ("ptb", "PTB")]
_LATEX_REA = [
    ("openbookqa", "Openb."),
    ("arc_easy", r"ARC\_e"),
    ("arc_challenge", r"ARC\_c"),
    ("winogrande", "WinoG."),
    ("hellaswag", "HellaS."),
    ("piqa", "PIQA"),
    ("boolq", "BoolQ"),
]


def _mu_tex(mu):
    """Render the regularization mu as a clean power of 10 where possible."""
    pows = {0.01: "10^{-2}", 0.1: "10^{-1}", 1.0: "1", 10.0: "10", 100.0: "100"}
    try:
        return pows.get(float(mu), str(mu))
    except (TypeError, ValueError):
        return str(mu)


_STRATEGY_LABEL = {
    "none": "Uncompressed",
    "asvd": "ASVD",
    "svd_llm": "SVD-LLM",
    "svd_llm_2": "SVD-LLM++",
    "dobi": "Dobi-SVD",
    "swift_svd": "Swift-SVD",
    "empty": "Empty",
    "svd": "SVD",
}


def _method_label(row):
    strategy = row.get("strategy", "coala")
    if strategy == "swift_svd":
        # Dynamic (Algorithm 2) allocation is the paper's starred variant; uniform
        # ranks are plain Swift-SVD.
        star = "*" if row.get("rank_source") in ("swift", "checkpoint") else ""
        return f"Swift-SVD{star}"
    if strategy != "coala":
        return _STRATEGY_LABEL.get(strategy, strategy)
    mu, adaptive = row["mu"], row["adaptive"]
    try:
        positive = float(mu) > 0
    except (TypeError, ValueError):
        positive = False
    if adaptive is True and positive:
        base = r"COALA-A ($\mu{=}%s$)" % _mu_tex(mu)
    else:
        base = r"COALA ($\mu{=}0$)"
    # Learned/loaded ranks = the COALA2 method (rank training on top of the
    # COALA operator) — name it so, vs the uniform-rank COALA baselines.
    if row.get("rank_source") in ("train", "checkpoint"):
        base = base.replace("COALA-A", "COALA2").replace("COALA ", "COALA2 ")
    return base


def to_latex(rows):
    """Emit a paper-ready LaTeX tabular; bold the best in each column."""
    best_ppl = {
        ds: min(
            (r["ppl"].get(ds) for r in rows if r["ppl"].get(ds) is not None),
            default=None,
        )
        for ds, _ in _LATEX_PPL
    }
    best_rea = {
        t: max(
            (r["accs"].get(t) for r in rows if r["accs"].get(t) is not None),
            default=None,
        )
        for t, _ in _LATEX_REA
    }
    best_avg = max((r["mean"] for r in rows if r["mean"] is not None), default=None)

    def cell(x, best, pct):
        if x is None:
            return "-"
        s = f"{x * 100:.2f}" if pct else f"{x:.2f}"
        return r"\textbf{%s}" % s if best is not None and abs(x - best) < 1e-9 else s

    ratios = {r.get("ratio") for r in rows if r.get("ratio") is not None}
    ratio_str = f"{next(iter(ratios)):g}" if len(ratios) == 1 else "mixed"
    sources = {r.get("rank_source", "uniform") for r in rows}
    if sources <= {"train", "checkpoint"}:
        rank_str = "learned"
    elif sources == {"uniform"}:
        rank_str = "uniform"
    else:
        rank_str = "uniform/learned"
    print(f"% {rank_str} ranks, ratio {ratio_str}; best in bold")
    print(r"\begin{tabular}{l|ccc|ccccccc|c}")
    print(r"\toprule")
    print(
        "Method & "
        + " & ".join(h + r"$\downarrow$" for _, h in _LATEX_PPL)
        + " & "
        + " & ".join(h for _, h in _LATEX_REA)
        + r" & \textbf{Average}$\uparrow$ \\"
    )
    print(r"\midrule")
    for r in rows:
        cells = [_method_label(r)]
        cells += [cell(r["ppl"].get(ds), best_ppl[ds], False) for ds, _ in _LATEX_PPL]
        cells += [cell(r["accs"].get(t), best_rea[t], True) for t, _ in _LATEX_REA]
        cells += [cell(r["mean"], best_avg, True)]
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main(argv):
    latex = "--latex" in argv
    argv = [a for a in argv if a != "--latex"]
    rows = [load_row(f) for f in _expand(argv or ["outputs/coala_compress"])]
    rows = [r for r in rows if r["mean"] is not None or any(r["ppl"].values())]
    if not rows:
        print("No results with metrics found.", file=sys.stderr)
        return 1
    if latex:
        _strategy_rank = {
            "none": -1,
            "asvd": 0,
            "svd_llm": 1,
            "svd_llm_2": 2,
            "dobi": 3,
            "swift_svd": 3.5,
        }

        def _sort_key(r):
            is_coala = r.get("strategy", "coala") == "coala"
            try:
                mu = float(r["mu"]) if is_coala else 0.0
            except (TypeError, ValueError):
                mu = 0.0
            return (
                _strategy_rank.get(r.get("strategy", "coala"), 4),
                bool(r["adaptive"]) if is_coala else False,
                mu,
            )

        rows.sort(key=_sort_key)
        to_latex(rows)
        return 0
    rows.sort(key=lambda r: (r["mean"] is None, -(r["mean"] or 0)))

    head = (
        f"{'config':14}{'mu':>5}{'adpt':>6}"
        + "".join(f"{p[:5]:>8}" for p in PPL)
        + "".join(f"{ABBR[t]:>7}" for t in REASON)
        + f"{'avg':>7}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        line = (
            f"{r['label']:14}{str(r['mu']):>5}{str(r['adaptive']):>6}"
            + "".join(f"{_num(r['ppl'].get(p)):>8}" for p in PPL)
            + "".join(f"{_num(r['accs'][t], pct=True):>7}" for t in REASON)
            + f"{_num(r['mean'], pct=True):>7}"
        )
        print(line)
    print(
        "\nperplexity (wiki2/ptb/c4): lower=better | reasoning: plain acc%, "
        "higher=better | avg = mean over tasks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
