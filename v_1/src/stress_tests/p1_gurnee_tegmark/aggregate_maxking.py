"""Aggregate the maximal-with-kings probe results into one scoreboard.

Reads p1_gurnee_tegmark/results/maxking/p1_maxking__<model>.json and prints (and
writes results/maxking/RESULTS_maxking.md) a table of, per model x site:
  ruler_clf macro-F1 (+ chance + shuffle),  year_strat Spearman + ±10yr acc,
  year_group Spearman (legacy).
Dependency-light (json + stdlib only) so it runs on a login node.

Usage:
    python v_1/src/stress_tests/p1_gurnee_tegmark/aggregate_maxking.py
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results" / "maxking"
SITES = ["mean", "king_last", "king_mean"]
MODEL_ORDER = ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b",
               "thalesian_akk300m", "thalesian_cunei400m", "umt5_base", "random"]


def _fmt(x, nd=3):
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def _row(model, site, s):
    if s.get("missing"):
        return f"| {model} | {site} | missing | | | | | |"
    if s.get("insufficient"):
        return f"| {model} | {site} | insufficient | | | | | |"
    b = s["best"]; rc = b["ruler_clf"]; ys = b["year_strat"]
    acc10 = ys["per_k"].get(str(ys["best_k"]), {}).get("acc10_mean")
    return (f"| {model} | {site} | L{s['best_layer']} "
            f"| {_fmt(rc['macro_f1_mean'])} "
            f"| {_fmt(rc['chance_macro_f1'],2)} / {_fmt(rc['shuffled_macro_f1'],2)} "
            f"| {_fmt(ys['spearman_mean'])} "
            f"| {_fmt(acc10,2)} "
            f"| {_fmt(b['year_group']['spearman_mean'])} |")


def main():
    files = {p.stem.replace("p1_maxking__", ""): p for p in RES.glob("p1_maxking__*.json")}
    if not files:
        print(f"no results in {RES}"); return
    order = [m for m in MODEL_ORDER if m in files] + [m for m in files if m not in MODEL_ORDER]

    lines = ["# maximal-with-kings scoreboard (balanced-MC, 5 rulers x k=9, king-found)\n",
             "All three sites on ONE cleaning (clean_maximal_keepking). `ruler_clf` is the "
             "control: if `random` matches trained models, the site reads name-token identity, "
             "not learned structure. `year_strat` = StratifiedKFold (in-distribution); "
             "`year_group` = legacy GroupKFold-by-ruler (degenerate for a per-king-constant "
             "label — near 0 by construction).\n",
             "| model | site | best L | ruler macro-F1 | chance / shuffle | year_strat Sp "
             "| year ±10yr acc | year_group Sp |",
             "|---|---|---|---|---|---|---|---|"]
    for m in order:
        d = json.loads(files[m].read_text())
        for site in SITES:
            lines.append(_row(m, site, d["sites"].get(site, {"missing": True})))
        lines.append("| | | | | | | | |")

    md = "\n".join(lines)
    (RES / "RESULTS_maxking.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwrote {RES / 'RESULTS_maxking.md'}")


if __name__ == "__main__":
    main()
