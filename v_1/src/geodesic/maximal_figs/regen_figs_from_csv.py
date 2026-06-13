#!/usr/bin/env python3
"""Regenerate maximal figures from the committed CSVs (no cluster probe JSONs needed).

Reads:
  tables/T1_year_pls_maximal.csv
  tables/T2_year_ridge_maximal.csv
  tables/T3_ruler_maximal.csv
  tables/ksweep_tradeoff_maximal.csv

Writes into figures/:
  fig1_maximal_ACD.png
  fig2_maximal_AB.png
  fig4_maximal_A.png
  ksweep_tradeoff_maximal.png
  ksweep_per_method_maximal.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEO = HERE.parent
sys.path.insert(0, str(GEO))
import plot_round3_story_figures as prs  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import make_maximal_figs as mmf  # noqa: E402

MODELS = ["tfidf", "mlm", "thalesian_akk300m", "thalesian_cunei400m",
          "qwen3_1b7", "qwen3_8b", "qwen3_32b", "random"]


def _float(s: str) -> float | None:
    try:
        return float(s) if s not in ("", "nan") else None
    except (ValueError, TypeError):
        return None


def load_data(tables: Path) -> dict:
    data: dict = {"pls": {}, "ridge": {}, "ruler": {}}
    for r in csv.DictReader(open(tables / "T1_year_pls_maximal.csv")):
        m, L = r["model"], int(r["layer"])
        data["pls"].setdefault(m, {})[L] = (_float(r["spearman_mean"]), _float(r["spearman_std"]))
    for r in csv.DictReader(open(tables / "T2_year_ridge_maximal.csv")):
        m, L = r["model"], int(r["layer"])
        data["ridge"].setdefault(m, {})[L] = (_float(r["spearman_mean"]), _float(r["spearman_std"]))
    for r in csv.DictReader(open(tables / "T3_ruler_maximal.csv")):
        data["ruler"][r["model"]] = (_float(r["macro_f1"]), _float(r["macro_f1_std"]),
                                     r.get("source", ""))
    return data


def regen_ksweep(tables: Path, fig_out: Path) -> None:
    ksweep_csv = tables / "ksweep_tradeoff_maximal.csv"
    if not ksweep_csv.exists():
        print(f"[skip] {ksweep_csv} not found")
        return

    curves: dict[str, dict] = {}
    for r in csv.DictReader(open(ksweep_csv)):
        m, k = r["model"], int(r["k"])
        curves.setdefault(m, {})[k] = (
            float(r["spearman_mean"]),
            float(r["spearman_std"]),
            _float(r.get("ridge_baseline", "")),
        )

    present = [m for m in MODELS if m in curves]

    # tradeoff: all models on one axis
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in present:
        dm = curves[m]
        ks = sorted(dm)
        means = [dm[k][0] for k in ks]
        rb = dm[ks[0]][2]
        c = prs.MODEL_COLOR.get(m, "#333")
        ax.plot(ks, means, "-o", color=c, label=prs.MODEL_SHORT.get(m, m))
        if rb is not None:
            ax.axhline(rb, ls="--", color=c, lw=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("PLS components (k) [log2]")
    ax.set_ylabel("Year Spearman (balanced, mean over draws)")
    ax.set_title("PLS components tradeoff — MAXIMAL/mean — dashed = Ridge")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_out / "ksweep_tradeoff_maximal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {fig_out}/ksweep_tradeoff_maximal.png")

    # per-method panels
    if not present:
        return
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.3), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, present):
        dm = curves[m]
        ks = sorted(dm)
        means = [dm[k][0] for k in ks]
        rb = dm[ks[0]][2]
        c = prs.MODEL_COLOR.get(m, "#333")
        ax.plot(ks, means, "-o", color=c)
        if rb is not None:
            ax.axhline(rb, ls="--", color="firebrick", lw=1.2, label=f"Ridge {rb:.3f}")
        bi = int(np.argmax(means))
        ax.scatter([ks[bi]], [means[bi]], marker="*", s=130, color=c,
                   edgecolor="black", zorder=4)
        ax.set_title(f"{prs.MODEL_SHORT.get(m, m)}\npeak k={ks[bi]} ({means[bi]:.3f})", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k [log2]")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("Year Spearman (balanced, maximal)")
    fig.suptitle("Per-method PLS k-sweep — MAXIMAL / mean / balanced", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_out / "ksweep_per_method_maximal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {fig_out}/ksweep_per_method_maximal.png")


def main() -> None:
    tables = HERE / "tables"
    fig_out = HERE / "figures"
    fig_out.mkdir(parents=True, exist_ok=True)

    data = load_data(tables)
    mmf.fig1_ACD(data, fig_out)
    mmf.fig2_AB(data, fig_out)
    mmf.fig4_A(data, fig_out)
    regen_ksweep(tables, fig_out)
    print("[done] all figures regenerated from CSVs")


if __name__ == "__main__":
    main()
