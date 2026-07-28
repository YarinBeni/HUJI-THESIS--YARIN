"""Deck figure: continued next-token pretraining on Akkadian, before against after.

Only the four families we actually fine-tuned appear, each as base against every
unfreezing depth we tried, under the honest (cleaned, length-controlled) regime.

    python plot_finetune_fig.py     # -> results/figs/fig_finetune_ntp.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCORES = os.path.join(ROOT, "v_1", "src", "finetune", "results", "scoreboard_best.csv")
FIGS = os.path.join(HERE, "results", "figs")

sys.path.insert(0, HERE)
from plot_cellA_figs import COLORS   # noqa: E402

FAMILIES = [("qwen3_1b7", "Qwen3-1.7B"), ("qwen3_8b", "Qwen3-8B"),
            ("qwen3_32b", "Qwen3-32B"), ("gpt_oss_120b", "gpt-oss-120B")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
})


def main():
    df = pd.read_csv(SCORES)
    df = df[df.cleaning == "maximal"]
    fig, ax = plt.subplots(figsize=(15, 7), layout="constrained")

    xt, xl, x = [], [], 0.0
    for fam, label in FAMILIES:
        g = df[df.family == fam]
        base = g[g.arm == "base"]
        fts = g[g.arm != "base"].sort_values("arm")
        vals = [("base", float(base.spearman_mean.iloc[0]),
                 float(base.spearman_std.iloc[0]))]
        vals += [(r.arm, float(r.spearman_mean), float(r.spearman_std))
                 for r in fts.itertuples()]
        xs = x + np.arange(len(vals)) * 0.85
        for xi, (arm, m, sd) in zip(xs, vals):
            is_base = arm == "base"
            ax.bar(xi, m, width=0.72,
                   color=COLORS[fam] if is_base else "white",
                   edgecolor=COLORS[fam], lw=2.2,
                   hatch="" if is_base else "///", zorder=3)
            ax.errorbar(xi, m, yerr=sd, fmt="none", ecolor="#333",
                        elinewidth=1.2, capsize=4, zorder=4)
            ax.text(xi, m + sd + 0.012, f"{m:.3f}", ha="center", fontsize=11.5,
                    color="#333", zorder=5)
        # a line at the base level makes "no movement" visible at a glance
        ax.hlines(vals[0][1], xs[0] - 0.45, xs[-1] + 0.45, color=COLORS[fam],
                  ls=":", lw=2, zorder=2)
        xt.append(xs.mean())
        xl.append(f"{label}\n(base + {len(vals)-1} unfreezing depths)")
        x = xs[-1] + 1.9

    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_ylabel("year Spearman $\\rho$ (200 balanced draws)")
    ax.set_ylim(0, 0.52)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.set_title("Continued next-token pretraining on our Akkadian corpus: "
                 "solid = base model, hatched = after fine-tuning",
                 pad=12, fontweight="bold")
    out = os.path.join(FIGS, "fig_finetune_ntp.png")
    fig.savefig(out, facecolor="white")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
