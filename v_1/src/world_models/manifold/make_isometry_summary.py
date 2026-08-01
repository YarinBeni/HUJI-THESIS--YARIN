#!/usr/bin/env python3
"""Summarise the 332 full-activation Modell isometry runs into one figure.

manifold_figs.py rendered an arc + isometry panel for every (surface, arm, dataset,
pooling) cell and wrote the two diagnostics alongside each:

  xi   Chatterjee correlation between feature distance and cosine similarity
       -- "does representational similarity fall off monotonically with the feature?"
  rho  Pearson correlation between feature distance and kNN-graph geodesic distance
       -- "is the manifold metric an isometry of the feature metric?"

Reading 332 individual panels is not analysis. What matters is whether either
diagnostic separates a TRAINED arm from its own RANDOM-INIT twin, per cell of the
salience x resource matrix -- because a diagnostic that scores random weights just as
high is measuring the geometry of the activation cloud, not a world model.

The answer differs between the two, which is the point of the figure:
  * rho separates (world_place .290 vs .101; ruler names high, twin absent)
  * xi does not, at fragment level (Llama-70B .556 vs its twin .463, and an untrained
    Qwen on the English gloss reaches .531)

    python make_isometry_summary.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(_HERE, "figs")
OUT = os.path.join(os.path.dirname(_HERE), "figures", "19_isometry_summary.png")

RANDOM_ARMS = {"random", "llama2_7b_random", "llama2_13b_random", "llama2_70b_random"}
#: trained arm -> the matched random twin to compare it against
TWIN = {"llama2_7b": "llama2_7b_random", "llama2_13b": "llama2_13b_random",
        "llama2_70b": "llama2_70b_random"}
LAB = {"llama2_70b": "Llama-2-70B", "llama2_13b": "Llama-2-13B",
       "llama2_7b": "Llama-2-7B", "qwen3_32b": "Qwen3-32B", "qwen3_8b": "Qwen3-8B",
       "qwen3_1b7": "Qwen3-1.7B", "gpt_oss_120b": "gpt-oss-120B",
       "thalesian_cunei400m": "cuneiform-400M", "thalesian_akk300m": "AKK-300M",
       "umt5_base": "uMT5-base"}

# (surface, dataset)                      -> (matrix cell, display title)
CELLS = [
    (("eng", "world_place"), ("A", "World places\n(EN)\nsalient·high-res")),
    (("eng", "historical_figure"), ("A", "Famous figures\n(EN)\nsalient·high-res")),
    (("ent", "assyrian_ruler"), ("B", "Ruler names\n(EN)\nobscure·high-res")),
    (("akk", "eng_tier0"), ("B'", "Fragments\n(EN gloss)\nobscure·glossed")),
    (("akk", "akk_maximal"), ("C", "Fragments\n(Akkadian)\nobscure·low-res")),
]
CELL_COL = {"A": "#0e7c6b", "B": "#2f7f5b", "B'": "#c9762f", "C": "#b0501a"}


def load():
    """-> {(surface, dataset, arm, pooling, metric): (xi, rho, n)}"""
    out = {}
    for f in glob.glob(os.path.join(FIGS, "*__stats.json")):
        d = json.load(open(f))
        p = d["tag"].split("__")
        surface, arm = p[0], p[1]
        if surface == "eng":
            dataset, pooling = p[2], p[3]
        elif surface == "ent":
            dataset, pooling = p[2], p[3]          # p[4] == "bare"
        else:                                       # akk: arm, variant, target, pooling
            dataset, pooling = p[2], p[4]
        for metric, v in d["stats"].items():
            out[(surface, dataset, arm, pooling, metric)] = (
                v["xi_cos"], v["rho_geodesic"], d["n"])
    return out


def best_for(S, surface, dataset, arm, which):
    """Best value of diagnostic `which` (0=xi, 1=rho) over poolings/metrics, or None."""
    vals = [v[which] for (su, ds, a, _, _), v in S.items()
            if (su, ds, a) == (surface, dataset, arm) and v[which] == v[which]]
    return max(vals) if vals else None


def main():
    S = load()
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.4))
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    for ax, (wi, name, blurb) in zip(axes, [
            (1, r"$\rho$  —  geodesic isometry",
             "graph-geodesic distance vs. feature distance"),
            (0, r"$\xi$  —  Chatterjee (cosine)",
             "cosine similarity vs. squared feature distance")]):
        xt, labels, colors = [], [], []
        for xi_pos, ((surface, dataset), (cell, title)) in enumerate(CELLS):
            arms = sorted({a for (su, ds, a, _, _) in S
                           if (su, ds) == (surface, dataset)} - RANDOM_ARMS,
                          key=lambda a: -(best_for(S, surface, dataset, a, wi) or -9))
            drawn = 0
            for a in arms:
                v = best_for(S, surface, dataset, a, wi)
                if v is None:
                    continue
                x = xi_pos + (drawn - 1.5) * 0.15
                ax.scatter([x], [v], s=64, color=CELL_COL[cell], zorder=3,
                           edgecolor="white", linewidth=0.8)
                tw = TWIN.get(a)
                rv = best_for(S, surface, dataset, tw, wi) if tw else None
                if rv is not None:
                    ax.scatter([x], [rv], s=64, facecolor="white", zorder=3,
                               edgecolor=CELL_COL[cell], linewidth=1.6)
                    ax.plot([x, x], [rv, v], color=CELL_COL[cell], lw=1.6,
                            alpha=.55, zorder=2)
                if drawn == 0:
                    ax.annotate(LAB.get(a, a), (x, v), textcoords="offset points",
                                xytext=(0, 9), ha="center", fontsize=7.4,
                                color=CELL_COL[cell])
                drawn += 1
                if drawn == 4:
                    break
            xt.append(xi_pos); labels.append(title); colors.append(CELL_COL[cell])

        ax.axhline(0, color="#999999", lw=0.9)
        ax.set_xticks(xt)
        ax.set_xticklabels(labels, fontsize=7.6, linespacing=1.45)
        for t, c in zip(ax.get_xticklabels(), colors):
            t.set_color(c)
        ax.set_title(name, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xlabel(blurb, fontsize=8.4, color="#666666", labelpad=10)
        ax.grid(axis="y", color="#e4e4e4", lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [Line2D([], [], marker="o", ls="", mfc="#444444", mec="white", ms=8,
                      label="trained arm (best pooling / metric)"),
               Line2D([], [], marker="o", ls="", mfc="white", mec="#444444", mew=1.6,
                      ms=8, label="its matched random-init twin")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.155),
               ncol=2, frameon=False, fontsize=8.6)

    fig.suptitle("Does the representation form a manifold of the feature — "
                 "and does training make it one?",
                 fontsize=15, fontweight="bold", x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.925,
             "Modell et al. isometry diagnostics on FULL activations — 332 runs, rank-4 "
             "uncentered SVD, k = 10 graph.\nEach filled point is an arm's best cell; the "
             "open marker is the same architecture with random weights.",
             fontsize=8.8, wrap=True, color="#555555", ha="left", va="top")
    fig.text(0.055, 0.105,
             "Read the gap, not the height. ρ behaves like a world-model measure: on "
             "salient English entities the trained model separates cleanly from its twin\n"
             "(world places .29 vs .10). ξ does not — at fragment level an untrained "
             "network reaches .46–.53 against .56 for Llama-2-70B,\nso ξ there is measuring "
             "the shape of the activation cloud, not learned chronology.",
             fontsize=8.2, color="#666666", ha="left", va="top")
    fig.subplots_adjust(left=0.055, right=0.975, top=0.775, bottom=0.30, wspace=0.16)
    fig.savefig(OUT, dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
