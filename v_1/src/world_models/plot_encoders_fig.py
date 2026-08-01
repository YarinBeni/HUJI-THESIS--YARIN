"""Deck figure: the three translation-line encoders, layer by layer.

The comparison the closing act needs: uMT5-base (multilingual pretraining, no
translation finetune) against AKK-300M (translation finetune on Akkadian only)
against cuneiform-400M (translation finetune on the multilingual cuneiform
family), with the random-init Qwen3-8B as the untrained comparator. No random
twin exists for the encoders themselves, and no 1.7B-scale random twin exists
at all, so the 8B random is the smallest untrained control available.

Eight panels: rows = cleaned Akkadian / English translation, columns = YEAR
(Spearman) and PLACE (R2) under last-token and average pooling.

    python plot_encoders_fig.py   # -> results/figs/fig_encoders_translation.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cellA_figs import COLORS, LABEL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AKK = os.path.join(HERE, "akkadian", "results", "layers_pls")
FIGS = os.path.join(HERE, "results", "figs")

ARMS = ["umt5_base", "thalesian_akk300m", "thalesian_cunei400m", "random"]
ROWS = [("akk_maximal", "cleaned Akkadian")]
COLS = [("year", "last", "test_spearman", "YEAR · last token · $\\rho$"),
        ("year", "mean", "test_spearman", "YEAR · average · $\\rho$"),
        ("geo", "last", "test_r2", "PLACE · last token · R$^2$"),
        ("geo", "mean", "test_r2", "PLACE · average · R$^2$")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 16, "axes.labelsize": 17, "axes.titlesize": 18.5,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 16,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
})


def main():
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.8), layout="constrained")
    for r, (variant, vlabel) in enumerate(ROWS):
        for c, (target, site, metric, title) in enumerate(COLS):
            ax = axes[c]
            for arm in ARMS:
                p = os.path.join(AKK, arm, f"{variant}.{target}.{site}.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                pl = sorted(d["per_layer"], key=lambda x: x["layer"])
                if len(pl) < 2:
                    continue
                top = max(x["layer"] for x in pl)
                xs = [x["layer"] / top for x in pl]
                ys = [x[metric] for x in pl]
                ctrl = arm == "random"
                ax.plot(xs, ys, color=COLORS[arm],
                        ls=(0, (5, 2)) if ctrl else "-",
                        lw=1.8 if ctrl else 3.0, alpha=0.8 if ctrl else 1.0,
                        label=LABEL[arm], zorder=2 if ctrl else 3)
                b = max(range(len(ys)), key=lambda i: ys[i])
                ax.plot(xs[b], ys[b], marker="*", ms=23 if not ctrl else 16,
                        color=COLORS[arm], mec="#111", mew=1.3,
                        zorder=7 if not ctrl else 5, clip_on=False)
            ax.set_title(title, pad=9, fontweight="bold")
            ax.set_xlabel("depth (layer / total layers)")
            ax.grid(alpha=0.22, lw=0.7)
            ax.axhline(0, color="#999", lw=0.8, zorder=0)
            if metric == "test_r2":
                ax.set_ylabel("test R$^2$")
                ax.set_ylim(-0.6, 0.6)
            else:
                ax.set_ylabel("test Spearman $\\rho$")
                ax.set_ylim(-0.05, 0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               frameon=False, handlelength=2.8)
    fig.suptitle("The translation line on cleaned Akkadian: no finetune (uMT5) vs "
                 "Akkadian-only translation (AKK-300M) vs multilingual cuneiform "
                 "translation (cuneiform-400M), against an untrained control",
                 fontsize=19, fontweight="bold")
    out = os.path.join(FIGS, "fig_encoders_translation.png")
    fig.savefig(out, facecolor="white")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
