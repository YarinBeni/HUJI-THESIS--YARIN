"""Cell-C deck figures: the Akkadian layer sweep and the Akkadian PLS-k sweep.

Same house style as the cell-A figures (plot_cellA_figs.py): one hue per model
family, controls purple and dashed, star on each arm's best point, so the two
halves of the deck can be read against each other directly.

Four panels each, all on the raw Akkadian text: YEAR under both poolings
(Spearman, the ranking read-out dating actually needs) and GEO under both
poolings (R2, the paper's read-out for coordinates). The English-gloss row that
used to sit underneath is dropped, since the cell-B slides already carry it.

    python plot_cellC_figs.py       # -> results/figs/fig_cellC_{layers,plsk}.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cellA_figs import COLORS, IS_CTRL, LABEL, ORDER, _legend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AKK = os.path.join(HERE, "akkadian", "results")
FIGS = os.path.join(HERE, "results", "figs")
VARIANT = "akk_maximal"

# (target, pooling, metric, panel title)
PANELS = [
    ("year", "last", "test_spearman", "YEAR · text last token · Spearman $\\rho$"),
    ("year", "mean", "test_spearman", "YEAR · text average · Spearman $\\rho$"),
    ("geo", "last", "test_r2", "PLACE · text last token · R$^2$"),
    ("geo", "mean", "test_r2", "PLACE · text average · R$^2$"),
]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
})


def load(arm, target, site):
    p = os.path.join(AKK, "layers_pls", arm, f"{VARIANT}.{target}.{site}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _style(ax, metric, xlabel, title):
    ax.set_title(title, pad=9, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test Spearman $\\rho$" if metric == "test_spearman"
                  else "test R$^2$")
    ax.grid(alpha=0.22, lw=0.7)
    ax.axhline(0, color="#999", lw=0.8, zorder=0)
    if metric == "test_r2":
        ax.set_yscale("symlog", linthresh=0.1, linscale=0.7)
        ax.set_yticks([-1, 0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["-1", "0", ".25", ".5", ".75", "1"])
        ax.set_ylim(-1.5, 1.15)
    else:
        ax.set_ylim(-0.1, 0.95)


def _draw(ax, curves, metric, xlabel, title, logx=False):
    for arm, xs, ys in curves:
        ctrl = arm in IS_CTRL
        ax.plot(xs, ys, color=COLORS[arm], ls=(0, (5, 2)) if ctrl else "-",
                lw=1.7 if ctrl else 2.8, alpha=0.75 if ctrl else 1.0,
                marker="o" if logx else None, ms=4.5,
                label=LABEL[arm], zorder=2 if ctrl else 3)
        b = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot(xs[b], ys[b], marker="*", ms=22 if not ctrl else 15,
                color=COLORS[arm], mec="#111", mew=1.3,
                zorder=7 if not ctrl else 5, clip_on=False)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    _style(ax, metric, xlabel, title)


def fig_layers():
    fig, axes = plt.subplots(2, 2, figsize=(19, 11.5), layout="constrained")
    for ax, (target, site, metric, title) in zip(axes.ravel(), PANELS):
        curves = []
        for arm in ORDER:
            d = load(arm, target, site)
            if not d or len(d["per_layer"]) < 2:
                continue
            pl = sorted(d["per_layer"], key=lambda r: r["layer"])
            top = max(r["layer"] for r in pl)
            curves.append((arm, [r["layer"] / top for r in pl],
                           [r[metric] for r in pl]))
        _draw(ax, curves, metric, "depth (layer / total layers)", title)
    _legend(fig, axes)
    fig.suptitle("Raw Akkadian (cell C): where in the network the year and the "
                 "find-spot live, per-layer ridge probe",
                 fontsize=20, fontweight="bold")
    out = os.path.join(FIGS, "fig_cellC_layers.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[write] {out}")


def fig_plsk():
    fig, axes = plt.subplots(2, 2, figsize=(19, 11.5), layout="constrained")
    for ax, (target, site, metric, title) in zip(axes.ravel(), PANELS):
        curves = []
        for arm in ORDER:
            d = load(arm, target, site)
            if not d:
                continue
            at = d["pls_at_best_layer"]
            ks = sorted(int(k) for k in at)
            curves.append((arm, ks, [at[str(k)][metric] for k in ks]))
        _draw(ax, curves, metric, "PLS components k", title, logx=True)
        ax.axvline(16, color="#c62828", lw=1.2, ls="-.", alpha=0.5, zorder=1)
    _legend(fig, axes)
    fig.suptitle("Raw Akkadian (cell C): how many PLS directions the signal needs, "
                 "k = 1 to 64 at each arm's best layer",
                 fontsize=20, fontweight="bold")
    out = os.path.join(FIGS, "fig_cellC_plsk.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[write] {out}")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_layers()
    fig_plsk()
