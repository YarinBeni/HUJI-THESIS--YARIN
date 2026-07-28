"""Cell-A deck figures: the English layer sweep and the English PLS-k sweep.

Both figures show **all four read-outs** the paper mentions, so nothing is a
deviation from it: R2 (their headline) and Spearman (which they also report,
averaged over lat/lon for space), under both poolings.

Layout, for each figure: rows = SPACE (World, USA, NYC -> lat/lon) and TIME
(Figures, Art, Headlines -> year); columns = last-token pooling then mean
pooling, R2 then Spearman within each. Curves are the mean over the three
datasets of the group. Random-init arms dashed, TF-IDF a dotted floor.

    python plot_cellA_figs.py            # -> results/figs/fig_cellA_{layers,plsk}.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(RESULTS, "figs")

SPACE = ["world_place", "us_place", "nyc_place"]
TIME = ["historical_figure", "art", "headline"]
GROUPS = [("SPACE", SPACE, "World / USA / NYC → latitude, longitude"),
          ("TIME", TIME, "Figures / Art / Headlines → year")]

# family colours, matching the deck's convention: Llama purples, Qwen blues,
# translation encoders orange/green, controls grey. Darker = larger.
COLORS = {
    "llama2_70b": "#4a148c", "llama2_13b": "#7b1fa2", "llama2_7b": "#ba68c8",
    "gpt_oss_120b": "#0d1b57", "qwen3_32b": "#0d47a1", "qwen3_8b": "#1976d2",
    "qwen3_1b7": "#64b5f6",
    "umt5_base": "#2e7d32", "thalesian_cunei400m": "#e65100",
    "thalesian_akk300m": "#f9a825",
    "random": "#9e9e9e", "llama2_7b_random": "#bdbdbd",
    "llama2_13b_random": "#8d8d8d", "llama2_70b_random": "#5f5f5f",
    "tfidf": "#000000",
}
LABEL = {
    "llama2_70b": "Llama-2-70B", "llama2_13b": "Llama-2-13B",
    "llama2_7b": "Llama-2-7B", "gpt_oss_120b": "gpt-oss-120B",
    "qwen3_32b": "Qwen3-32B", "qwen3_8b": "Qwen3-8B", "qwen3_1b7": "Qwen3-1.7B",
    "umt5_base": "uMT5-base", "thalesian_cunei400m": "cuneiform-400M",
    "thalesian_akk300m": "AKK-300M", "random": "random Qwen3-8B",
    "llama2_7b_random": "Llama-2-7B rand", "llama2_13b_random": "Llama-2-13B rand",
    "llama2_70b_random": "Llama-2-70B rand", "tfidf": "TF-IDF floor",
}
ORDER = ["llama2_70b", "llama2_13b", "llama2_7b", "gpt_oss_120b", "qwen3_32b",
         "qwen3_8b", "qwen3_1b7", "umt5_base", "thalesian_cunei400m",
         "thalesian_akk300m", "tfidf", "llama2_70b_random", "llama2_13b_random",
         "llama2_7b_random", "random"]
IS_CTRL = {"random", "tfidf", "llama2_7b_random", "llama2_13b_random",
           "llama2_70b_random"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.linewidth": 0.9, "lines.linewidth": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
})


def _style(ax, metric, xlabel, title, sub=None):
    ax.set_title(title, pad=8, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test R$^2$" if metric == "r2" else "test Spearman $\\rho$")
    ax.grid(alpha=0.22, lw=0.7)
    ax.axhline(0, color="#999", lw=0.8, zorder=0)
    if metric == "r2":
        # symlog: linear within +-0.1, logarithmic outside, so the deep negative
        # tail of the failing arms fits without squashing the 0-1 band.
        ax.set_yscale("symlog", linthresh=0.1, linscale=0.7)
        ax.set_yticks([-10, -1, 0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["-10", "-1", "0", ".25", ".5", ".75", "1"])
        # a couple of arms dive far below zero in the earliest layers; clip the
        # view so that excursion stays visible without owning the panel.
        ax.set_ylim(-3, 1.25)
    else:
        ax.set_ylim(-0.15, 1.02)


def _legend(fig, axes):
    handles, labels = [], []
    for a in axes.ravel():
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    idx = sorted(range(len(labels)),
                 key=lambda i: ORDER.index(
                     next((k for k, v in LABEL.items() if v == labels[i]), "random")))
    fig.legend([handles[i] for i in idx], [labels[i] for i in idx],
               loc="outside lower center", ncol=8, frameon=False)


def fig_layers():
    df = pd.read_csv(os.path.join(RESULTS, "summary_layerwise.csv"))
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.8), layout="constrained")
    for r, (gname, datasets, gsub) in enumerate(GROUPS):
        for c, (site, metric) in enumerate([("last", "r2"), ("last", "spearman"),
                                            ("mean", "r2"), ("mean", "spearman")]):
            ax = axes[r, c]
            col = "test_r2" if metric == "r2" else "test_spearman"
            sub = df[(df.entity_type.isin(datasets)) & (df.site == site)]
            for method in ORDER:
                g = sub[sub.method == method]
                if g.empty:
                    continue
                if method == "tfidf":
                    ax.axhline(g[col].mean(), color="k", lw=1.3, ls=":",
                               label=LABEL[method], zorder=1)
                    continue
                curve = g.groupby("layer")[col].mean().sort_index()
                if len(curve) < 2:
                    continue
                x = curve.index / curve.index.max()
                ctrl = method in IS_CTRL
                ax.plot(x, curve.values, color=COLORS[method],
                        ls="--" if ctrl else "-", lw=1.5 if ctrl else 2.1,
                        alpha=0.85 if ctrl else 1.0, label=LABEL[method], zorder=2)
                b = curve.values.argmax()
                ax.plot(x[b], curve.values[b], marker="*", ms=13,
                        color=COLORS[method], mec="white", mew=0.8, zorder=3)
            pool = "last token" if site == "last" else "mean pool"
            met = "R$^2$" if metric == "r2" else "Spearman $\\rho$"
            _style(ax, metric, "depth (layer / total layers)",
                   f"{gname} · {pool} · {met}")
            if c == 0:
                ax.text(-0.30, 0.5, gsub, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11.5, color="#444")
    _legend(fig, axes)
    fig.suptitle("English (cell A): where in the network space and time live. "
                 "Per-layer ridge probe, both metrics, both poolings.",
                 fontsize=18, fontweight="bold")
    out = os.path.join(FIGS, "fig_cellA_layers.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[write] {out}")


def fig_plsk():
    recs = []
    for arm_dir in sorted(glob.glob(os.path.join(RESULTS, "eng_pls", "*"))):
        arm = os.path.basename(arm_dir)
        for path in glob.glob(os.path.join(arm_dir, "*.json")):
            ds, site, _ = os.path.basename(path).split(".")
            d = json.load(open(path))
            for k, sc in d["pls_at_best_layer"].items():
                recs.append({"method": arm, "entity_type": ds, "site": site,
                             "k": int(k), "test_r2": sc["test_r2"],
                             "test_spearman": sc["test_spearman"]})
    df = pd.DataFrame(recs)
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.8), layout="constrained")
    for r, (gname, datasets, gsub) in enumerate(GROUPS):
        for c, (site, metric) in enumerate([("last", "r2"), ("last", "spearman"),
                                            ("mean", "r2"), ("mean", "spearman")]):
            ax = axes[r, c]
            col = "test_r2" if metric == "r2" else "test_spearman"
            sub = df[(df.entity_type.isin(datasets)) & (df.site == site)]
            for method in ORDER:
                g = sub[sub.method == method]
                if g.empty:
                    continue
                curve = g.groupby("k")[col].mean().sort_index()
                if len(curve) < 2:
                    continue
                ctrl = method in IS_CTRL
                ax.plot(curve.index, curve.values, color=COLORS[method],
                        marker="o", ms=4, ls="--" if ctrl else "-",
                        lw=1.5 if ctrl else 2.1, alpha=0.85 if ctrl else 1.0,
                        label=LABEL[method], zorder=2)
                b = curve.values.argmax()
                ax.plot(curve.index[b], curve.values[b], marker="*", ms=13,
                        color=COLORS[method], mec="white", mew=0.8, zorder=3)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
            ax.axvline(16, color="#c62828", lw=1.1, ls="-.", alpha=0.55, zorder=1)
            pool = "last token" if site == "last" else "mean pool"
            met = "R$^2$" if metric == "r2" else "Spearman $\\rho$"
            _style(ax, metric, "PLS components k", f"{gname} · {pool} · {met}")
            if c == 0:
                ax.text(-0.30, 0.5, gsub, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11.5, color="#444")
    _legend(fig, axes)
    fig.suptitle("English (cell A): how many PLS directions the world model needs. "
                 "k = 1 to 64 at each arm's best ridge layer; dash-dot line marks k = 16.",
                 fontsize=18, fontweight="bold")
    out = os.path.join(FIGS, "fig_cellA_plsk.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[write] {out}")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_layers()
    fig_plsk()
