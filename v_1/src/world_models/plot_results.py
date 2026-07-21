"""W figures: (a) layerwise test-R2 curves per dataset (the Figure-2 analog,
random arms dashed), (b) best-layer world-map / timeline scatter per method.

    python plot_results.py            # writes results/figs/*.png
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm_lib.entity_data import ENTITY_TYPES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGS_DIR = os.path.join(RESULTS_DIR, "figs")

COLORS = {
    "llama2_70b": "#7b1fa2", "llama2_13b": "#9c27b0", "llama2_7b": "#ce93d8",
    "gpt_oss_120b": "#1a237e", "qwen3_32b": "#0d47a1", "qwen3_8b": "#1976d2",
    "qwen3_1b7": "#64b5f6",
    "umt5_base": "#2e7d32", "thalesian_cunei400m": "#e65100",
    "thalesian_akk300m": "#ef6c00",
    "random": "#9e9e9e", "llama2_7b_random": "#bdbdbd",
    "llama2_13b_random": "#757575", "llama2_70b_random": "#424242",
    "tfidf": "#000000",
}


def layer_curves():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "summary_layerwise.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    for ax, et in zip(axes.ravel(), ENTITY_TYPES):
        sub = df[df.entity_type == et]
        for method, g in sub.groupby("method"):
            # one canonical site per method (first in file order is fine here)
            site = g.site.iloc[0]
            g = g[g.site == site].sort_values("layer")
            if g.layer.max() <= 0:  # tfidf: draw as a horizontal floor
                ax.axhline(g.test_r2.iloc[0], color="k", lw=1, ls=":",
                           label="tfidf")
                continue
            frac = g.layer / g.layer.max()
            dashed = "random" in method
            ax.plot(frac, g.test_r2, label=method,
                    color=COLORS.get(method), ls="--" if dashed else "-",
                    lw=1.2 if dashed else 1.8)
        ax.set_title(et)
        ax.set_xlabel("layer depth fraction")
        ax.set_ylabel("test R²")
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
               frameon=False)
    fig.suptitle("Linear decodability of space & time by layer "
                 "(Gurnee & Tegmark protocol, our ladder + random controls)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = os.path.join(FIGS_DIR, "layer_curves_test_r2.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def projection_maps():
    for path in glob.glob(os.path.join(
            RESULTS_DIR, "projections", "*", "*.csv.gz")):
        method = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path).replace(".csv.gz", "")
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(7, 5))
        te = df[df.is_test]
        if "pred_lon" in df.columns:
            ax.scatter(te.pred_lon, te.pred_lat, s=2, alpha=0.3,
                       c=te.lat, cmap="viridis")
            ax.set_xlabel("predicted longitude")
            ax.set_ylabel("predicted latitude")
        else:
            ax.scatter(te["true"], te["pred"], s=2, alpha=0.3)
            lims = [te["true"].min(), te["true"].max()]
            ax.plot(lims, lims, "k--", lw=1)
            ax.set_xlabel("true year")
            ax.set_ylabel("predicted year")
        ax.set_title(f"{method} — {name} (test set)")
        fig.tight_layout()
        out = os.path.join(FIGS_DIR, f"proj_{method}_{name}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    os.makedirs(FIGS_DIR, exist_ok=True)
    layer_curves()
    projection_maps()
