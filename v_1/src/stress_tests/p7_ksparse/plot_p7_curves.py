"""plot_p7_curves.py — the per-k sparsity curves Yarin asked for.

Two figures, each with THREE panels (tier0 | maximal | maxking):
  fig_p7_kcurves_cls.png — x = k neurons (log2), y = macro-F1 (classification probe)
  fig_p7_kcurves_reg.png — x = k neurons (log2), y = Spearman  (year-regression probe)
One line per model, evaluated at that model x cleaning's BEST layer (by the full-k
metric of the family being plotted). Random model drawn black/dashed as the floor.

Reads results/v2/p7_v2__<method>__<cleaning>.json (from probe_p7_v2.py).
Usage:  python plot_p7_curves.py    (writes to ../results/eda/)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE / "results" / "v2"
OUT = HERE.parents[0] / "results" / "eda"
CLEANINGS = ["tier0", "maximal", "maxking"]
MODELS = ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b",
          "thalesian_akk300m", "thalesian_cunei400m", "umt5_base", "random"]
COLORS = {"qwen3_1b7": "#8ecae6", "qwen3_8b": "#219ebc", "qwen3_32b": "#125e8a",
          "gpt_oss_120b": "#5a189a", "thalesian_akk300m": "#f4a261",
          "thalesian_cunei400m": "#e76f51", "umt5_base": "#2a9d8f", "random": "#222222"}


def curve(d, family):
    """Return (ks, values) at the best layer for this family."""
    key = "macro_f1" if family == "cls" else "reg_spearman"
    bl = d.get("best_layer_cls" if family == "cls" else "best_layer_reg")
    if bl is None or "per_layer" not in d:
        return None
    ks = sorted(int(k) for k in d["per_layer"][bl])
    return ks, [d["per_layer"][bl][str(k)][key] for k in ks]


def make(family, ylabel, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, cl in zip(axes, CLEANINGS):
        for m in MODELS:
            fp = V2 / f"p7_v2__{m}__{cl}.json"
            if not fp.exists():
                continue
            d = json.loads(fp.read_text())
            c = curve(d, family)
            if c is None:
                continue
            ks, vals = c
            style = dict(color=COLORS.get(m, "#888"), lw=2,
                         ls="--" if m == "random" else "-",
                         marker="o", ms=3.5, label=m)
            ax.plot(ks, vals, **style)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        ax.set_title(cl); ax.set_xlabel("k neurons"); ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"P7 k-sparse curves — {ylabel} at each model's best layer (random dashed)", y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / fname)


if __name__ == "__main__":
    make("cls", "macro-F1 (before/after-median classification)", "fig_p7_kcurves_cls.png")
    make("reg", "Spearman (year regression)", "fig_p7_kcurves_reg.png")
