"""Local, self-contained maximal/mean/balanced year-PLS comparison plot.

Reads only committed tables (no cluster activations needed):
  - base 8 models:  v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv
  - fine-tune arms: v_1/src/finetune/results/scoreboard_layers.csv (cleaning=maximal)

Produces two panels into v_1/src/finetune/results/figures/:
  maximal_pls_bestlayer.png  — best-layer Spearman per model/arm (bars, ±std)
  maximal_pls_layerwise.png  — Spearman vs layer curves per model/arm

Re-run after FT5b (32B) lands to fold in the new arms automatically — it
picks up whatever is in scoreboard_layers.csv.

    venv/bin/python v_1/src/finetune/plot_maximal_pls.py
    venv/bin/python v_1/src/finetune/plot_maximal_pls.py --with-ft   # add FT arms (best cut/family)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
T1 = REPO / "v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv"
SCORE = REPO / "v_1/src/finetune/results/scoreboard_layers.csv"
OUT = REPO / "v_1/src/finetune/results/figures"

# the 8 base models, display order + labels (matches the maximal_figs set)
BASE_ORDER = [
    ("tfidf", "TF-IDF"),
    ("mlm", "MLM-37M"),
    ("thalesian_akk300m", "Thalesian-300M"),
    ("thalesian_cunei400m", "Thalesian-400M"),
    ("random", "random-Qwen3"),
    ("qwen3_1b7", "Qwen3-1.7B"),
    ("qwen3_8b", "Qwen3-8B"),
    ("qwen3_32b", "Qwen3-32B"),
]


def load_base():
    """{model -> [(layer, mean, std), ...]} from the committed maximal table."""
    curves = defaultdict(list)
    with open(T1) as f:
        for r in csv.DictReader(f):
            curves[r["model"]].append(
                (int(str(r["layer"]).lstrip("L")),
                 float(r["spearman_mean"]),
                 float(r["spearman_std"] or "nan")))
    for m in curves:
        curves[m].sort()
    return curves


def load_ft_best():
    """{(family,arm) -> [(layer,mean,std)]} for maximal FT arms (arm != base)."""
    curves = defaultdict(list)
    if not SCORE.exists():
        return curves
    with open(SCORE) as f:
        for r in csv.DictReader(f):
            if r["cleaning"] != "maximal" or r["arm"] == "base":
                continue
            curves[(r["family"], r["arm"])].append(
                (int(r["layer"]), float(r["spearman_mean"]),
                 float(r["spearman_std"] or "nan")))
    for k in curves:
        curves[k].sort()
    return curves


def best_of(curve):
    return max(curve, key=lambda t: t[1])  # (layer, mean, std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-ft", action="store_true",
                    help="overlay the best fine-tuned cut per family")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    base = load_base()
    bars, layerwise = [], []   # (label, mean, std, color), (label, curve, color, style)

    # base models
    for i, (m, lab) in enumerate(BASE_ORDER):
        if m not in base:
            continue
        c = plt.cm.tab10(i % 10)
        L, mean, std = best_of(base[m])
        bars.append((lab, mean, std, c))
        layerwise.append((f"{lab}", base[m], c, "-"))

    # fine-tuned arms (best cut per family) — optional
    ft_label = ""
    if args.with_ft:
        ft = load_ft_best()
        per_family = defaultdict(list)
        for (fam, arm), curve in ft.items():
            per_family[fam].append((arm, best_of(curve), curve))
        fam_color = {"qwen3_1b7": plt.cm.tab10(5), "qwen3_8b": plt.cm.tab10(6),
                     "qwen3_32b": plt.cm.tab10(7), "gpt_oss_120b": "black"}
        fam_lab = {"qwen3_1b7": "Qwen3-1.7B", "qwen3_8b": "Qwen3-8B",
                   "qwen3_32b": "Qwen3-32B", "gpt_oss_120b": "gpt-oss-120B"}
        for fam, arms in per_family.items():
            arm, (L, mean, std), curve = max(arms, key=lambda a: a[1][1])
            lab = f"{fam_lab.get(fam, fam)} +NTP ({arm})"
            bars.append((lab, mean, std, fam_color.get(fam, "gray")))
            layerwise.append((lab, curve, fam_color.get(fam, "gray"), "--"))
        ft_label = " + best NTP arm/family"

    # ---- panel 1: best-layer bars ----
    bars.sort(key=lambda b: b[1])
    fig, ax = plt.subplots(figsize=(9, 5))
    ys = range(len(bars))
    ax.barh([b[0] for b in bars], [b[1] for b in bars],
            xerr=[b[2] for b in bars], color=[b[3] for b in bars],
            alpha=0.85, capsize=3)
    ax.set_xlabel("best-layer year-PLS Spearman")
    ax.set_title(f"Maximal · mean-pool · balanced (200 draws){ft_label}")
    ax.grid(axis="x", alpha=0.3)
    for i, b in enumerate(bars):
        ax.text(b[1] + 0.005, i, f"{b[1]:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    p1 = OUT / "maximal_pls_bestlayer.png"
    fig.savefig(p1, dpi=150); plt.close(fig)

    # ---- panel 2: layerwise curves ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lab, curve, color, style in layerwise:
        xs = [t[0] for t in curve]; ys = [t[1] for t in curve]
        ax.plot(xs, ys, style, color=color, lw=1.6, ms=3, marker="o", label=lab)
    ax.set_xlabel("layer (hidden-state index)")
    ax.set_ylabel("year-PLS Spearman")
    ax.set_title(f"Maximal · mean-pool · balanced — layerwise{ft_label}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p2 = OUT / "maximal_pls_layerwise.png"
    fig.savefig(p2, dpi=150); plt.close(fig)

    print(f"[plot] {len(bars)} series")
    print(f"[plot] {p1}")
    print(f"[plot] {p2}")


if __name__ == "__main__":
    main()
