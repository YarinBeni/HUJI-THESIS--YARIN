"""
Pillar 1 — 1b T/A/O/F factor-ladder figures (CPU, local).

Visualizes the maximal-balanced PLS control ladder that decides T/A/O/F.
Data sources (both already on disk; no cluster):
  - canonical per-layer maximal PLS for all base models + baselines:
    v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv
  - vanilla uMT5-base per-layer (from P1b job 9661):
    results/probes/umt5_base_pls__mc_balanced_maximal__summary.json
  - gpt-oss best layer (P1b): results/ladder_table.csv

Outputs:
  results/figures/factor_ladder_bars.png      — best-layer Spearman per model,
        random floor line, the (F) finetune gap (uMT5 -> Thalesian) annotated,
        size-matched Qwen-1.7B marked.
  results/figures/factor_ladder_layerwise.png — Spearman vs fractional depth for
        the four decisive models (Thalesian, uMT5, Qwen-1.7B, random).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
T1 = REPO / "v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv"
UMT5_SUM = HERE / "results/probes/umt5_base_pls__mc_balanced_maximal__summary.json"
LADDER = HERE / "results/ladder_table.csv"
FIG = HERE / "results/figures"

FLOOR_LABEL = "random (untrained)"


def umt5_layers():
    d = json.load(open(UMT5_SUM))
    rows = []
    for k, m in d["per_config"].items():
        if "__year-log" not in k:
            continue
        L = int([p for p in k.split("__") if p.startswith("L")][0][1:])
        rows.append((L, m["spearman_mean"], m.get("spearman_std", 0)))
    rows.sort()
    return (np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(T1)

    def curve(model):
        s = df[df.model == model].sort_values("layer")
        return s.layer.values, s.spearman_mean.values, s.spearman_std.values

    uL, uS, uStd = umt5_layers()

    # best-layer per model (canonical T1) + uMT5 + gpt-oss from ladder_table
    best = {}
    for m in df.model.unique():
        sub = df[df.model == m]
        i = sub.spearman_mean.idxmax()
        best[m] = (df.loc[i, "spearman_mean"], int(df.loc[i, "layer"]))
    best["umt5_base"] = (float(uS.max()), int(uL[uS.argmax()]))
    try:
        lad = pd.read_csv(LADDER)
        g = lad[(lad.cleaning == "maximal") & (lad.method == "gpt_oss_120b")]
        if len(g):
            best["gpt_oss_120b"] = (float(g.spearman.iloc[0]), -1)
    except Exception:
        pass

    floor = best["random"][0]

    # ---- Figure 1: best-layer bars -----------------------------------------
    order = [
        ("thalesian_cunei400m", "Thalesian\n(uMT5+FT)", "#1b7837"),
        ("thalesian_akk300m",   "Thal. akk300m\n(uMT5+FT)", "#5aae61"),
        ("qwen3_8b",            "Qwen3-8B", "#2166ac"),
        ("qwen3_1b7",           "Qwen3-1.7B\n(size-matched)", "#4393c3"),
        ("qwen3_32b",           "Qwen3-32B", "#92c5de"),
        ("gpt_oss_120b",        "gpt-oss-120B", "#b2abd2"),
        ("mlm",                 "MLM", "#bbbbbb"),
        ("tfidf",               "TF-IDF", "#cccccc"),
        ("random",              "random\n(floor)", "#999999"),
        ("umt5_base",           "vanilla uMT5\n(no FT)", "#b2182b"),
    ]
    order = [o for o in order if o[0] in best]
    labels = [o[1] for o in order]
    vals = [best[o[0]][0] for o in order]
    cols = [o[2] for o in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(order))
    ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.6)
    ax.axhline(floor, ls="--", color="black", lw=1.2)
    ax.text(-0.4, floor + 0.004, f"random floor {floor:.3f}",
            ha="left", va="bottom", fontsize=9)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PLS year Spearman (best layer, maximal, balanced)")
    ax.set_ylim(0, 0.46)
    ax.set_title("Pillar 1 — control ladder: why the small model wins (T/A/O/F)")

    # annotate the (F) finetune gap: uMT5 -> Thalesian
    i_um = [o[0] for o in order].index("umt5_base")
    i_th = [o[0] for o in order].index("thalesian_cunei400m")
    y_um, y_th = best["umt5_base"][0], best["thalesian_cunei400m"][0]
    ax.annotate("", xy=(i_th, y_th), xytext=(i_um, y_um),
                arrowprops=dict(arrowstyle="->", color="#1b7837", lw=2))
    ax.text((i_um + i_th) / 2, (y_um + y_th) / 2 + 0.025,
            f"(F) cuneiform finetune\nΔ=+{y_th - y_um:.3f}",
            color="#1b7837", fontsize=11, ha="center", fontweight="bold")
    ax.annotate("(A)+(T): pretrained enc-dec base\n= random floor (no FT signal)",
                xy=(i_um, y_um), xytext=(i_um - 1.3, 0.42),
                color="#b2182b", fontsize=9.5, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.5))
    fig.tight_layout()
    fig.savefig(FIG / "factor_ladder_bars.png", dpi=140)
    print("saved", FIG / "factor_ladder_bars.png")

    # ---- Figure 2: layerwise curves (fractional depth) ---------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    series = [
        ("thalesian_cunei400m", "Thalesian (uMT5+FT)", "#1b7837", "o"),
        ("qwen3_1b7",           "Qwen3-1.7B (size-matched)", "#2166ac", "s"),
        ("random",              "random (floor)", "#999999", "^"),
    ]
    for m, lab, c, mk in series:
        L, S, St = curve(m)
        frac = L / L.max()
        ax.plot(frac, S, marker=mk, color=c, label=lab, lw=1.8, ms=4)
        ax.fill_between(frac, S - St, S + St, color=c, alpha=0.12)
    # uMT5
    frac = uL / uL.max()
    ax.plot(frac, uS, marker="D", color="#b2182b", label="vanilla uMT5 (no FT)", lw=2, ms=4)
    ax.fill_between(frac, uS - uStd, uS + uStd, color="#b2182b", alpha=0.12)

    ax.axhline(floor, ls="--", color="black", lw=1, alpha=0.7)
    ax.set_xlabel("fractional encoder/transformer depth (layer / max layer)")
    ax.set_ylabel("PLS year Spearman (maximal, balanced)")
    ax.set_title("Pillar 1 — depth profile: uMT5 peaks at the embedding layer and "
                 "decays below floor;\nthe finetune builds a deep dating representation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "factor_ladder_layerwise.png", dpi=140)
    print("saved", FIG / "factor_ladder_layerwise.png")


if __name__ == "__main__":
    main()
