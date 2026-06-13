#!/usr/bin/env python3
"""Round-3 paper figures — table-driven.

Reads the released CSVs under results/tables/ and writes PNG+PDF figures
under results/figures/round3_story/, plus T_headlines.csv.

Design principles (top-conf rewrite):
  - Each model gets a unique color; family is preserved by hue gradient.
  - Each panel has: (1) descriptive in-axes title, (2) one-line config
    subtitle (regime / cleaning / pooling / n_draws), (3) explicit
    chance + shuffled reference lines where applicable.
  - Prompt variants (pv0..pv3) and king-probe variants (kp0..kp2) get
    human-readable labels.
  - A new Figure 0 summarizes everything as a forest plot.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-lititure-review")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
TABLE_DIR = HERE / "results" / "tables"
OUT_DIR = HERE / "results" / "figures" / "round3_story"

# ---------------------------------------------------------------------------
# Model registry — every model has a unique color + marker
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "metadata",
    "tfidf",
    "mlm",
    "thalesian_akk300m",
    "thalesian_cunei400m",
    "qwen3_1b7",
    "qwen3_8b",
    "qwen3_32b",
    "gpt_oss_120b",
    "random",
]

MODEL_LABEL = {
    "metadata": "Metadata-only",
    "tfidf": "TF-IDF (char n-gram)",
    "mlm": "MLM 37M",
    "thalesian_akk300m": "Thalesian 300M",
    "thalesian_cunei400m": "Thalesian 400M",
    "qwen3_1b7": "Qwen3 1.7B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_32b": "Qwen3 32B",
    "gpt_oss_120b": "GPT-OSS 120B",
    "random": "Random-init 8B",
}

MODEL_SHORT = {  # for tight axis tick labels
    "metadata": "Metadata",
    "tfidf": "TF-IDF",
    "mlm": "MLM 37M",
    "thalesian_akk300m": "Thal 300M",
    "thalesian_cunei400m": "Thal 400M",
    "qwen3_1b7": "Qwen3 1.7B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_32b": "Qwen3 32B",
    "gpt_oss_120b": "GPT-OSS 120B",
    "random": "Random 8B",
}

PARAMS_B = {
    "metadata": 0.0,
    "tfidf": 0.0,
    "mlm": 0.0367,
    "thalesian_akk300m": 0.300,
    "thalesian_cunei400m": 0.400,
    "qwen3_1b7": 1.7,
    "qwen3_8b": 8.0,
    "qwen3_32b": 32.0,
    "gpt_oss_120b": 120.0,
    "random": 8.0,
}

FAMILY = {
    "metadata": "metadata",
    "tfidf": "surface",
    "mlm": "akkadian",
    "thalesian_akk300m": "akkadian",
    "thalesian_cunei400m": "akkadian",
    "qwen3_1b7": "qwen",
    "qwen3_8b": "qwen",
    "qwen3_32b": "qwen",
    "gpt_oss_120b": "gpt_oss",
    "random": "random",
}

# Per-model unique colors — family hue preserved via gradient.
# akkadian = warm oranges/reds, qwen = greens, surface = blue,
# random = purple, metadata = grey.
MODEL_COLOR = {
    "metadata":            "#7f7f7f",
    "tfidf":               "#1f6fb4",
    "mlm":                 "#f4a259",   # light orange
    "thalesian_akk300m":   "#d8602a",   # mid orange
    "thalesian_cunei400m": "#8b3a0e",   # dark orange / brick
    "qwen3_1b7":           "#8fcf6e",   # light green
    "qwen3_8b":            "#4b8f3a",   # mid green
    "qwen3_32b":           "#1f4d12",   # dark green
    "gpt_oss_120b":        "#c0392b",   # crimson red
    "random":              "#7a4ec0",
}

MODEL_MARKER = {
    "metadata":            "P",
    "tfidf":               "D",
    "mlm":                 "s",
    "thalesian_akk300m":   "s",
    "thalesian_cunei400m": "s",
    "qwen3_1b7":           "o",
    "qwen3_8b":            "o",
    "qwen3_32b":           "o",
    "gpt_oss_120b":        "^",
    "random":              "X",
}

# linestyle per family — for layer-wise line plots
FAMILY_LINESTYLE = {
    "akkadian": "-",
    "qwen":     "--",
    "gpt_oss":  (0, (5, 2)),
    "random":   ":",
    "surface":  "-.",
    "metadata": (0, (1, 1)),
}

# Human-readable variant names
PV_LABEL = {
    "pv0": "raw\n(no instruction)",
    "pv1": "zero-shot\n('what year?')",
    "pv2": "uncertainty\n('decline if unsure')",
    "pv3": "instruct\n('[INST] date…')",
}

KP_LABEL = {
    "kp0": "date recall\n(±50 yr)",
    "kp1": "king recall\n(period→kings)",
    "kp2": "hallucination\n(fake-king test)",
}


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def read_tables() -> dict[str, pd.DataFrame]:
    return {
        "t1":  pd.read_csv(TABLE_DIR / "T1_year_pls.csv"),
        "t2":  pd.read_csv(TABLE_DIR / "T2_year_ridge.csv"),
        "t3b": pd.read_csv(TABLE_DIR / "T3b_ruler_plsda.csv"),
        "t4":  pd.read_csv(TABLE_DIR / "T4_geodesic.csv"),
        "t5":  pd.read_csv(TABLE_DIR / "T5_loro.csv"),
        "t7":  pd.read_csv(TABLE_DIR / "T7_name_masking.csv"),
        "t8":  pd.read_csv(TABLE_DIR / "T8_metadata_baseline.csv"),
        "t9":  pd.read_csv(TABLE_DIR / "T9_elicitation.csv"),
        "t10": pd.read_csv(TABLE_DIR / "T10_prompt_reprobe.csv"),
    }


def layer_int(value) -> int:
    if pd.isna(value):
        return -1
    text = str(value)
    if text.startswith("L"):
        text = text[1:]
    try:
        return int(text)
    except ValueError:
        return -1


def with_model_meta(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    out = df.copy()
    out["params_b"] = out[model_col].map(PARAMS_B)
    out["family"] = out[model_col].map(FAMILY)
    out["model_label"] = out[model_col].map(MODEL_LABEL)
    out["color"] = out[model_col].map(MODEL_COLOR)
    out["marker"] = out[model_col].map(MODEL_MARKER)
    out["model_order"] = out[model_col].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out


def best_rows(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
    ascending: bool = False,
) -> pd.DataFrame:
    valid = df.dropna(subset=[metric]).copy()
    if valid.empty:
        return valid
    if ascending:
        idx = valid.groupby(group_cols, dropna=False)[metric].idxmin()
    else:
        idx = valid.groupby(group_cols, dropna=False)[metric].idxmax()
    return valid.loc[idx].reset_index(drop=True)


def model_x(model: str, zero_offsets: dict[str, float] | None = None) -> float:
    params = PARAMS_B[model]
    if params == 0:
        offset = 0.0 if zero_offsets is None else zero_offsets.get(model, 0.0)
        return -1.75 + offset
    return float(np.log10(params))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "regular",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, dx: float = -0.14, dy: float = 1.10) -> None:
    ax.text(
        dx, dy, label,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        va="bottom", ha="left",
    )


def set_panel_title(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    """Bold descriptive title with a small grey config subtitle below."""
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=14)
    if subtitle:
        ax.text(
            0.0, 1.015, subtitle,
            transform=ax.transAxes,
            fontsize=8.5, color="#555555",
            style="italic", va="bottom", ha="left",
        )


def add_shuffled_band(ax: plt.Axes, low: float, high: float, label: str = "shuffled-label null") -> None:
    """Grey band across the axes to mark the shuffled baseline."""
    ax.axhspan(low, high, color="#d0d0d0", alpha=0.45, zorder=0, label=label)


def model_legend_handles(models: list[str]) -> list[plt.Line2D]:
    return [
        plt.Line2D([0], [0],
                   marker=MODEL_MARKER[m], color=MODEL_COLOR[m],
                   markersize=7, linestyle="none",
                   markeredgecolor="black" if m == "random" else "white",
                   markeredgewidth=0.6,
                   label=MODEL_SHORT[m])
        for m in models
    ]


def smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple centered rolling mean; window must be odd."""
    if len(y) < window:
        return y
    pad = window // 2
    yp = np.pad(y, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(yp, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Headlines summary CSV (kept the same shape, used by Fig 0)
# ---------------------------------------------------------------------------

def build_headlines(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    t1 = tables["t1"]
    raw = t1[t1["year_transform"].eq("raw")]
    b = best_rows(raw, ["regime", "model"], "spearman_mean")
    rows.append(b.assign(
        test="T1_year_pls", unit="year",
        headline_metric="spearman_mean",
        headline_value=b["spearman_mean"], headline_std=b["spearman_std"]))

    t2 = tables["t2"]
    raw = t2[t2["year_transform"].eq("raw")]
    b = best_rows(raw, ["regime", "model"], "spearman_mean")
    rows.append(b.assign(
        test="T2_year_ridge", unit="year",
        headline_metric="spearman_mean",
        headline_value=b["spearman_mean"], headline_std=b["spearman_std"]))

    t3b = tables["t3b"]
    b = best_rows(t3b, ["regime", "model"], "macro_f1")
    rows.append(b.assign(
        test="T3b_ruler_plsda", unit="ruler",
        headline_metric="macro_f1",
        headline_value=b["macro_f1"], headline_std=b["macro_f1_std"]))

    t4 = tables["t4"].rename(columns={"method": "model"})
    b = best_rows(t4, ["regime", "model"], "isomap_pairwise_acc")
    rows.append(b.assign(
        test="T4_geodesic", unit="geometry",
        headline_metric="isomap_pairwise_acc",
        headline_value=b["isomap_pairwise_acc"],
        headline_std=b["isomap_pairwise_acc_std"]))

    t5 = tables["t5"].rename(columns={"method": "model"})
    rows.append(t5.assign(
        test="T5_loro", unit="geometry_control",
        headline_metric="drop",
        headline_value=t5["drop"], headline_std=t5["drop_std"]))

    t7 = tables["t7"].copy()
    rows.append(t7.assign(
        test="T7_name_masking", regime="balanced",
        model="tfidf", pool="na", layer="L00",
        unit="name_masking",
        headline_metric="year_spearman_mean",
        headline_value=t7["year_spearman_mean"],
        headline_std=t7["year_spearman_std"]))

    t8 = tables["t8"].copy()
    raw = t8[t8["year_transform"].eq("raw")]
    rows.append(raw.assign(
        test="T8_metadata_baseline", model="metadata",
        unit="metadata",
        headline_metric="spearman_mean",
        headline_value=raw["spearman_mean"],
        headline_std=raw["spearman_std"]))

    t9 = tables["t9"].copy()
    rows.append(t9.assign(
        test="T9_elicitation", regime="prompt",
        unit="elicitation",
        headline_metric=t9["headline_metric"],
        headline_value=t9["headline_value"], headline_std=np.nan))

    t10 = tables["t10"].dropna(subset=["headline_value"]).copy()
    b = best_rows(t10, ["model", "variant", "task"], "headline_value")
    rows.append(b.assign(
        test="T10_prompt_reprobe", regime="prompt",
        unit=b["task"],
        headline_metric=b["headline_metric"],
        headline_value=b["headline_value"], headline_std=b["std"]))

    out = pd.concat(rows, ignore_index=True, sort=False)
    if "method" in out.columns:
        out = out.drop(columns=["method"])
    out = with_model_meta(out, "model")
    out = out.sort_values(
        ["test", "regime", "model_order", "variant", "task"],
        na_position="last",
    ).reset_index(drop=True)
    out.to_csv(TABLE_DIR / "T_headlines.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# FIGURE 0 — money-figure forest plot of balanced year-Spearman
# ---------------------------------------------------------------------------

def plot_summary_forest(tables: dict[str, pd.DataFrame]) -> None:
    t1 = tables["t1"]
    t8 = tables["t8"]

    # PLS + Ridge balanced/balanced_last, year-raw, best layer per (model, regime, pool)
    cfg = {
        "balanced":      ("PLS · balanced · mean-pool",  "balanced",      "T1"),
        "balanced_last": ("PLS · balanced · last-token", "balanced_last", "T1"),
    }
    sources = []
    for label, (pretty, regime, _) in cfg.items():
        sub = t1[(t1["regime"].eq(regime)) & (t1["year_transform"].eq("raw"))]
        b = best_rows(sub, ["model"], "spearman_mean")
        b = b.assign(cfg_label=pretty)
        sources.append(b)
    df = pd.concat(sources, ignore_index=True)
    df = with_model_meta(df)

    # Plot order: model_order (top-down), within model the two configs in fixed order
    cfg_order = ["PLS · balanced · mean-pool", "PLS · balanced · last-token"]
    df["cfg_rank"] = df["cfg_label"].map({c: i for i, c in enumerate(cfg_order)})
    df = df.sort_values(["model_order", "cfg_rank"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.6))

    # Shuffled null band (from T1 shuffled_spearman across the table, ~0)
    shuf = t1["shuffled_spearman_mean"].dropna()
    shuf_lo, shuf_hi = float(shuf.quantile(0.05)), float(shuf.quantile(0.95))
    add_shuffled_band(ax, shuf_lo, shuf_hi, label="shuffled-label null (5–95%)")

    # Metadata-only vertical reference
    meta = t8[t8["regime"].eq("balanced") & t8["year_transform"].eq("raw")].iloc[0]
    ax.axvline(meta["spearman_mean"], color="#777777", linestyle=":", linewidth=1.4,
               label=f"metadata-only Sp={meta['spearman_mean']:.2f}")

    y = np.arange(len(df))
    for yi, row in zip(y, df.itertuples()):
        ax.errorbar(
            row.spearman_mean, yi,
            xerr=row.spearman_std,
            marker=row.marker, color=row.color,
            markersize=8, capsize=3,
            markeredgecolor="black" if row.model == "random" else "white",
            markeredgewidth=0.6,
            linestyle="none", elinewidth=1.1,
        )
        ax.text(
            row.spearman_mean + row.spearman_std + 0.012, yi,
            f"{row.spearman_mean:+.2f}",
            fontsize=8, va="center", color=row.color, fontweight="bold",
        )

    labels = [f"{MODEL_SHORT[r.model]}   ·   {r.cfg_label.split('·')[-1].strip()}"
              for r in df.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-0.10, 0.55)
    ax.set_xlabel("Balanced year Spearman  (higher = better temporal ordering)")
    set_panel_title(
        ax,
        "Summary — temporal-ordering signal across models & poolings",
        "200 MC draws × 5-fold CV  ·  error bars = ±1 SD across draws  ·  "
        "year-raw target  ·  best layer per row selected by Spearman",
    )
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "fig0_summary_forest")


# ---------------------------------------------------------------------------
# FIGURE 1 — supervised dating signal
# ---------------------------------------------------------------------------

def plot_supervised_signal(tables: dict[str, pd.DataFrame]) -> None:
    t1 = tables["t1"]; t2 = tables["t2"]; t3b = tables["t3b"]
    t7 = tables["t7"]; t8 = tables["t8"]

    t1_bal = best_rows(
        t1[(t1["regime"].eq("balanced")) & (t1["year_transform"].eq("raw"))],
        ["model"], "spearman_mean")
    t2_bal = best_rows(
        t2[(t2["regime"].eq("balanced")) & (t2["year_transform"].eq("raw"))],
        ["model"], "spearman_mean")
    t3_bal = best_rows(t3b[t3b["regime"].eq("balanced")], ["model"], "macro_f1")
    t1_bal = with_model_meta(t1_bal)
    t2_bal = with_model_meta(t2_bal)
    t3_bal = with_model_meta(t3_bal)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))

    # ── A — paired PLS vs Ridge, sorted by Ridge ──────────────────────────
    ax = axes[0, 0]
    merge = t1_bal[["model", "spearman_mean", "spearman_std"]].rename(
        columns={"spearman_mean": "pls", "spearman_std": "pls_std"}
    ).merge(
        t2_bal[["model", "spearman_mean", "spearman_std"]].rename(
            columns={"spearman_mean": "ridge", "spearman_std": "ridge_std"}),
        on="model", how="outer",
    )
    merge = with_model_meta(merge).sort_values("ridge", ascending=False).reset_index(drop=True)
    x = np.arange(len(merge))
    shuf = t1["shuffled_spearman_mean"].dropna()
    add_shuffled_band(ax, float(shuf.quantile(0.05)), float(shuf.quantile(0.95)),
                      label="shuffled-label null")
    for i, row in merge.iterrows():
        ax.errorbar(i - 0.14, row["pls"], yerr=row["pls_std"],
                    marker="o", color=row["color"], capsize=3,
                    markeredgecolor="white", markeredgewidth=0.5,
                    markersize=7, linestyle="none")
        ax.errorbar(i + 0.14, row["ridge"], yerr=row["ridge_std"],
                    marker="s", color=row["color"], capsize=3,
                    markeredgecolor="white", markeredgewidth=0.5,
                    markersize=7, linestyle="none")
    meta = t8[t8["regime"].eq("balanced") & t8["year_transform"].eq("raw")].iloc[0]
    ax.axhline(meta["spearman_mean"], color="#777777", linestyle=":", linewidth=1.4,
               label=f"metadata-only ({meta['spearman_mean']:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in merge["model"]], rotation=30, ha="right")
    ax.set_ylim(-0.02, 0.50)
    ax.set_ylabel("Balanced year Spearman")
    ax.set_xlabel("Model  (sorted by Ridge Spearman)")
    set_panel_title(ax,
        "Year-regression Spearman: PLS vs Ridge",
        "balanced · 200 MC draws · year-raw · 5-fold CV · best layer per model")
    handles = [
        plt.Line2D([0], [0], marker="o", color="#555", linestyle="none", label="PLS (best k)"),
        plt.Line2D([0], [0], marker="s", color="#555", linestyle="none", label="Ridge"),
        plt.Line2D([0], [0], color="#777", linestyle=":", label=f"metadata-only ({meta['spearman_mean']:.2f})"),
        Patch(facecolor="#d0d0d0", alpha=0.5, label="shuffled null"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", ncol=2,
              bbox_to_anchor=(0.0, 0.0))
    add_panel_label(ax, "A")

    # ── B — mean-pool vs last-token (slope / dumbbell) ────────────────────
    ax = axes[0, 1]
    mean = best_rows(
        t1[(t1["regime"].eq("balanced")) & (t1["year_transform"].eq("raw"))],
        ["model"], "spearman_mean").set_index("model")
    last = best_rows(
        t1[(t1["regime"].eq("balanced_last")) & (t1["year_transform"].eq("raw"))],
        ["model"], "spearman_mean").set_index("model")
    paired = [m for m in MODEL_ORDER if m in mean.index and m in last.index]
    for model in paired:
        y0, y1 = mean.loc[model, "spearman_mean"], last.loc[model, "spearman_mean"]
        color = MODEL_COLOR[model]
        ax.plot([0, 1], [y0, y1], color=color, linewidth=2.0, alpha=0.9, zorder=2)
        ax.scatter([0], [y0], marker="o", color=color, s=70,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.scatter([1], [y1], marker=MODEL_MARKER[model], color=color, s=70,
                   edgecolor="black" if model == "random" else "white",
                   linewidth=0.6, zorder=3)
        delta = y1 - y0
        ax.text(1.05, y1, f"{MODEL_SHORT[model]}  Δ={delta:+.02f}",
                fontsize=8.5, va="center", color=color, fontweight="bold")
    ax.set_xlim(-0.18, 1.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mean-pool\n(default)", "last-token"])
    ax.set_ylabel("Balanced year Spearman (PLS, best layer)")
    ax.set_xlabel("Activation pooling")
    set_panel_title(ax,
        "Pooling matters: trained models lose 0.10–0.20 Sp at last-token",
        "balanced · 200 MC draws · year-raw · per-model unique color")
    add_panel_label(ax, "B")

    # ── C — ruler Macro-F1 (PLS-DA, balanced) ─────────────────────────────
    ax = axes[1, 0]
    models = [m for m in MODEL_ORDER if m in set(t3_bal["model"])]
    t3_lookup = t3_bal.set_index("model")
    x = np.arange(len(models))
    y = [t3_lookup.loc[m, "macro_f1"] for m in models]
    err = [t3_lookup.loc[m, "macro_f1_std"] for m in models]
    colors = [MODEL_COLOR[m] for m in models]
    ax.bar(x, y, color=colors, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor="black",
                elinewidth=0.9, capsize=3)
    chance = float(t3_lookup["chance_macro_f1"].dropna().iloc[0])
    shuf = float(t3_lookup["shuffled_macro_f1"].dropna().mean())
    ax.axhline(chance, color="black", linestyle="--", linewidth=1.0,
               label=f"chance ({chance:.2f})")
    ax.axhline(shuf, color="#777777", linestyle=":", linewidth=1.2,
               label=f"shuffled labels ({shuf:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], rotation=30, ha="right")
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Model")
    ax.set_ylabel("Balanced ruler Macro-F1")
    set_panel_title(ax,
        "Ruler identification — surface beats neural",
        "balanced · 8 rulers × 21 frags · 200 MC draws · PLS-DA · 5-fold CV")
    ax.legend(frameon=False, loc="upper right")
    add_panel_label(ax, "C")

    # ── D — name masking 2×2 grouped bars ─────────────────────────────────
    ax = axes[1, 1]
    cleanings = ["tier0", "maximal"]
    tasks = [("year_spearman", "Year Spearman", "#1f6fb4"),
             ("ruler_macro_f1", "Ruler Macro-F1", "#c2571a")]
    conditions = ["unmasked", "masked"]
    group_w = 0.38
    bar_w = 0.16
    # x positions: 2 cleaning blocks × 2 tasks × 2 conditions
    n_groups = 4  # (tier0,year),(tier0,ruler),(maximal,year),(maximal,ruler)
    centers = np.arange(n_groups) * 1.2
    xticks, xticklabels = [], []
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="unmasked"),
        Patch(facecolor="white", edgecolor="black", hatch="////", label="masked (names removed)"),
    ]
    gi = 0
    for cleaning in cleanings:
        for metric_key, task_label, task_color in tasks:
            cx = centers[gi]
            xticks.append(cx)
            xticklabels.append(f"{cleaning}\n{task_label}")
            for k, cond in enumerate(conditions):
                row = t7[(t7["cleaning"].eq(cleaning)) & (t7["condition"].eq(cond))].iloc[0]
                val = row[f"{metric_key}_mean"]
                err = row[f"{metric_key}_std"]
                xb = cx + (k - 0.5) * bar_w * 1.05
                bar = ax.bar(xb, val, width=bar_w, color=task_color,
                             edgecolor="black", linewidth=0.6,
                             hatch="////" if cond == "masked" else None)
                ax.errorbar(xb, val, yerr=err, fmt="none", ecolor="black",
                            elinewidth=0.7, capsize=2)
            # delta annotation
            row_u = t7[(t7["cleaning"].eq(cleaning)) & (t7["condition"].eq("unmasked"))].iloc[0]
            row_m = t7[(t7["cleaning"].eq(cleaning)) & (t7["condition"].eq("masked"))].iloc[0]
            delta = row_m[f"{metric_key}_mean"] - row_u[f"{metric_key}_mean"]
            top = max(row_u[f"{metric_key}_mean"], row_m[f"{metric_key}_mean"]) + 0.04
            color = "#1a8d3a" if abs(delta) < 0.03 else "#b23b3b"
            ax.text(cx, top, f"Δ={delta:+.02f}",
                    ha="center", fontsize=9, color=color, fontweight="bold")
            gi += 1
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Score (mean over 200 MC draws)")
    ax.set_xlabel("Text cleaning  ·  task")
    set_panel_title(ax,
        "Name-masking confound check — dating survives, ruler-ID drops",
        "TF-IDF probe · 200 MC draws · mask m-/f- + theophoric d-…names · "
        "Δ = masked − unmasked")
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    add_panel_label(ax, "D")

    fig.tight_layout()
    save_figure(fig, "fig1_supervised_signal")


# ---------------------------------------------------------------------------
# FIGURE 2 — model size scaling
# ---------------------------------------------------------------------------

def plot_model_size_scaling(tables: dict[str, pd.DataFrame]) -> None:
    t1 = tables["t1"]; t2 = tables["t2"]
    t4 = tables["t4"].rename(columns={"method": "model"})
    t8 = tables["t8"]

    panels = []
    t1_bal = best_rows(
        t1[(t1["regime"].eq("balanced")) & (t1["year_transform"].eq("raw"))],
        ["model"], "spearman_mean")
    panels.append(("Year-PLS Spearman", "spearman_mean", "spearman_std", t1_bal,
                   "Balanced year Spearman"))

    t2_bal = best_rows(
        t2[(t2["regime"].eq("balanced")) & (t2["year_transform"].eq("raw"))],
        ["model"], "spearman_mean")
    panels.append(("Year-Ridge Spearman", "spearman_mean", "spearman_std", t2_bal,
                   "Balanced year Spearman"))

    t4_bal = best_rows(t4[t4["regime"].eq("balanced")], ["model"], "isomap_pairwise_acc")
    panels.append(("Isomap pairwise-order acc", "isomap_pairwise_acc",
                   "isomap_pairwise_acc_std", t4_bal,
                   "Balanced Isomap pacc"))

    meta = t8[t8["regime"].eq("balanced") & t8["year_transform"].eq("raw")].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharex=True)
    zero_offsets = {"metadata": -0.12, "tfidf": 0.12}

    # log-params x for stat test
    def _stat_text(df: pd.DataFrame, col: str) -> str:
        only_neural = df[df["model"].isin(
            ["mlm", "thalesian_akk300m", "thalesian_cunei400m",
             "qwen3_1b7", "qwen3_8b", "qwen3_32b"])].dropna(subset=[col])
        if len(only_neural) < 3:
            return ""
        from scipy.stats import spearmanr  # type: ignore
        try:
            x = np.log10(only_neural["params_b"].astype(float))
            r, p = spearmanr(x, only_neural[col])
            return f"Sp(log-params, score) = {r:+.2f}  (p={p:.2f}, n={len(only_neural)})"
        except Exception:
            return ""

    for panel_idx, (ax, (title, metric, std_col, df, ylab)) in enumerate(zip(axes, panels)):
        df = with_model_meta(df)

        # add metadata baseline for year panels
        if metric == "spearman_mean":
            metadata_row = {
                "model": "metadata", "params_b": 0.0, "family": "metadata",
                "model_label": "Metadata-only",
                "color": MODEL_COLOR["metadata"], "marker": MODEL_MARKER["metadata"],
                metric: meta["spearman_mean"], std_col: meta["spearman_std"],
            }
            df = pd.concat([df, pd.DataFrame([metadata_row])], ignore_index=True)

        # Qwen3 connecting trend line
        qwen = df[df["model"].isin(["qwen3_1b7", "qwen3_8b", "qwen3_32b"])].copy()
        if not qwen.empty:
            qwen["x"] = qwen["model"].map(lambda m: model_x(m))
            qwen = qwen.sort_values("params_b")
            ax.plot(qwen["x"], qwen[metric], color="#4b8f3a",
                    linewidth=1.2, alpha=0.45, zorder=2,
                    label="Qwen3 1.7B→8B→32B")

        # Per-model points. Stagger labels above/below to reduce overlap.
        # If two Thalesian points overlap (close params), nudge them up/down.
        label_y_off = {
            "metadata": -0.030,
            "tfidf": +0.025,
            "mlm": +0.025,
            "thalesian_akk300m": -0.030,
            "thalesian_cunei400m": +0.030,
            "qwen3_1b7": -0.030,
            "qwen3_8b": +0.030,
            "qwen3_32b": +0.025,
            "random": +0.030,
        }
        for _, row in df.sort_values("params_b").iterrows():
            model = row["model"]
            x = model_x(model, zero_offsets)
            y = row[metric]
            yerr = row.get(std_col, np.nan)
            ax.errorbar(x, y,
                        yerr=None if pd.isna(yerr) else yerr,
                        marker=MODEL_MARKER[model],
                        color=MODEL_COLOR[model],
                        markeredgecolor="black" if model == "random" else "white",
                        markeredgewidth=0.6,
                        markersize=8.5, capsize=3,
                        linestyle="none", zorder=3)
            dy = label_y_off.get(model, 0.025)
            va = "bottom" if dy >= 0 else "top"
            ax.text(x, y + dy, MODEL_SHORT[model], fontsize=7.8,
                    ha="center", va=va, color=MODEL_COLOR[model],
                    fontweight="bold")

        # reference lines
        if metric == "isomap_pairwise_acc":
            ax.axhline(0.5, color="black", linewidth=0.8, label="chance (0.50)")
            ax.axhline(0.70, color="#b23b3b", linestyle=":", linewidth=1.2,
                       label="pre-committed gate (0.70)")
            ax.set_ylim(0.45, 0.82)
        else:
            ax.axhline(0.0, color="black", linewidth=0.8)
            shuf = t1["shuffled_spearman_mean"].dropna()
            add_shuffled_band(ax, float(shuf.quantile(0.05)), float(shuf.quantile(0.95)),
                              label="shuffled null")
            ax.set_ylim(-0.05, 0.50)

        # stat annotation
        stat = _stat_text(df, metric)
        if stat:
            ax.text(0.02, 0.97, stat, transform=ax.transAxes,
                    fontsize=8, va="top", ha="left", color="#333",
                    bbox=dict(facecolor="white", edgecolor="#cccccc",
                              boxstyle="round,pad=0.25"))

        set_panel_title(ax, title,
                        "balanced · best layer per model · error bars = ±1 SD over 200 draws")
        ax.set_ylabel(ylab)
        ax.set_xlabel("Model parameters (log scale)  ·  0-param baselines at left")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        add_panel_label(ax, chr(ord("A") + panel_idx))

    # Collapse 300M/400M into one tick to avoid label overlap on log scale
    ticks = [-1.75, np.log10(0.0367),
             (np.log10(0.300) + np.log10(0.400)) / 2,
             np.log10(1.7), np.log10(8.0), np.log10(32.0)]
    ticklabels = ["0 params", "37M", "300M/400M", "1.7B", "8B", "32B"]
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, rotation=30, ha="right")
        ax.set_xlim(-2.1, np.log10(32.0) + 0.45)

    fig.suptitle("Model scale does not drive the chronology signal",
                 y=1.02, fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig2_model_size_scaling")


# ---------------------------------------------------------------------------
# FIGURE 3 — geometry & LORO controls
# ---------------------------------------------------------------------------

def plot_geometry_controls(tables: dict[str, pd.DataFrame]) -> None:
    t4 = tables["t4"].rename(columns={"method": "model"})
    t5 = tables["t5"].rename(columns={"method": "model"})
    t2 = tables["t2"]

    t4_bal = best_rows(t4[t4["regime"].eq("balanced")], ["model"], "isomap_pairwise_acc")
    t4_bal = with_model_meta(t4_bal)
    t2_bal = best_rows(
        t2[(t2["regime"].eq("balanced")) & (t2["year_transform"].eq("raw"))],
        ["model"], "spearman_mean")
    t2_bal = with_model_meta(t2_bal)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6),
                             gridspec_kw={"width_ratios": [1.0, 1.2, 1.3]})

    # ── A — bars of Isomap pacc ──────────────────────────────────────────
    ax = axes[0]
    models = [m for m in MODEL_ORDER if m in set(t4_bal["model"])]
    lookup = t4_bal.set_index("model")
    x = np.arange(len(models))
    y = [lookup.loc[m, "isomap_pairwise_acc"] for m in models]
    err = [lookup.loc[m, "isomap_pairwise_acc_std"] for m in models]
    colors = [MODEL_COLOR[m] for m in models]
    ax.bar(x, y, color=colors, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor="black",
                elinewidth=0.8, capsize=3)
    ax.axhline(0.5, color="black", linewidth=0.8, label="chance (0.50)")
    ax.axhline(0.70, color="#b23b3b", linestyle=":", linewidth=1.3,
               label="pre-committed gate (0.70)")
    ax.set_ylim(0.45, 0.82)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], rotation=30, ha="right")
    ax.set_ylabel("Isomap pairwise-order accuracy")
    ax.set_xlabel("Model  (best layer & config per model)")
    set_panel_title(ax,
        "Unsupervised chronological geometry",
        "balanced · 200 MC draws · year never shown to Isomap")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "A")

    # ── B — LORO forest plot ─────────────────────────────────────────────
    ax = axes[1]
    rows = t5.sort_values(["regime", "model", "cleaning"]).reset_index(drop=True)
    y = np.arange(len(rows))
    for yi, r in zip(y, rows.itertuples()):
        col = MODEL_COLOR[r.model]
        cfg = f"{r.cleaning}/L{layer_int(r.layer):02d}"
        ax.errorbar(
            r.drop, yi,
            xerr=None if pd.isna(r.drop_std) else r.drop_std,
            marker=MODEL_MARKER[r.model], color=col,
            markersize=9, capsize=3,
            markeredgecolor="white", markeredgewidth=0.6,
            linestyle="none", elinewidth=1.1,
        )
        ax.text(0.12, yi, f"{MODEL_SHORT[r.model]} · {r.regime} · {cfg}",
                fontsize=8.5, va="center", color=col)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.10, color="#b23b3b", linestyle="--", linewidth=1.3,
               label="pre-committed gate (drop=0.10)")
    ax.set_yticks([])
    ax.set_xlim(-0.06, 0.34)
    ax.invert_yaxis()
    ax.set_xlabel("LORO pacc drop  (pacc_full − pacc_loro)  ·  smaller = real temporal axis")
    set_panel_title(ax,
        "Timeline survives held-out rulers (drops < 0.06 < 0.10 gate)",
        "Leave-One-Ruler-Out · 200 MC draws (balanced rows) · "
        "drop > gate would mean ruler-cluster confound")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "B")

    # ── C — supervised vs unsupervised scatter ──────────────────────────
    ax = axes[2]
    merged = t2_bal.merge(
        t4_bal[["model", "isomap_pairwise_acc", "isomap_pairwise_acc_std"]],
        on="model", how="inner", suffixes=("_ridge", "_geo"))
    # OLS line through points (no regression libs assumed)
    if len(merged) >= 3:
        xv = merged["spearman_mean"].values
        yv = merged["isomap_pairwise_acc"].values
        slope, intercept = np.polyfit(xv, yv, 1)
        xline = np.linspace(xv.min() - 0.01, xv.max() + 0.01, 50)
        ax.plot(xline, slope * xline + intercept,
                color="#999", linestyle="--", linewidth=1.0,
                label=f"OLS fit (slope={slope:+.2f})")
    # Custom label-offset directions per model so nothing overlaps the markers
    label_off = {
        "thalesian_akk300m":   ( +0.006, -0.008, "left", "top"),
        "thalesian_cunei400m": ( +0.006, +0.007, "left", "bottom"),
        "qwen3_1b7":           ( -0.006, -0.008, "right", "top"),
        "qwen3_8b":            ( +0.006, -0.008, "left", "top"),
        "qwen3_32b":           ( +0.006, +0.007, "left", "bottom"),
        "random":              ( -0.006, +0.007, "right", "bottom"),
    }
    for _, row in merged.iterrows():
        model = row["model"]
        ax.errorbar(
            row["spearman_mean"], row["isomap_pairwise_acc"],
            xerr=row["spearman_std"] if not pd.isna(row["spearman_std"]) else None,
            yerr=row["isomap_pairwise_acc_std"] if not pd.isna(row["isomap_pairwise_acc_std"]) else None,
            marker=MODEL_MARKER[model], color=MODEL_COLOR[model],
            markeredgecolor="black" if model == "random" else "white",
            markeredgewidth=0.6, capsize=3,
            linestyle="none", markersize=9,
        )
        dx, dy, ha, va = label_off.get(model, (+0.006, +0.005, "left", "bottom"))
        ax.text(row["spearman_mean"] + dx,
                row["isomap_pairwise_acc"] + dy,
                MODEL_SHORT[model], fontsize=8.5,
                ha=ha, va=va,
                color=MODEL_COLOR[model], fontweight="bold")
    # callout for random — anchored in the empty top-left quadrant
    if "random" in merged["model"].values:
        r = merged[merged["model"].eq("random")].iloc[0]
        ax.annotate(
            "Random-init lands at the\ngeometric ceiling without\nany learned chronology",
            xy=(r["spearman_mean"] - 0.005, r["isomap_pairwise_acc"] + 0.004),
            xytext=(0.260, 0.805),
            fontsize=8.0, color="#333",
            ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="#bbb",
                      boxstyle="round,pad=0.3", linewidth=0.6),
            arrowprops=dict(arrowstyle="->", color="#888", linewidth=0.8,
                            connectionstyle="arc3,rad=-0.25"),
        )
    ax.set_xlabel("Balanced Ridge year Spearman  (supervised)")
    ax.set_ylabel("Balanced Isomap pacc  (unsupervised)")
    ax.set_xlim(0.25, 0.46)
    ax.set_ylim(0.64, 0.82)
    set_panel_title(ax,
        "Supervised dating vs unsupervised geometry",
        "balanced · best layer per model · 200 MC draws · same model in both axes")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "C")

    fig.tight_layout()
    save_figure(fig, "fig3_geometry_controls")


# ---------------------------------------------------------------------------
# FIGURE 4 — layerwise depth
# ---------------------------------------------------------------------------

def plot_layerwise_depth(tables: dict[str, pd.DataFrame]) -> None:
    t1 = tables["t1"].copy()
    t4 = tables["t4"].rename(columns={"method": "model"}).copy()

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4))

    def _per_layer(df: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame]:
        df = df.copy()
        df["layer_num"] = df["layer"].map(layer_int)
        df = best_rows(df, ["model", "layer_num"], metric)
        out: dict[str, pd.DataFrame] = {}
        for model, sub in df.groupby("model"):
            sub = sub.sort_values("layer_num")
            max_layer = max(sub["layer_num"].max(), 1)
            sub = sub.assign(depth=sub["layer_num"] / max_layer)
            out[model] = sub
        return out

    # ── A — year PLS by depth ────────────────────────────────────────────
    ax = axes[0]
    df = t1[
        t1["regime"].eq("balanced")
        & t1["year_transform"].eq("raw")
        & t1["cleaning"].eq("tier0")
        & t1["pool"].eq("mean")
    ]
    per_model = _per_layer(df, "spearman_mean")
    selected = ["mlm", "thalesian_akk300m", "thalesian_cunei400m",
                "qwen3_1b7", "qwen3_8b", "qwen3_32b"]
    shuf = t1["shuffled_spearman_mean"].dropna()
    add_shuffled_band(ax, float(shuf.quantile(0.05)), float(shuf.quantile(0.95)),
                      label="shuffled null")
    for model in selected:
        if model not in per_model:
            continue
        sub = per_model[model]
        x = sub["depth"].values
        y = sub["spearman_mean"].values
        ys = smooth(y, 3)
        ls = FAMILY_LINESTYLE[FAMILY[model]]
        ax.plot(x, y, color=MODEL_COLOR[model], alpha=0.25, linewidth=1.0, zorder=2)
        ax.plot(x, ys, color=MODEL_COLOR[model], linewidth=2.0,
                linestyle=ls, alpha=0.95, label=MODEL_SHORT[model], zorder=3)
        # mark the best layer
        best = sub.iloc[int(np.argmax(y))]
        ax.scatter([best["depth"]], [best["spearman_mean"]],
                   marker="*", color=MODEL_COLOR[model], s=130,
                   edgecolor="black", linewidth=0.6, zorder=4)
        ax.text(best["depth"] + 0.012, best["spearman_mean"] + 0.005,
                f"L{int(best['layer_num'])}",
                fontsize=7.5, color=MODEL_COLOR[model])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized layer depth  (0 = embedding, 1 = final)")
    ax.set_ylabel("Year PLS Spearman  (balanced)")
    ax.set_ylim(-0.05, 0.50)
    set_panel_title(ax,
        "Supervised year signal by layer  (★ = best layer)",
        "balanced · tier0 · mean-pool · 200 MC draws · raw + 3-layer smoothing")
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=8.5)
    add_panel_label(ax, "A")

    # ── B — Isomap pacc by depth ─────────────────────────────────────────
    ax = axes[1]
    df = t4[
        t4["regime"].eq("balanced")
        & t4["cleaning"].eq("maximal")
        & t4["pool"].eq("mean")
    ]
    per_model = _per_layer(df, "isomap_pairwise_acc")
    selected = ["random", "thalesian_akk300m", "thalesian_cunei400m",
                "qwen3_1b7", "qwen3_8b", "qwen3_32b"]
    for model in selected:
        if model not in per_model:
            continue
        sub = per_model[model]
        x = sub["depth"].values
        y = sub["isomap_pairwise_acc"].values
        ys = smooth(y, 3)
        ls = FAMILY_LINESTYLE[FAMILY[model]]
        ax.plot(x, y, color=MODEL_COLOR[model], alpha=0.25, linewidth=1.0, zorder=2)
        ax.plot(x, ys, color=MODEL_COLOR[model], linewidth=2.0,
                linestyle=ls, alpha=0.95, label=MODEL_SHORT[model], zorder=3)
        best = sub.iloc[int(np.argmax(y))]
        ax.scatter([best["depth"]], [best["isomap_pairwise_acc"]],
                   marker="*", color=MODEL_COLOR[model], s=130,
                   edgecolor="black", linewidth=0.6, zorder=4)
        ax.text(best["depth"] + 0.012, best["isomap_pairwise_acc"] + 0.005,
                f"L{int(best['layer_num'])}",
                fontsize=7.5, color=MODEL_COLOR[model])
    ax.axhline(0.5, color="black", linewidth=0.8, label="chance (0.50)")
    ax.axhline(0.70, color="#b23b3b", linestyle=":", linewidth=1.2,
               label="pre-committed gate (0.70)")
    ax.set_xlabel("Normalized layer depth  (0 = embedding, 1 = final)")
    ax.set_ylabel("Isomap pairwise-order accuracy")
    ax.set_ylim(0.55, 0.80)
    set_panel_title(ax,
        "Geometry peaks early  (within first 10–20% of depth)",
        "balanced · maximal · mean-pool · 200 MC draws · raw + 3-layer smoothing")
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=8.5)
    add_panel_label(ax, "B")

    fig.tight_layout()
    save_figure(fig, "fig4_layerwise_depth")


# ---------------------------------------------------------------------------
# FIGURE 5 — elicitation + prompt sensitivity
# ---------------------------------------------------------------------------

def plot_prompt_and_elicitation(tables: dict[str, pd.DataFrame]) -> None:
    t9 = tables["t9"]
    t10 = tables["t10"].dropna(subset=["headline_value"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))

    # ── A — elicitation, with explicit N and 0.0 labels ──────────────────
    ax = axes[0]
    models = ["qwen3_1b7", "qwen3_8b", "qwen3_32b"]
    variants = ["kp0", "kp1", "kp2"]
    palette = {"kp0": "#4b8f3a", "kp1": "#1f6fb4", "kp2": "#b23b3b"}
    x = np.arange(len(models))
    width = 0.25
    for j, variant in enumerate(variants):
        sub = t9[t9["variant"].eq(variant)].set_index("model")
        for i, m in enumerate(models):
            row = sub.loc[m]
            val = row["headline_value"]
            n_score = int(row["n_scoreable"])
            n_total = int(row["n_total"])
            xb = x[i] + (j - 1) * width
            bar = ax.bar(xb, max(val, 0.005),  # tiny visible stub when 0
                         width=width, color=palette[variant],
                         edgecolor="white", linewidth=0.6,
                         label=KP_LABEL[variant] if i == 0 else None)
            # annotation: value, n_scoreable / n_total
            if val == 0:
                ax.text(xb, 0.05, "0.00\nPASS",
                        ha="center", fontsize=8, fontweight="bold",
                        color=palette[variant])
            else:
                ax.text(xb, val + 0.025, f"{val:.2f}",
                        ha="center", fontsize=8, fontweight="bold",
                        color=palette[variant])
            ax.text(xb, -0.06, f"n={n_score}/{n_total}",
                    ha="center", fontsize=7, color="#555")

    ax.axhline(0.30, color="#b23b3b", linestyle="--", linewidth=1.0,
               label="hallucination gate (0.30)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models])
    ax.set_ylim(-0.10, 1.05)
    ax.set_xlabel("Model  (n_scoreable / n_total below each bar)")
    ax.set_ylabel("Score")
    set_panel_title(ax,
        "Explicit elicitation — small-N, mixed signal",
        "8 questions per probe · base/instruct chat mode · "
        "labels: kp0=date recall, kp1=king recall, kp2=hallucination")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    add_panel_label(ax, "A")

    # ── B — prompt-version sensitivity for qwen3_32b ─────────────────────
    ax = axes[1]
    sub = t10[(t10["model"].eq("qwen3_32b")) & (t10["task"].eq("year"))].copy()
    best = best_rows(sub, ["variant"], "headline_value").sort_values("variant")
    x = np.arange(len(best))
    base_val = float(best[best["variant"].eq("pv0")]["headline_value"].iloc[0])
    ax.axhspan(base_val - 0.05, base_val + 0.05,
               color="#d0d0d0", alpha=0.45, label="pv0 ± 0.05 band")
    ax.errorbar(x, best["headline_value"], yerr=best["std"],
                color=MODEL_COLOR["qwen3_32b"], marker="o",
                markersize=9, capsize=4, linewidth=2.0,
                markeredgecolor="white", markeredgewidth=0.6)
    # Show the selected layer/pool per variant as a small annotation just
    # above each marker, with a white box so it doesn't sit on the error bar.
    for i, row in enumerate(best.itertuples()):
        cfg = f"{row.pool}-pool · L{layer_int(row.layer):02d}"
        ax.annotate(
            cfg,
            xy=(i, row.headline_value),
            xytext=(0, 16), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5, color="#333",
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.18", linewidth=0.5),
        )
    spread = best["headline_value"].max() - best["headline_value"].min()
    ax.text(0.97, 0.04,
            f"prompt-induced spread = {spread:.03f}\n"
            f"(qwen3_32b only — pv1–pv3 walltimed for 1B7/8B)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="#bbbbbb",
                      boxstyle="round,pad=0.3"))
    ax.set_xticks(x)
    ax.set_xticklabels([PV_LABEL[v] for v in best["variant"]], fontsize=8)
    ax.set_ylim(0.20, 0.62)
    ax.set_xlabel("Prompt variant fed to model before extracting activations")
    ax.set_ylabel("Prompted year-PLS Spearman  (best layer per variant)")
    set_panel_title(ax,
        "Prompt wording barely moves the latent dating signal",
        "qwen3_32b · year-raw · 5-fold CV · 200 MC draws · per-variant best layer")
    ax.legend(frameon=False, loc="upper left")
    add_panel_label(ax, "B")

    fig.tight_layout()
    save_figure(fig, "fig5_prompt_elicitation")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = read_tables()
    build_headlines(tables)
    plot_summary_forest(tables)
    plot_supervised_signal(tables)
    plot_model_size_scaling(tables)
    plot_geometry_controls(tables)
    plot_layerwise_depth(tables)
    plot_prompt_and_elicitation(tables)
    print(f"Wrote figures to {OUT_DIR}")
    print(f"Wrote headlines table to {TABLE_DIR / 'T_headlines.csv'}")


if __name__ == "__main__":
    main()
