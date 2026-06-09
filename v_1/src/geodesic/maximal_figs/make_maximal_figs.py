#!/usr/bin/env python3
"""Recreate fig1 (A,C,D), fig2 (A,B) and fig4 (A) for the MAXIMAL config.

Config recreated: cleaning=maximal · pooling=mean · regime=balanced (200 MC
draws) · year-PLS / year-Ridge. Reads the maximal balanced-MC summaries written
by M1 (`probes/<model>_<probe>__mc_balanced_maximal__summary.json`) plus the
already-existing maximal rows of `results/tables/T7_name_masking.csv` (panel D).

ISOLATED: imports only the styling registries + a few helpers from
`plot_round3_story_figures.py`; writes its own tables/figures under this folder.
Nothing in the canonical Round-3 pipeline is touched.

Usage:
    python make_maximal_figs.py                 # uses ./probes, writes ./tables, ./figures
    python make_maximal_figs.py --probes-dir DIR --tag mc_balanced_maximal
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GEO = HERE.parent                       # .../src/geodesic
sys.path.insert(0, str(GEO))
import plot_round3_story_figures as prs  # noqa: E402  styling registries + helpers

import matplotlib.pyplot as plt  # noqa: E402  (prs already set Agg)

CANON_TABLES = GEO / "results" / "tables"
CLEANING, POOL = "maximal", "mean"
_LKEY = re.compile(r"__L(\d+)__")

# models that have layers (for the layerwise fig4); tfidf has none.
LAYERED = ["mlm", "thalesian_akk300m", "thalesian_cunei400m",
           "qwen3_1b7", "qwen3_8b", "qwen3_32b", "random"]
ALL_MODELS = ["tfidf"] + LAYERED


# --------------------------------------------------------------------------- #
# summary readers
# --------------------------------------------------------------------------- #
def _summary(probes: Path, model: str, probe: str, tag: str) -> dict | None:
    p = probes / f"{model}_{probe}__{tag}__summary.json"
    if not p.exists():
        print(f"[skip] missing {p.name}")
        return None
    return json.load(open(p)).get("per_config", {})


def _per_layer(per_config: dict, suffix: str, value_key: str) -> dict[int, tuple]:
    """{layer_int -> (mean, std)} for configs matching maximal/mean/<suffix>."""
    out: dict[int, tuple] = {}
    for key, rec in per_config.items():
        if f"__{CLEANING}__{POOL}__" not in key or not key.endswith(suffix):
            continue
        m = _LKEY.search(key)
        if not m:
            continue
        L = int(m.group(1))
        out[L] = (rec.get(f"{value_key}_mean"), rec.get(f"{value_key}_std"))
    return out


def pls_year(probes, model, tag):       # {L -> (sp_mean, sp_std)}
    pc = _summary(probes, model, "pls", tag)
    return _per_layer(pc, "year-raw", "spearman") if pc else {}


def ridge_year(probes, model, tag):
    pc = _summary(probes, model, "cls_numeric", tag)
    return _per_layer(pc, "year-raw", "spearman") if pc else {}


def ruler_macro_f1(probes, model, tag):
    """Best balanced ruler Macro-F1 over CLS (logistic) and PLS-DA, any layer."""
    best = (None, None, None)           # (f1, std, src)
    for probe, src in (("cls", "cls"), ("pls", "plsda")):
        pc = _summary(probes, model, probe, tag)
        if not pc:
            continue
        pl = _per_layer(pc, "ruler", "macro_f1")
        for L, (mean, std) in pl.items():
            if mean is not None and (best[0] is None or mean > best[0]):
                best = (mean, std, src)
    return best


def _best(per_layer: dict[int, tuple]) -> tuple:
    """(layer, mean, std) of the highest-mean layer; (None,None,None) if empty."""
    items = [(L, m, s) for L, (m, s) in per_layer.items() if m is not None]
    if not items:
        return (None, None, None)
    L, m, s = max(items, key=lambda t: t[1])
    return (L, m, s)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def build_tables(probes: Path, tag: str, tables_out: Path) -> dict:
    tables_out.mkdir(parents=True, exist_ok=True)
    rows_pls, rows_ridge, rows_ruler = [], [], []
    data = {"pls": {}, "ridge": {}, "ruler": {}}
    for m in ALL_MODELS:
        pls = pls_year(probes, m, tag)
        rid = ridge_year(probes, m, tag)
        data["pls"][m] = pls
        data["ridge"][m] = rid
        data["ruler"][m] = ruler_macro_f1(probes, m, tag)
        for L, (mean, std) in sorted(pls.items()):
            rows_pls.append([m, L, mean, std])
        for L, (mean, std) in sorted(rid.items()):
            rows_ridge.append([m, L, mean, std])
        f1, fstd, src = data["ruler"][m]
        rows_ruler.append([m, f1, fstd, src])
    pd.DataFrame(rows_pls, columns=["model", "layer", "spearman_mean", "spearman_std"]
                 ).to_csv(tables_out / "T1_year_pls_maximal.csv", index=False)
    pd.DataFrame(rows_ridge, columns=["model", "layer", "spearman_mean", "spearman_std"]
                 ).to_csv(tables_out / "T2_year_ridge_maximal.csv", index=False)
    pd.DataFrame(rows_ruler, columns=["model", "macro_f1", "macro_f1_std", "source"]
                 ).to_csv(tables_out / "T3_ruler_maximal.csv", index=False)
    print(f"[tables] wrote 3 CSVs to {tables_out}")
    return data


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _models_present(d: dict) -> list[str]:
    return [m for m in prs.MODEL_ORDER if m in d and d[m]]


def fig1_ACD(data: dict, fig_out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # A — year PLS (best layer) vs Ridge (best layer), per model
    ax = axes[0]
    models = [m for m in prs.MODEL_ORDER
              if m in data["pls"] and (data["pls"][m] or data["ridge"].get(m))]
    xs = np.arange(len(models))
    for i, m in enumerate(models):
        Lp, mp, sp = _best(data["pls"].get(m, {}))
        Lr, mr, sr = _best(data["ridge"].get(m, {}))
        c = prs.MODEL_COLOR[m]
        if mp is not None:
            ax.errorbar(i - 0.10, mp, yerr=sp or 0, fmt="o", color=c, ms=8,
                        capsize=3, label="PLS (best k)" if i == 0 else None)
            ax.annotate(f"L{Lp}", (i - 0.10, mp), fontsize=6, ha="center", va="bottom")
        if mr is not None:
            ax.errorbar(i + 0.10, mr, yerr=sr or 0, fmt="s", color=c, ms=7,
                        capsize=3, alpha=0.7, label="Ridge" if i == 0 else None)
    ax.axhline(0.20, ls=":", color="grey", lw=1, label="metadata-only (~0.20)")
    ax.set_xticks(xs)
    ax.set_xticklabels([prs.MODEL_SHORT[m] for m in models], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Balanced year Spearman")
    prs.set_panel_title(ax, "Year-regression Spearman: PLS vs Ridge",
                        "balanced · maximal · mean · 200 MC draws · best layer/k")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    prs.add_panel_label(ax, "A")

    # C — ruler identification (surface vs neural), best balanced Macro-F1
    ax = axes[1]
    models = [m for m in prs.MODEL_ORDER if data["ruler"].get(m, (None,))[0] is not None]
    xs = np.arange(len(models))
    vals = [data["ruler"][m][0] for m in models]
    errs = [data["ruler"][m][1] or 0 for m in models]
    ax.bar(xs, vals, yerr=errs, capsize=3,
           color=[prs.MODEL_COLOR[m] for m in models])
    ax.axhline(0.125, ls="--", color="black", lw=1, label="chance (1/8)")
    ax.set_xticks(xs)
    ax.set_xticklabels([prs.MODEL_SHORT[m] for m in models], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Balanced ruler Macro-F1")
    prs.set_panel_title(ax, "Ruler identification (best of CLS / PLS-DA)",
                        "balanced · maximal · mean · 8 rulers × 21 · 200 MC draws")
    ax.legend(frameon=False, fontsize=8)
    prs.add_panel_label(ax, "C")

    # D — name-masking confound (TF-IDF), maximal rows from canonical T7
    ax = axes[2]
    t7 = CANON_TABLES / "T7_name_masking.csv"
    if t7.exists():
        df = pd.read_csv(t7)
        df = df[df["cleaning"].astype(str).eq("maximal")]
        conds = ["unmasked", "masked"]
        yr = [df[df["condition"].eq(c)]["year_spearman_mean"].mean() for c in conds]
        rl = [df[df["condition"].eq(c)]["ruler_macro_f1_mean"].mean() for c in conds]
        w = 0.35
        x = np.arange(2)
        ax.bar(x - w/2, yr, w, label="Year Spearman", color="#1f6fb4")
        ax.bar(x + w/2, rl, w, label="Ruler Macro-F1", color="#8b3a0e")
        ax.set_xticks(x)
        ax.set_xticklabels(["unmasked", "masked\n(names removed)"], fontsize=8)
        ax.set_ylabel("Score (mean over 200 draws)")
        prs.set_panel_title(ax, "Name-masking confound check",
                            "TF-IDF · balanced · maximal · year vs ruler")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "T7_name_masking.csv not found", ha="center")
    prs.add_panel_label(ax, "D")

    fig.tight_layout()
    fig_out.mkdir(parents=True, exist_ok=True)
    out = fig_out / "fig1_maximal_ACD.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {out}")


def fig2_AB(data: dict, fig_out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, (title, key, label) in zip(
            axes,
            [("Year-PLS Spearman", "pls", "A"),
             ("Year-Ridge Spearman", "ridge", "B")]):
        models = [m for m in prs.MODEL_ORDER
                  if m in data[key] and _best(data[key][m])[1] is not None]
        xs, ys, es, cs = [], [], [], []
        for m in models:
            L, mean, std = _best(data[key][m])
            px = prs.PARAMS_B[m]
            xs.append(px if px > 0 else 0.01)      # 0-param baselines pinned left
            ys.append(mean); es.append(std or 0); cs.append(prs.MODEL_COLOR[m])
            ax.annotate(prs.MODEL_SHORT[m], (xs[-1], mean), fontsize=7,
                        ha="left", va="bottom")
        if not xs:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            prs.set_panel_title(ax, title, "balanced · maximal · mean · best layer · 200 MC draws")
            prs.add_panel_label(ax, label)
            continue
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="grey", alpha=0.5, capsize=2, zorder=1)
        ax.scatter(xs, ys, c=cs, s=70, zorder=2, edgecolor="black", linewidth=0.4)
        ax.set_xscale("log")
        ax.axhline(0.20, ls=":", color="grey", lw=1)
        ax.set_xlabel("Model parameters (B, log scale) · 0-param baselines at left")
        prs.set_panel_title(ax, title, "balanced · maximal · mean · best layer · 200 MC draws")
        prs.add_panel_label(ax, label)
    axes[0].set_ylabel("Balanced year Spearman")
    fig.suptitle("Model scale does not drive the chronology signal (maximal)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = fig_out / "fig2_maximal_AB.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {out}")


def fig4_A(data: dict, fig_out: Path) -> None:
    """Layerwise year-PLS, RAW layer index on x (not normalized depth)."""
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for m in LAYERED:
        pl = data["pls"].get(m, {})
        if not pl:
            continue
        Ls = sorted(pl)
        y = [pl[L][0] for L in Ls]
        c = prs.MODEL_COLOR[m]
        ls = prs.FAMILY_LINESTYLE.get(prs.FAMILY[m], "-")
        ax.plot(Ls, y, color=c, lw=2.0, ls=ls, alpha=0.9, label=prs.MODEL_SHORT[m])
        bi = int(np.nanargmax(y))
        ax.scatter([Ls[bi]], [y[bi]], marker="*", s=130, color=c,
                   edgecolor="black", linewidth=0.6, zorder=4)
        ax.annotate(f"L{Ls[bi]}", (Ls[bi], y[bi]), fontsize=7, color=c)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Layer index  (raw: 0 = embedding … max = final)")
    ax.set_ylabel("Year PLS Spearman  (balanced)")
    prs.set_panel_title(ax, "Supervised year signal by layer  (★ = best layer)",
                        "balanced · maximal · mean-pool · 200 MC draws · raw layer index")
    ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="best")
    prs.add_panel_label(ax, "A")
    fig.tight_layout()
    out = fig_out / "fig4_maximal_A.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-dir", type=Path, default=HERE / "probes")
    ap.add_argument("--tag", default="mc_balanced_maximal")
    ap.add_argument("--tables-out", type=Path, default=HERE / "tables")
    ap.add_argument("--fig-out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    data = build_tables(args.probes_dir, args.tag, args.tables_out)
    fig1_ACD(data, args.fig_out)
    fig2_AB(data, args.fig_out)
    fig4_A(data, args.fig_out)
    print("[done] maximal figs 1(A,C,D) / 2(A,B) / 4(A)")


if __name__ == "__main__":
    main()
