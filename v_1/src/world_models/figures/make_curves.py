"""Regenerate the layer-curve and PLS-k figures (committed PNGs 02-08).

  02_cellA_layers          English (cell A) per-layer, TIME=Spearman / SPACE=R2, last|mean
  03_cellA_pls             English best-layer PLS, k = 1..64
  04_cellB_entity_layers   34 rulers / 25 places, per-layer, ent_last|ent_mean
  05/06_cellB_entity_pls   the same surfaces, PLS k = 1..64 (rows: bare | all)
  07_fragment_layers       akk_maximal + eng_tier0 per-layer, YEAR=rho / GEO=R2, last|mean
  08_fragment_pls          the same, best-layer PLS k = 1..64

Conventions used throughout: Qwen blues / Llama greens / encoders orange-purple, darker =
larger; dashed = random-init controls; dotted = TF-IDF floor; a star marks each curve's
maximum; GEO panels use a symlog y-axis so the negative tail does not squash the 0-1 band.

    python make_curves.py [--which all|cellA|cellB|fragment] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_HERE, "lib"))
from _style import COL, LAB, ENC, ORDER, isr, sty, star   # noqa: E402
# Resolution/format policy lives in figures/lib/_save.py (300 dpi PNG + vector PDF)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'lib'))
from _save import save as _save_fig  # noqa: E402


KS = [1, 2, 3, 5, 8, 16, 32, 64]
SPACE = ["world_place", "us_place", "nyc_place"]
TIME = ["historical_figure", "art", "headline"]


# --------------------------------------------------------------- cell A (English)
def cell_a(out):
    d = pd.read_csv(os.path.join(_WM, "results", "summary_layerwise.csv"))
    d = d[d.method != "tfidf"].copy()
    d["nd"] = d.groupby(["method", "site"]).layer.transform(
        lambda s: (s - s.min()) / max(1, (s.max() - s.min())))
    cells = [("TIME", "test_spearman", "Spearman", TIME, "last"),
             ("TIME", "test_spearman", "Spearman", TIME, "mean"),
             ("SPACE", "test_r2", "R²", SPACE, "last"),
             ("SPACE", "test_r2", "R²", SPACE, "mean")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (grp, fld, ml, dsets, site) in zip(axes.ravel(), cells):
        g = (d[(d.entity_type.isin(dsets)) & (d.site == site)]
             .groupby(["method", "nd"])[fld].mean().reset_index())
        n = 0
        for m in ORDER:
            s = g[g.method == m].sort_values("nd")
            if s.empty:
                continue
            ax.plot(s.nd.values, s[fld].values, **sty(m), label=LAB.get(m, m))
            star(ax, s.nd.values, s[fld].values, m)
            n += 1
        if grp == "SPACE":
            ax.set_yscale("symlog", linthresh=0.1)
        ax.grid(alpha=.25); ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"English · {grp} · pool={site}  ({ml})  [{n} arms]", fontsize=11)
        ax.set_xlabel("normalized depth"); ax.set_ylabel(f"{ml} (mean of 3 datasets)")
    axes[0, 0].legend(fontsize=6.5, ncol=2, loc="lower center")
    fig.suptitle("ENGLISH (Gurnee & Tegmark redo) — TIME=Spearman, SPACE=R² · "
                 "last | mean pooling · ★=max", fontweight="bold")
    fig.tight_layout(); _save_fig(fig, f"{out}/02_cellA_layers.png"); plt.close(fig)

    P = os.path.join(_WM, "results", "eng_pls")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (grp, fld, ml, dsets, site) in zip(axes.ravel(), cells):
        n = 0
        for m in ORDER:
            ys = []
            for k in KS:
                v = []
                for et in dsets:
                    f = f"{P}/{m}/{et}.{site}.json"
                    if os.path.exists(f):
                        blk = json.load(open(f))["pls_at_best_layer"].get(str(k), {})
                        if fld in blk:
                            v.append(blk[fld])
                ys.append(np.mean(v) if v else np.nan)
            if not np.isfinite(ys).any():
                continue
            xx = np.arange(len(KS))
            ax.plot(xx, ys, **sty(m), marker="o", ms=3, label=LAB.get(m, m))
            star(ax, xx, ys, m); n += 1
        if grp == "SPACE":
            ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS, fontsize=8)
        ax.grid(alpha=.25); ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"English · {grp} · pool={site}  ({ml})  [{n} arms]", fontsize=11)
        ax.set_xlabel("PLS components k"); ax.set_ylabel(f"{ml} (mean of 3 datasets)")
    axes[0, 0].legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.suptitle("ENGLISH best-layer PLS — TIME=Spearman, SPACE=R² vs k (1…64) · ★=best k",
                 fontweight="bold")
    fig.tight_layout(); _save_fig(fig, f"{out}/03_cellA_pls.png"); plt.close(fig)


# --------------------------------------------- fragment cells (akk_maximal / eng_tier0)
def fragment(out):
    P = os.path.join(_WM, "akkadian", "results", "layers_pls")
    cols = [("year", "last", "test_spearman", "Spearman"),
            ("year", "mean", "test_spearman", "Spearman"),
            ("geo", "last", "test_r2", "R²"),
            ("geo", "mean", "test_r2", "R²")]
    for kind in ("layers", "pls"):
        fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
        for ri, v in enumerate(("akk_maximal", "eng_tier0")):
            for ci, (t, s, fld, ml) in enumerate(cols):
                ax = axes[ri, ci]; n = 0
                for m in ORDER:
                    if m in ENC and s == "last":
                        continue                       # encoders: mean only
                    f = f"{P}/{m}/{v}.{t}.{s}.json"
                    if not os.path.exists(f):
                        continue
                    d = json.load(open(f))
                    if kind == "layers":
                        x = np.array([p["nd"] for p in d["per_layer"]])
                        y = [p[fld] for p in d["per_layer"]]
                    else:
                        pk = d["pls_at_best_layer"]; x = np.arange(len(KS))
                        y = [pk.get(str(k), {}).get(fld, np.nan) for k in KS]
                    ax.plot(x, y, **sty(m), marker=None if kind == "layers" else "o",
                            ms=3, label=LAB.get(m, m))
                    star(ax, x, y, m); n += 1
                if t == "geo":
                    ax.set_yscale("symlog", linthresh=0.1)
                if kind == "pls":
                    ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS, fontsize=7)
                    ax.set_xlabel("PLS k")
                else:
                    ax.set_xlabel("normalized depth")
                ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25)
                ax.set_title(f"{v} · {t} · {s} ({ml}) [{n}]", fontsize=9.5)
                if ci == 0:
                    ax.set_ylabel(v, fontsize=11)
        axes[0, 1].legend(fontsize=6, ncol=2,
                          loc="lower center" if kind == "layers" else "lower right")
        ttl = ("AKKADIAN layer curves" if kind == "layers"
               else "AKKADIAN best-layer PLS vs k (1…64)")
        fig.suptitle(f"{ttl} — YEAR=Spearman, GEO=R² (symlog) · last | mean · "
                     f"encoders in mean · ★=max", fontweight="bold")
        fig.tight_layout()
        _save_fig(fig, f"{out}/{'07_fragment_layers' if kind=='layers' else '08_fragment_pls'}.png")
        plt.close(fig)


# ------------------------------------------------------ cell B, entity level (rulers)
def cell_b(out):
    PE = os.path.join(_WM, "akkadian", "results", "probes_entity")
    PP = os.path.join(_WM, "akkadian", "results", "probes_entity_pls")
    rows = [("mesopotamian_place", "SPACE (place → lon/lat)"),
            ("assyrian_ruler", "TIME (ruler → reign year)")]
    cols = [("ent_last", "mc_r2", "R²"), ("ent_last", "mc_rho", "Spearman ρ"),
            ("ent_mean", "mc_r2", "R²"), ("ent_mean", "mc_rho", "Spearman ρ")]
    arms = ORDER + ["tfidf"]

    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for ri, (et, rlab) in enumerate(rows):
        for ci, (site, fld, ml) in enumerate(cols):
            ax = axes[ri, ci]
            for m in arms:
                f = f"{PE}/{m}/{et}.{'text' if m=='tfidf' else site}.json"
                if not os.path.exists(f):
                    continue
                L = json.load(open(f))["layers"]
                lids = sorted(int(k) for k in L)
                xs, ys = [], []
                for li in lids:
                    e = L[str(li)].get("bare") or L[str(li)].get("all")
                    if not e:
                        continue
                    xs.append((li - lids[0]) / max(1, lids[-1] - lids[0]))
                    ys.append(e.get("ridge_mc", {}).get(fld))
                if not xs:
                    continue
                if m == "tfidf":
                    ax.axhline(np.nanmean(ys), color="k", ls=":", lw=1.2,
                               label="TF-IDF floor")
                else:
                    ax.plot(xs, ys, **sty(m), label=LAB.get(m, m)); star(ax, xs, ys, m)
            if fld == "mc_r2":
                ax.set_yscale("symlog", linthresh=0.5)
            ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25)
            ax.set_title(f"{rlab.split(' ')[0]} · {site} · {ml}", fontsize=10)
            ax.set_xlabel("depth (layer / total layers)")
            if ci == 0:
                ax.set_ylabel(rlab, fontsize=9)
    axes[0, 1].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle("Cell B at ENTITY level (34 rulers / 25 places, English) — per-layer "
                 "entity-MC probe, `bare` rows · ★=best layer", fontweight="bold")
    fig.tight_layout(); _save_fig(fig, f"{out}/04_cellB_entity_layers.png")
    plt.close(fig)

    for tag, fn in (("bare", "05_cellB_entity_pls_bare"), ("all", "06_cellB_entity_pls_all")):
        fig, axes = plt.subplots(2, 4, figsize=(19, 9))
        for ri, (et, rlab) in enumerate(rows):
            for ci, (site, fld, ml) in enumerate(cols):
                ax = axes[ri, ci]
                for m in arms:
                    f = f"{PP}/{m}/{et}.{'text' if m=='tfidf' else site}.json"
                    if not os.path.exists(f):
                        continue
                    row = json.load(open(f)).get("rows", {}).get(tag)
                    if not row:
                        continue
                    pk = row.get("pls_by_k", {})
                    ys = [pk.get(str(k), {}).get(fld, np.nan) for k in KS]
                    if not np.isfinite(ys).any():
                        continue
                    xx = np.arange(len(KS))
                    if m == "tfidf":
                        ax.plot(xx, ys, color="k", ls=":", lw=1.4, label="TF-IDF floor")
                    else:
                        ax.plot(xx, ys, **sty(m), marker="o", ms=3, label=LAB.get(m, m))
                        star(ax, xx, ys, m)
                if fld == "mc_r2":
                    ax.set_yscale("symlog", linthresh=0.5)
                ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS, fontsize=8)
                ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25)
                ax.set_title(f"{rlab.split(' ')[0]} · {site} · {ml}", fontsize=10)
                ax.set_xlabel("PLS components k")
                if ci == 0:
                    ax.set_ylabel(rlab, fontsize=9)
        axes[0, 1].legend(fontsize=6, ncol=2, loc="lower right")
        fig.suptitle(f"Cell B at ENTITY level — best-layer PLS, k = 1…64 · rows='{tag}' "
                     f"(entity-MC) · ★=best k", fontweight="bold")
        fig.tight_layout(); _save_fig(fig, f"{out}/{fn}.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "cellA", "cellB", "fragment"])
    ap.add_argument("--out", default=_HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.which in ("all", "cellA"):
        cell_a(a.out); print("cell A done")
    if a.which in ("all", "fragment"):
        fragment(a.out); print("fragment done")
    if a.which in ("all", "cellB"):
        cell_b(a.out); print("cell B done")


if __name__ == "__main__":
    main()
