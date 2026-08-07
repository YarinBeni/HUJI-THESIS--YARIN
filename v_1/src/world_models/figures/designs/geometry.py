#!/usr/bin/env python3
"""DESIGN 6 — "The geometry strip".

2 rows x 4 cols of embedding scatters (row 1: supervised PLS-2D, row 2: UMAP)
for the four canonical cells A / B / B' / C, all coloured by chronological rank
(viridis, one shared colorbar). Teal spines = entity level, warm spines =
fragment (document) level. Each column stamped with the matching ridge-probe
Spearman rho from the TIDY table (trained / random twin / TF-IDF floor).

All numbers are read from committed result files — nothing fabricated.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

# The TIDY table and the output suffix are overridable so the same code renders
# both read-outs (raw = ridge everywhere; deck = the per-cell probe the thesis
# reports — PLS at fragment level, PLS-5 for obscure entities, ridge for cell A).
# The table defaults to the mc_group family and follows FIG_TAG, so `FIG_TAG=__deck`
# reads the deck table without a second environment variable. It used to default to
# TIDY_all_year_results.csv, which is the pre-mc_group (StratifiedKFold) build: running
# a design script the obvious way silently rendered leaky numbers under a caption
# claiming GroupKFold, and a newly added arm was simply absent. TIDY_CSV still
# overrides for one-offs.
_FIGDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIDY_CSV = os.environ.get(
    "TIDY_CSV",
    os.path.join(_FIGDIR, f"TIDY_all_year_results__mc_group{os.environ.get('FIG_TAG', '')}.csv"))
_TAG = os.environ.get("FIG_TAG", "")

# Captions must not hard-code "ridge": under FIG_TAG=__deck the numbers are the
# per-cell probe the thesis reports (PLS at fragment level, PLS-5 for obscure
# entities, ridge for cell A), so the phrase changes with the table.
_PROBE_PHRASE = {"": "ridge probe",
                 "__deck": "thesis read-out (PLS · fragments / PLS-5 · obscure entities / ridge · A)",
                 "__pls": "best-k PLS probe (ridge only where no PLS sweep exists)",
                 }.get(_TAG, "ridge probe")
_PROBE_PHRASE_CAP = _PROBE_PHRASE[0].upper() + _PROBE_PHRASE[1:]

from matplotlib.patches import FancyBboxPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
# Resolution/format policy lives in figures/lib/_save.py (300 dpi PNG + vector PDF)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'lib'))
from _save import save as _save_fig  # noqa: E402


# ---------------------------------------------------------------- paths + data
ROOT = "/home/user/HUJI-THESIS--YARIN/v_1/src/world_models"
MAN = f"{ROOT}/manifold/results"
TIDY = pd.read_csv(TIDY_CSV)
TIDY = TIDY[(TIDY.metric == "spearman") & (TIDY.target == "year")]


def rho(arm, probe="ridge", **kw):
    m = TIDY[(TIDY.arm == arm) & (TIDY.probe == probe)]
    for k, v in kw.items():
        m = m[m[k] == v]
    assert len(m) == 1, (arm, kw, m)
    return float(m.value.iloc[0])


# The four columns of the strip, in narrative order (entity -> fragment).
#   tag        : coords npz stem
#   flip       : y is "years BCE-ish" (bigger = earlier) for the Assyrian cells,
#                so chronological order = -y there; panel A is CE years as-is.
PANELS = [
    dict(
        letter="A", regime="entity",
        tag="eng__qwen3_1b7__historical_figure__last", flip=False,
        title="Famous figures — names",
        sub="Qwen3-1.7B · entity · EN · last",
        n_lab="n = 4,000 people",
        stats=dict(
            model=rho("qwen3_1b7", cleaning="historical_figure", pooling="last"),
            rand=rho("random", cleaning="historical_figure", pooling="last"),
            tfidf=rho("tfidf", cleaning="historical_figure", pooling="text"),
        ),
    ),
    dict(
        letter="B", regime="entity",
        tag="ent__llama2_70b__assyrian_ruler__ent_last", flip=True,
        title="Assyrian rulers — names",
        sub="Llama-2-70B · entity · EN · ent-last",
        n_lab="n = 204 rulers",
        stats=dict(
            model=rho("llama2_70b", cleaning="rows_bare", pooling="ent_last"),
            rand=rho("llama2_70b_random", cleaning="rows_bare", pooling="ent_last"),
            tfidf=rho("tfidf", cleaning="rows_bare", pooling="text"),
        ),
    ),
    dict(
        letter="B′", regime="fragment",
        tag="akk__llama2_13b__eng_tier0__year__last", flip=True,
        title="Glossed fragments (EN)",
        sub="Llama-2-13B · document · EN · last",
        n_lab="n = 1,174 fragments",
        stats=dict(
            model=rho("llama2_13b", cleaning="tier0", pooling="last"),
            rand=rho("llama2_13b_random", cleaning="tier0", pooling="last"),
            tfidf=rho("tfidf", cleaning="tier0", pooling="text"),
        ),
    ),
    dict(
        letter="C", regime="fragment",
        tag="akk__llama2_13b__akk_maximal__year__last", flip=True,
        title="Raw Akkadian fragments",
        sub="Llama-2-13B · document · AKK · last",
        n_lab="n = 1,174 fragments",
        stats=dict(
            model=rho("llama2_13b", cleaning="maximal", pooling="last"),
            rand=rho("llama2_13b_random", cleaning="maximal", pooling="last"),
            tfidf=rho("tfidf", cleaning="maximal", pooling="text"),
        ),
    ),
]

REGIME = {  # the design-law colour regimes
    "entity":   dict(edge="#0e7c6b", band="#0e7c6b", face="#f2faf8", name="ENTITY LEVEL"),
    "fragment": dict(edge="#b0501a", band="#b0501a", face="#fdf6f0", name="FRAGMENT (DOCUMENT) LEVEL"),
}

CMAP = plt.get_cmap("viridis")
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- figure shell
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "text.color": "#1c1c1c",
})

fig, axs = plt.subplots(2, 4, figsize=(13.4, 8.0))
fig.subplots_adjust(left=0.060, right=0.925, top=0.785, bottom=0.132,
                    wspace=0.10, hspace=0.10)

TITLE_Y, SUB_Y = 0.966, 0.932
fig.text(0.060, TITLE_Y, "Where the year gradient lives — and where it dies",
         fontsize=16.5, fontweight="bold", ha="left", va="center")
fig.text(0.060, SUB_Y,
         "2-D views of frozen-LLM activations at the probe's best layer, coloured by time. "
         "The chronological gradient that organises entity embeddings (A, B) dissolves at "
         "document level (B′, C):\nyear-supervised PLS can still forge a weak axis, but "
         "unsupervised UMAP shows no intrinsic temporal structure left to find.",
         fontsize=9.3, ha="left", va="top", color="#444444", linespacing=1.45)


def window(xy, pad=1.10, q=(0.5, 99.5)):
    """Square, robust data window (equal aspect without box distortion)."""
    x0, x1 = np.percentile(xy[:, 0], q)
    y0, y1 = np.percentile(xy[:, 1], q)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max(x1 - x0, y1 - y0) / 2 * pad
    return (cx - h, cx + h), (cy - h, cy + h)


for j, P in enumerate(PANELS):
    d = np.load(f"{MAN}/{P['tag']}.coords.npz")
    y = d["y"].astype(float)
    t = -y if P["flip"] else y                    # chronological order (early -> late)
    c = np.argsort(np.argsort(t)) / (len(t) - 1)  # chronological rank in [0, 1]
    order = rng.permutation(len(t))               # honest draw order
    n = len(t)
    size, alpha = (26, 0.90) if n < 400 else ((8, 0.62) if n < 2000 else (4.5, 0.50))
    rg = REGIME[P["regime"]]

    for i, key in enumerate(["pls", "umap"]):
        ax = axs[i, j]
        xy = d[key][:, :2].astype(float)
        ax.scatter(xy[order, 0], xy[order, 1], c=c[order], cmap=CMAP,
                   vmin=0, vmax=1, s=size, alpha=alpha, linewidths=0,
                   rasterized=True)
        xl, yl = window(xy)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(rg["face"])
        for s in ax.spines.values():
            s.set_color(rg["edge"]); s.set_linewidth(2.0)

        if i == 0:  # column header + panel letter + probe stamp
            ax.set_title(f"{P['title']}\n{P['sub']}", fontsize=9.0, pad=5,
                         color="#222222", linespacing=1.35)
            st = P["stats"]
            ax.text(0.035, 0.965,
                    f"probe ρ = {st['model']:.2f}\n"
                    f"random† {st['rand']:.2f}   TF-IDF {st['tfidf']:.2f}",
                    transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
                    linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec=rg["edge"], lw=0.8, alpha=0.92))
        else:
            ax.text(0.035, 0.965, P["n_lab"], transform=ax.transAxes,
                    fontsize=7.2, va="top", ha="left", color="#555555",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#bbbbbb", lw=0.6, alpha=0.9))
        ax.text(0.955, 0.045, P["letter"], transform=ax.transAxes,
                fontsize=13, fontweight="bold", color=rg["edge"],
                ha="right", va="bottom")

# ------------------------------------------------------------------ row labels
axs[0, 0].set_ylabel("PLS-2D\n(supervised by year)", fontsize=9.5, labelpad=8,
                     linespacing=1.4)
axs[1, 0].set_ylabel("UMAP\n(unsupervised)", fontsize=9.5, labelpad=8,
                     linespacing=1.4)

# --------------------------------------------------------------- regime bands
fig.canvas.draw()  # freeze axes positions
for cols, rkey in (((0, 1), "entity"), ((2, 3), "fragment")):
    b0 = axs[0, cols[0]].get_position()
    b1 = axs[0, cols[1]].get_position()
    rg = REGIME[rkey]
    x0, x1 = b0.x0, b1.x1
    yb, hb = 0.848, 0.030
    fig.add_artist(FancyBboxPatch(
        (x0, yb), x1 - x0, hb, boxstyle="round,pad=0.0022,rounding_size=0.006",
        transform=fig.transFigure, fc=rg["band"], ec="none", alpha=0.94))
    fig.text((x0 + x1) / 2, yb + hb / 2, rg["name"], ha="center", va="center",
             fontsize=9.5, fontweight="bold", color="white")
# cliff marker between the two regimes
gap_x = (axs[0, 1].get_position().x1 + axs[0, 2].get_position().x0) / 2
fig.text(gap_x, 0.848 + 0.015, "▸", ha="center", va="center",
         fontsize=11, color="#7a3b10")

# ------------------------------------------------------------- shared colorbar
cax = fig.add_axes([0.938, 0.20, 0.013, 0.52])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), cax=cax)
cb.set_ticks([0.02, 0.98]); cb.set_ticklabels(["earliest", "latest"], fontsize=8)
cax.tick_params(size=0)
cb.outline.set_linewidth(0.7)
cb.set_label("chronological rank within panel", fontsize=8.5, labelpad=5)

# -------------------------------------------------------------------- footnote
foot = (
    "Colour: percentile of the year target within each panel — A: death year, 935 BCE–2021 CE;  B–C: attested year of ruler / fragment, ≈ 1132–261 BCE.\n"
    f"ρ: Spearman of a {_PROBE_PHRASE} on full-dimension activations at the best layer, year target; pooling matches the embedding site shown (last / ent-last).  "
    "TF-IDF = char-n-gram on the raw text (surface floor).\n"
    "Protocols differ by level (not directly comparable): A = i.i.d. holdout;  B = leave-entities-out (entity MC);  B′, C = leave-rulers-out (ruler MC, r8).  "
    "† random = random-init twin (A: random-init Qwen3-8B).\n"
    "The PLS row is fit on the year target and can manufacture a gradient from weak signal; the UMAP row is label-free and shows which gradients are intrinsic to the geometry."
)
fig.text(0.060, 0.086, foot, fontsize=7.3, color="#555555", va="top",
         linespacing=1.6)

OUT = os.path.dirname(os.path.abspath(__file__)) + f"/geometry{_TAG}.png"
_save_fig(fig, OUT)
print("saved", OUT)
