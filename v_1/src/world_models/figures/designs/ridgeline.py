#!/usr/bin/env python3
"""DESIGN 1 -- 'The collapse ridgeline'.

One ridge per canonical configuration (10 total, entity -> fragment, top -> bottom).
x = year-decoding Spearman rho. Each ridge is a discrete 0.05-bin histogram over the
14 model arms' scores; per-arm markers sit on the baseline; TF-IDF is a black diamond.
Hard hue switch (teal -> warm) at the entity/document boundary, with a labeled divider.
All numbers come from the committed TIDY table -- nothing is fabricated.
"""
import csv
import os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as pe

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib')
sys.path.insert(0, SCRATCH)
from _style import COL, LAB, isr  # noqa: E402
# Resolution/format policy lives in figures/lib/_save.py (300 dpi PNG + vector PDF)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'lib'))
from _save import save as _save_fig  # noqa: E402


# The TIDY table and the output suffix are overridable so the same code renders
# both read-outs (raw = ridge everywhere; deck = the per-cell probe the thesis
# reports — PLS at fragment level, PLS-5 for obscure entities, ridge for cell A).
TIDY_CSV = os.environ.get("TIDY_CSV", "/home/user/HUJI-THESIS--YARIN/v_1/src/world_models/figures/TIDY_all_year_results.csv")
_TAG = os.environ.get("FIG_TAG", "")

# Captions must not hard-code "ridge": under FIG_TAG=__deck the numbers are the
# per-cell probe the thesis reports (PLS at fragment level, PLS-5 for obscure
# entities, ridge for cell A), so the phrase changes with the table.
_PROBE_PHRASE = {"": "ridge probe",
                 "__deck": "thesis read-out (PLS · fragments / PLS-5 · obscure entities / ridge · A)",
                 "__pls": "best-k PLS probe (ridge only where no PLS sweep exists)",
                 }.get(_TAG, "ridge probe")
_PROBE_PHRASE_CAP = _PROBE_PHRASE[0].upper() + _PROBE_PHRASE[1:]


TIDY = TIDY_CSV
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/ridgeline{_TAG}.png"

# ----------------------------------------------------------------------------- data
rows = list(csv.DictReader(open(TIDY)))

# (level, salience, cleaning, pooling, cell, terse label)
CFGS = [
    ("entity",   "salient", "historical_figure", "last",     "A",  "A · famous names · last"),
    ("entity",   "salient", "historical_figure", "mean",     "A",  "A · famous names · mean"),
    ("entity",   "obscure", "rows_bare",         "ent_last", "B",  "B · ruler name · ent-last"),
    ("entity",   "obscure", "rows_bare",         "ent_mean", "B",  "B · ruler name · ent-mean"),
    ("entity",   "obscure", "rows_all",          "ent_last", "B",  "B · name in sent. · ent-last"),
    ("entity",   "obscure", "rows_all",          "last",     "B",  "B · name in sent. · sent-last"),
    ("fragment", "obscure", "tier0",             "last",     "B'", "B′ · EN-gloss frag · last"),
    ("fragment", "obscure", "tier0",             "mean",     "B'", "B′ · EN-gloss frag · mean"),
    ("fragment", "obscure", "maximal",           "last",     "C",  "C · Akkadian frag · last"),
    ("fragment", "obscure", "maximal",           "mean",     "C",  "C · Akkadian frag · mean"),
]

def pull(level, sal, clean, pool):
    """Return ({arm: rho} for the 14 model arms, tfidf rho) for one configuration."""
    arms, tfidf = {}, None
    for r in rows:
        if (r["metric"], r["probe"], r["target"]) != ("spearman", "ridge", "year"):
            continue
        if (r["level"], r["salience"], r["cleaning"]) != (level, sal, clean):
            continue
        if r["arm"] == "tfidf" and r["pooling"] == "text":
            tfidf = float(r["value"])
        elif r["pooling"] == pool and r["arm"] != "tfidf":
            arms[r["arm"]] = float(r["value"])
    return arms, tfidf

DATA = [pull(l, s, c, p) for l, s, c, p, _, _ in CFGS]
for (arms, tf), cfg in zip(DATA, CFGS):
    assert len(arms) == 14 and tf is not None, (cfg, len(arms), tf)

# ------------------------------------------------------------------------ histogram
# Each ridge summarises exactly 14 arms. A Gaussian KDE over 14 points invents a smooth
# density that the data does not support — it implies mass between arms and hides how
# few there are. These are discrete counts in fixed 0.05-wide bins, drawn as stairs, so
# one bar of height 1 is literally one model arm.
BIN_W = 0.05
BIN_EDGES = np.arange(-0.10, 1.0001 + BIN_W, BIN_W)


def hist_steps(vals):
    """-> (x, y) tracing the top of the bars, closed to the baseline at both ends."""
    counts, _ = np.histogram(np.asarray(vals, float), bins=BIN_EDGES)
    x = np.repeat(BIN_EDGES, 2)
    y = np.concatenate([[0.0], np.repeat(counts.astype(float), 2), [0.0]])
    return x, y


GRID = np.linspace(-0.18, 1.18, 900)

# ----------------------------------------------------------------------------- layout
S = 1.0            # baseline spacing within a regime
CLIFF_GAP = 3.3    # extra-wide gap at the entity/document boundary
PEAK = 1.55        # vertical budget reserved above each baseline
COUNT_H = 0.22     # height of ONE arm; bar heights are counts, comparable across ridges

ys, y = [], 0.0
for i in range(len(CFGS)):
    if i > 0:
        y -= CLIFF_GAP if (CFGS[i - 1][0] == "entity" and CFGS[i][0] == "fragment") else S
    ys.append(y)
ys = np.array(ys)
DIV_Y = ys[5] - 1.18   # divider sits inside the widened gap

ENT_C = LinearSegmentedColormap.from_list("ent", ["#a9ddd6", "#0b4f4a"])
FRG_C = LinearSegmentedColormap.from_list("frg", ["#f7bd7e", "#7f2704"])
def ridge_face(i):
    lvl = CFGS[i][0]
    if lvl == "entity":
        return ENT_C(0.14 + 0.72 * i / 5)
    return FRG_C(0.18 + 0.74 * (i - 6) / 3)

def darken(c, f=0.60):
    r, g, b = to_rgb(c)
    return (r * f, g * f, b * f)

# ----------------------------------------------------------------------------- figure
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5,
})
fig, ax = plt.subplots(figsize=(9.4, 10.8))
fig.subplots_adjust(left=0.252, right=0.862, top=0.908, bottom=0.128)

XL, XR = -0.075, 1.04
ax.set_xlim(XL, XR)
ax.set_ylim(ys[-1] - 0.85, ys[0] + PEAK + 0.42)

# vertical gridlines
for gx in np.arange(0.0, 1.01, 0.2):
    ax.axvline(gx, color="#d9d9d9", lw=0.7, zorder=0)
ax.axvline(0.0, color="#9a9a9a", lw=0.9, zorder=0)

# TF-IDF trace across ridges (drawn behind fills so ridges occlude it cleanly)
tfx = [tf for _, tf in DATA]
ax.plot(tfx, ys, color="#000000", lw=1.0, ls=(0, (1.5, 2.2)), alpha=0.55, zorder=1)

# ---- ridges, painted top -> bottom so lower ridges sit in front (joyplot order)
for i, ((arms, tf), (lvl, sal, clean, pool, cell, label)) in enumerate(zip(DATA, CFGS)):
    vals = np.array(list(arms.values()))
    hx, hy = hist_steps(vals)
    # one arm = COUNT_H, so bar height is comparable across ridges (a 4-high bar is
    # 4 arms wherever it appears); PEAK stays the layout budget for the tallest bar.
    h = hy * COUNT_H
    face = ridge_face(i)
    edge = darken(face)
    z = 2 + i
    ax.fill_between(hx, ys[i], ys[i] + h, step=None, facecolor=face, edgecolor="none",
                    zorder=z)
    ax.plot(hx, ys[i] + h, color=edge, lw=1.2, zorder=z + 0.1,
            solid_joinstyle="miter")
    ax.plot([XL, XR], [ys[i], ys[i]], color=edge, lw=0.9, zorder=z + 0.1)
    # faint interior bin separators so the discreteness is unmistakable
    for e in BIN_EDGES[1:-1]:
        c = hy[np.searchsorted(hx, e, side="right") - 1]
        if c > 0:
            ax.plot([e, e], [ys[i], ys[i] + c * COUNT_H], color="white", lw=0.6,
                    alpha=0.55, zorder=z + 0.05)

    # left-hand terse label
    ax.text(XL - 0.014, ys[i] + 0.06, label, ha="right", va="bottom", fontsize=8.6,
            color=darken(face, 0.55), clip_on=False, zorder=80,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

    # right margin: best arm + rho
    best = max(list(arms.items()) + [("tfidf", tf)], key=lambda kv: kv[1])
    bc = "#000000" if best[0] == "tfidf" else COL[best[0]]
    ax.text(XR + 0.014, ys[i], f"{LAB[best[0]]}  ρ={best[1]:.2f}".replace("0.", "."),
            ha="left", va="center", fontsize=7.0, color=bc, clip_on=False)

# ---- per-arm markers on each baseline (tiny downward beeswarm on collisions)
def beeswarm(items, eps=0.021, dy=0.185):
    out, placed = [], []
    for arm, v in sorted(items, key=lambda kv: kv[1]):
        row = 0
        while any(abs(v - pv) < eps and pr == row for pv, pr in placed):
            row += 1
        placed.append((v, row))
        out.append((arm, v, -row * dy))
    return out

for i, (arms, tf) in enumerate(DATA):
    for arm, v, off in beeswarm(list(arms.items())):
        if isr(arm):
            ax.plot(v, ys[i] + off, "o", ms=5.6, mfc="white", mec=COL[arm], mew=1.3,
                    zorder=60, clip_on=False)
        else:
            ax.plot(v, ys[i] + off, "o", ms=6.2, mfc=COL[arm], mec="white", mew=0.7,
                    zorder=61, clip_on=False)
    ax.plot(tf, ys[i], "D", ms=6.4, mfc="#000000", mec="white", mew=0.8, zorder=62,
            clip_on=False)

# ---- the cliff divider
ax.plot([XL, XR], [DIV_Y, DIV_Y], color="#3a3a3a", lw=1.4, ls=(0, (6, 3)), zorder=40,
        clip_on=False)
ax.text((XL + XR) / 2, DIV_Y + 0.14, "THE  ENTITY → DOCUMENT  CLIFF",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#3a3a3a",
        zorder=41, path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
ax.text((XL + XR) / 2, DIV_Y - 0.18,
        "below this line, trained LLMs fall to — or under — their random-init twins; "
        + ("the char-n-gram floor drops out entirely and only the cuneiform encoders "
           "stay ahead" if _TAG else
           "only TF-IDF and the cuneiform encoders stay ahead"),
        ha="center", va="top", fontsize=7.4, style="italic", color="#555555", zorder=41,
        path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

# ---- regime sidebars
side = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
for (i0, i1, txt, col) in [(0, 5, "ENTITY  ·  names", ENT_C(0.80)),
                           (6, 9, "DOCUMENT  ·  fragments", FRG_C(0.82))]:
    ym = (ys[i0] + PEAK * 0.55 + ys[i1]) / 2
    ax.plot([-0.298, -0.298], [ys[i1] - 0.25, ys[i0] + PEAK * 0.8], transform=side,
            color=col, lw=3.0, clip_on=False, solid_capstyle="butt")
    ax.text(-0.316, ym, txt, transform=side, rotation=90, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=col, clip_on=False)

# ---- axes cosmetics
ax.set_yticks([])
for sp in ("left", "top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_bounds(0.0, 1.0)
ax.set_xticks(np.arange(0.0, 1.01, 0.2))
ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1.0"])
ax.set_xlabel("Spearman ρ  (probe-decoded year vs. true year)", fontsize=10)
ax.tick_params(axis="x", length=3, width=0.8)

ax.text(XR + 0.014, ys[0] + PEAK + 0.42, "best arm", ha="left", va="center",
        fontsize=7.2, color="#777777", style="italic", clip_on=False)

# ---- titles
fig.text(0.042, 0.968, "Linear world-models of time collapse at the entity → document boundary",
         fontsize=14.5, fontweight="bold", color="#1a1a1a", ha="left")
fig.text(0.042, 0.947,
         "Year decoding (Spearman ρ) · 10 configurations × 14 model arms (dots) + char-n-gram "
         "TF-IDF floor (♦) · ridges are discrete histograms (bin = 0.05)",
         fontsize=8.7, color="#555555", ha="left")

# ---- legend
handles = []
for m in ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b",
          "llama2_7b", "llama2_13b", "llama2_70b",
          "thalesian_akk300m", "thalesian_cunei400m", "umt5_base"]:
    handles.append(Line2D([], [], marker="o", ls="none", ms=6.2, mfc=COL[m], mec="white",
                          mew=0.7, label=LAB[m]))
handles.append(Line2D([], [], marker="o", ls="none", ms=5.6, mfc="white", mec="#666666",
                      mew=1.3, label="random-init twin (open)"))
handles.append(Line2D([], [], marker="D", ls="none", ms=6.2, mfc="#000000", mec="white",
                      mew=0.8, label="char-n-gram TF-IDF"))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.047), ncol=6,
           frameon=False, fontsize=7.4, handletextpad=0.25, columnspacing=1.05,
           borderaxespad=0.0)

# ---- footnote (two lines so nothing runs off the canvas)
fig.text(0.042, 0.030,
         f"{_PROBE_PHRASE_CAP}, target = year.  Protocols differ by level, not pooled: "
         "A = i.i.d. holdout · B = entity-level MC · B′/C = ruler-grouped MC (r = 8).",
         fontsize=6.6, color="#777777", ha="left")
fig.text(0.042, 0.016,
         "Bars are counts in 0.05-wide bins — one bar-step = one arm, so heights compare "
         "across ridges.  Open markers = random-init controls; ♦ = char-n-gram TF-IDF "
         "(no LLM); * = untrained baseline.",
         fontsize=6.6, color="#777777", ha="left")

_save_fig(fig, OUT)
print("wrote", OUT)
