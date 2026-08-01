#!/usr/bin/env python3
"""DESIGN 7 — "How many latent dimensions does the date live in?"

One panel per fragment cell (B' EN gloss, C raw Akkadian) x pooling. x = PLS latent
dimension k (log axis), y = grouped-MC Spearman rho. One line per arm; the chosen k is
ringed. A dashed vertical marks the OLD grid ceiling (k = 5) that the thesis inherited
from shared/mc_probe.py.

This is the figure that answers "should k be swept to 64 instead of 5?". If the curves
are still climbing at the old ceiling, the grid was choosing k, not the data — which is
what 18 of 58 cells showed. After WAk_pls_ksweep lands, the same code plots 11 k values
instead of 4 and the answer is legible directly.

Reads `pls_per_k` out of the mc_group block of every fragment probe JSON, so it needs
no cluster access and updates itself when the sweep re-commits those files.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from _style import COL, LAB, isr, rc                                # noqa: E402
from _save import save as _save_fig                                 # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.dirname(os.path.dirname(_HERE))
PROBES = os.path.join(_WM, "akkadian", "results", "probes")
OUT = os.path.join(_HERE, "kprofile.png")

OLD_CEILING = 5          # the inherited PLS_KS = {1,2,3,5}
CELLS = [("eng_tier0", "mean", "B′ · English gloss · mean"),
         ("eng_tier0", "last", "B′ · English gloss · last"),
         ("akk_maximal", "mean", "C · raw Akkadian · mean"),
         ("akk_maximal", "last", "C · raw Akkadian · last")]
WARM, INK, MUT = "#b0501a", "#1c1c1c", "#6d6d6d"


def curves(variant, pool):
    """-> {arm: ([k], [rho], chosen_k)} from the committed mc_group blocks."""
    out = {}
    for d in sorted(glob.glob(os.path.join(PROBES, "*"))):
        arm = os.path.basename(d)
        pl = "text" if arm == "tfidf" else pool
        f = os.path.join(d, f"{variant}.r8.year.{pl}.ridge.json")
        if not os.path.exists(f):
            continue
        blk = (json.load(open(f)) or {}).get("mc_group")
        if not isinstance(blk, dict):
            continue
        per_k = blk.get("pls_per_k") or {}
        ks, rs = [], []
        for k, v in sorted(per_k.items(), key=lambda kv: int(kv[0])):
            r = (v or {}).get("spearman_mean")
            if isinstance(r, (int, float)) and r == r:
                ks.append(int(k)); rs.append(float(r))
        if len(ks) >= 2:
            out[arm] = (ks, rs, blk.get("pls_best_k"))
    return out


fig, axes = plt.subplots(1, len(CELLS), figsize=(5.6 * len(CELLS), 6.2), sharey=True)
rc()

kmax_seen = OLD_CEILING
for ax, (variant, pool, title) in zip(axes, CELLS):
    C = curves(variant, pool)
    for arm, (ks, rs, bk) in sorted(C.items(), key=lambda kv: -max(kv[1][1])):
        kmax_seen = max(kmax_seen, max(ks))
        if arm == "tfidf":
            ax.plot(ks, rs, color="#000000", lw=2.4, ls=(0, (1.5, 1.6)), zorder=5)
        elif isr(arm):
            ax.plot(ks, rs, color="#a8a8a8", lw=1.1, ls=(0, (5, 3)), zorder=2)
        else:
            ax.plot(ks, rs, color=COL.get(arm, "#888888"), lw=1.9, alpha=.95, zorder=3)
        if bk in ks:                       # ring the k the sweep selected
            ax.scatter([bk], [rs[ks.index(bk)]], s=52, facecolor="none", zorder=6,
                       edgecolor=("#000000" if arm == "tfidf"
                                  else COL.get(arm, "#888888")), linewidth=1.6)
    ax.axvline(OLD_CEILING, color=WARM, lw=1.3, ls=(0, (4, 3)), zorder=1)
    # log spacing (k is geometric), but ticks labelled with the actual k values —
    # "2^0 / 2^1" tick labels tell the reader nothing about which k was fitted.
    ax.set_xscale("log", base=2)
    ticks = [k for k in (1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64) if k <= kmax_seen]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(k) for k in ticks], fontsize=13)
    ax.minorticks_off()
    ax.set_xlabel("PLS latent dimensions  k", fontsize=15, color=MUT)
    ax.set_title(title, fontsize=16.5, color=INK, fontweight="bold", pad=8)
    ax.axhline(0, color="#9a9a9a", lw=0.9)
    ax.grid(color="#ececec", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("Spearman ρ   (grouped MC, r = 8)", fontsize=15, color=INK)
axes[0].annotate("old grid ceiling", xy=(OLD_CEILING, axes[0].get_ylim()[1]),
                 xytext=(-4, -12), textcoords="offset points", ha="right",
                 fontsize=12, color=WARM, rotation=90, va="top")

swept = kmax_seen > OLD_CEILING
fig.suptitle("How many latent dimensions does the date live in?",
             fontsize=20, fontweight="bold", x=0.045, ha="left", y=0.985)
fig.text(0.045, 0.925,
         "Grouped-MC Spearman ρ as a function of the PLS rank k, one line per arm; the "
         "ring marks the k that was selected. Dashed vertical = the k = 5 ceiling "
         "inherited from shared/mc_probe.py.",
         fontsize=13, color="#555555", ha="left", va="top")
fig.text(0.045, 0.035,
         ("Curves extend past the old ceiling, so k is now chosen by the data rather "
          "than by the grid. Where a curve is still rising at the right-hand edge, the "
          "sweep is still truncated."
          if swept else
          "Every curve stops at k = 5 because that is where the inherited grid stopped "
          "— 18 of 58 cells selected exactly k = 5, which is the signature of a grid "
          "that is binding. WAk_pls_ksweep extends this to k = 64."),
         fontsize=12, color=("#555555" if swept else WARM), ha="left", va="top")
fig.subplots_adjust(left=0.055, right=0.985, top=0.775, bottom=0.155, wspace=0.08)
_save_fig(fig, OUT)
