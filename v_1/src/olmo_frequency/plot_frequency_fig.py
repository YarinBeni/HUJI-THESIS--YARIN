#!/usr/bin/env python3
"""STEP 3c: the dose-response figure — the actual deliverable of this experiment.

One dot per held-out entity: x = how often it appears in OLMo's training data,
y = how far off the probe's year estimate is. Trained OLMo on the left, its untrained
twin on the right, on shared axes so the two panels can be read against each other
without arithmetic.

WHAT WOULD COUNT AS A RESULT. A downhill trend on the left and a flat one on the right.
Downhill on both would mean the entity STRING carries the signal (long famous names
tokenise differently), not the training exposure — which is exactly why the twin is here
and why it shares the axes.

The third panel is the confound control: the same correlation computed inside each
death-century bin. Old people are both rarer and harder to date, so the overall trend
could be pure age; if the per-century bars stay on the same side of zero, it is not.

House style is the deck's (figures/lib/_style.py), so this can sit next to the existing
slides. OLMo has no deck colour of its own — it is a new arm — so it takes a teal that
is not in use by any family, and the twin takes the deck's control purple, dashed,
because that is what purple means everywhere else in this thesis.

    python plot_frequency_fig.py
    FIG_DPI=450 python plot_frequency_fig.py        # poster resolution
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WM = os.path.join(os.path.dirname(HERE), "world_models")
sys.path.insert(0, os.path.join(WM, "figures"))
from lib import _save, _style                                        # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(RESULTS, "figs")
OUT = os.path.join(FIGS, "fig_frequency_doseresponse.png")

TRAINED, TWIN = "olmo2_7b", "olmo2_7b_random"
# teal: unused by the deck's blue/green/warm families, so it reads as "a new arm"
# rather than as a member of one. The twin keeps the deck's control purple.
C_TRAINED, C_TWIN = "#0f8b8d", _style.COL["llama2_70b_random"]
TITLE = {TRAINED: "OLMo-2-7B  (trained)", TWIN: "its random-init twin  (control)"}


def binned_median(x, y, nbins=12):
    """Median error per frequency decile-ish bin — the trend line the eye needs when
    thousands of dots overlap. Returns bin centres and medians."""
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return np.array([]), np.array([])
    idx = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    cx, cy = [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() >= 8:
            cx.append(float(np.median(x[m])))
            cy.append(float(np.median(y[m])))
    return np.asarray(cx), np.asarray(cy)


def main():
    import matplotlib.pyplot as plt

    st_path = os.path.join(RESULTS, "frequency_stats.json")
    if not os.path.exists(st_path):
        sys.exit(f"no {st_path} — run analyze_frequency.py first.")
    st = json.load(open(st_path))

    _style.rc()
    fig = plt.figure(figsize=(17, 6.6), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.92])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    frames = {a: pd.read_csv(os.path.join(RESULTS, f"joined_{a}.csv"))
              for a in (TRAINED, TWIN)}
    ymax = float(np.quantile(pd.concat(frames.values())["abs_err"], 0.99))

    for ax, arm, col in ((axes[0], TRAINED, C_TRAINED), (axes[1], TWIN, C_TWIN)):
        d = frames[arm]
        x, y = d["logc"].values, d["abs_err"].values
        ax.scatter(x, y, s=9, alpha=0.16, color=col, edgecolors="none", zorder=2)
        bx, by = binned_median(x, y)
        if len(bx):
            ax.plot(bx, by, color=col, lw=3.0, marker="o", ms=8, mec="white", mew=1.4,
                    zorder=5, label="median error per frequency bin")
        s = st[arm]
        # the stats sit on their own line at a smaller size: at the deck's 16.5pt
        # title scale a single line runs wider than the panel and collides with
        # its neighbour
        ax.set_title(TITLE[arm], pad=26, fontweight="bold")
        ax.text(0.5, 1.012,
                f"$\\rho$ = {s['overall_rho']:+.3f}   ·   "
                f"within-century {s['within_century_rho']:+.3f}   ·   n = {s['n']}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=12.5,
                color="#333")
        ax.set_xlabel("times the name appears in OLMo's training data\n"
                      "$\\log_{10}(\\mathrm{count}+1)$")
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.22, lw=0.7)
        ax.legend(loc="upper right", frameon=False)
    axes[0].set_ylabel("| predicted year $-$ true year |   (held-out)")
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)

    # --- panel 3: the age confound, bin by bin ------------------------------------
    ax = axes[2]
    cents = sorted(set(st[TRAINED]["bins"]) | set(st[TWIN]["bins"]), key=int)
    ypos = np.arange(len(cents))
    h = 0.38
    for off, arm, col in ((+h / 2, TRAINED, C_TRAINED), (-h / 2, TWIN, C_TWIN)):
        vals = [st[arm]["bins"].get(c, {}).get("rho", np.nan) for c in cents]
        ax.barh(ypos + off, vals, height=h, color=col,
                hatch="//" if arm == TWIN else None,
                edgecolor="white", lw=0.6, label=TITLE[arm].split("  ")[0])
    ax.axvline(0, color="#444", lw=1.1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{int(c)}s" for c in cents])
    ax.set_xlabel("$\\rho$(frequency, error) inside one century")
    ax.set_title("the age confound, controlled\n"
                 f"bins with n $\\geq$ {st.get('min_bin', 30)}", pad=10,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.22, lw=0.7)
    ax.legend(loc="lower left", frameon=False, fontsize=12)

    fig.suptitle("Does the probe date an entity better when OLMo saw it more often?"
                 f"   ·   counts from {st.get('index_used', '?')}"
                 "   ·   negative $\\rho$ = more frequent, better dated",
                 fontsize=17.5, fontweight="bold")
    os.makedirs(FIGS, exist_ok=True)
    _save.save(fig, OUT)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
