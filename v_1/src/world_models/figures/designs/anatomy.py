#!/usr/bin/env python3
"""DESIGN 5 -- "Stimulus anatomy cards".

Four cards (A / B / B' / C), each stacked: (top) the stimulus as clean monospace
typography with the pooled tokens underlined, (middle) horizontal rho bars for
top-7 arms + random controls + TF-IDF, (bottom) mini layer-depth curves.
Entity cards live in a teal regime, document cards in a warm regime.
All numbers are read from committed result files -- nothing fabricated.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

sys.path.insert(0, "/tmp/claude-0/-home-user-HUJI-THESIS--YARIN/c76dd482-0f95-59a6-8588-5dc18a42def3/scratchpad")
from style import COL, LAB, isr  # noqa: E402

ROOT = "/home/user/HUJI-THESIS--YARIN/v_1/src/world_models"
OUT = "/tmp/claude-0/-home-user-HUJI-THESIS--YARIN/c76dd482-0f95-59a6-8588-5dc18a42def3/scratchpad/viz/anatomy.png"

# ----------------------------------------------------------------- regimes
TEAL = "#0e7c6b"      # entity regime accent
TEAL_TINT = "#e9f4f1"
WARM = "#b3541e"      # document regime accent
WARM_TINT = "#f9efe3"
INK = "#26282b"
MUT = "#6b6f74"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": "#c9ccd0",
    "axes.linewidth": 0.7,
    "xtick.color": MUT,
    "ytick.color": INK,
    "hatch.linewidth": 0.7,
    "svg.fonttype": "none",
    "text.hinting": "none",       # float glyph advances -> exact mono grid math
})

MONO = "DejaVu Sans Mono"

# ----------------------------------------------------------------- tidy table
df = pd.read_csv(f"{ROOT}/figures/TIDY_all_year_results.csv")
df = df[(df.metric == "spearman") & (df.target == "year") & (df.probe == "ridge")]


def cell_values(level, cleaning, pooling):
    s = df[(df.level == level) & (df.cleaning == cleaning) & (df.pooling == pooling)]
    vals = dict(zip(s.arm, s.value.astype(float)))
    tf = df[(df.level == level) & (df.cleaning == cleaning) & (df.pooling == "text")]
    vals["tfidf"] = float(tf.value.iloc[0])
    return vals


def pick_bars(vals):
    """Top-7 trained arms + the two random-init controls + TF-IDF, sorted desc."""
    trained = [a for a in vals if not isr(a) and a != "tfidf"]
    top7 = sorted(trained, key=lambda a: -vals[a])[:7]
    arms = top7 + ["random", "llama2_70b_random", "tfidf"]
    arms = sorted(set(arms), key=lambda a: vals[a])          # ascending for barh
    return arms


# ----------------------------------------------------------------- layer curves
def layers_A(arm):
    d = json.load(open(f"{ROOT}/results/eng_pls/{arm}/historical_figure.last.json"))
    p = d["per_layer"]
    return [q["nd"] for q in p], [q["test_spearman"] for q in p]


def layers_B(arm):
    d = json.load(open(f"{ROOT}/akkadian/results/probes_entity/{arm}/assyrian_ruler.ent_last.json"))
    Ls = sorted(int(k) for k in d["layers"])
    lo, hi = Ls[0], Ls[-1]
    nd = [(l - lo) / (hi - lo) if hi > lo else 0.0 for l in Ls]
    rho = [d["layers"][str(l)]["bare"]["ridge_mc"]["mc_rho"] for l in Ls]
    return nd, rho


def layers_frag(variant):
    def f(arm):
        d = json.load(open(f"{ROOT}/akkadian/results/layers_pls/{arm}/{variant}.year.mean.json"))
        p = d["per_layer"]
        return [q["nd"] for q in p], [q["test_spearman"] for q in p]
    return f


HEAD_ARMS = ["llama2_70b", "qwen3_32b", "thalesian_cunei400m"]
CTRL_ARM = "random"                       # random-init Qwen3-8B, exists everywhere

# ----------------------------------------------------------------- card specs
CARDS = [
    dict(key="A", regime="ent",
         title="A · entity · salient",
         proto="holdout",
         stim="George Washington",
         tok_last=("Washington", None),          # token text; None => find at end
         subsplit=None,
         featured="last",
         pool_last_lab="last token",
         pool_mean_lab="mean of all tokens",
         bars=cell_values("entity", "historical_figure", "last"),
         pool_tag="pooling · last token (underlined above)",
         layers=layers_A,
         verdict="trained ≫ random"),
    dict(key="B", regime="ent",
         title="B · entity · obscure",
         proto="entity_MC",
         stim="Ashurbanipal",
         tok_last=("pal", None),
         subsplit=[5, 9],                        # Ashur|bani|pal
         featured="last",
         pool_last_lab="last name token (ent_last)",
         pool_mean_lab="mean of name tokens (ent_mean)",
         bars=cell_values("entity", "rows_bare", "ent_last"),
         pool_tag="pooling · ent_last (underlined above)",
         layers=layers_B,
         verdict="obscure, yet learned"),
    dict(key="Bp", regime="doc",
         title="B′ · document · EN gloss",
         proto="ruler_MC_r8",
         stim="warrior smite with weapon ox sheep … herald of",
         tok_last=("of", None),
         subsplit=None,
         featured="mean",
         pool_last_lab="last token",
         pool_mean_lab="mean of all tokens",
         bars=cell_values("fragment", "tier0", "mean"),
         pool_tag="pooling · mean (underlined above)",
         layers=layers_frag("eng_tier0"),
         verdict="TF-IDF wins; trained ≈ random"),
    dict(key="C", regime="doc",
         title="C · document · Akkadian",
         proto="ruler_MC_r8",
         stim="lu-qu-ra-di-šu u-ra-si-bu ina ṣe-e-ni … bal",
         tok_last=("bal", None),
         subsplit=None,
         featured="mean",
         pool_last_lab="last token",
         pool_mean_lab="mean of all tokens",
         bars=cell_values("fragment", "maximal", "mean"),
         pool_tag="pooling · mean (underlined above)",
         layers=layers_frag("akk_maximal"),
         verdict="trained ≤ random; encoder holds"),
]

# ----------------------------------------------------------------- figure scaffold
FW, FH = 17.0, 11.1
fig = plt.figure(figsize=(FW, FH), facecolor="white", dpi=200)
gs = fig.add_gridspec(4, 4,
                      left=0.062, right=0.985, top=0.878, bottom=0.115,
                      height_ratios=[0.50, 2.45, 4.55, 2.05],
                      hspace=0.34, wspace=0.62)

ax_head = [fig.add_subplot(gs[0, i]) for i in range(4)]
ax_stim = [fig.add_subplot(gs[1, i]) for i in range(4)]
ax_bars = [fig.add_subplot(gs[2, i]) for i in range(4)]
ax_lay = [fig.add_subplot(gs[3, i]) for i in range(4)]

fig.canvas.draw()                       # settle positions for text metrics
rend = fig.canvas.get_renderer()

# monospace advance (axes-fraction units), one shared size for all cards.
# Measured as a width DIFFERENCE so glyph-ink vs advance bias cancels exactly.
FS0 = 10.0


def _w(n):
    t = ax_stim[0].text(0, 0, "M" * n, family=MONO, fontsize=FS0,
                        transform=ax_stim[0].transAxes)
    w = t.get_window_extent(rend).width
    t.remove()
    return w


adv0 = (_w(60) - _w(20)) / 40.0          # px per char at FS0
ax_w = ax_stim[0].get_window_extent().width
maxlen = max(len(c["stim"]) for c in CARDS)
FS = min(FS0, FS0 * (0.93 * ax_w) / (maxlen * adv0))
CW = (adv0 * FS / FS0) / ax_w            # char width, axes fraction


def accent(c):
    return TEAL if c["regime"] == "ent" else WARM


def tint(c):
    return TEAL_TINT if c["regime"] == "ent" else WARM_TINT


# ----------------------------------------------------------------- headers
for ax, c in zip(ax_head, CARDS):
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor=accent(c), edgecolor="none"))
    ax.text(0.04, 0.5, c["title"], transform=ax.transAxes, ha="left", va="center",
            fontsize=11, fontweight="bold", color="white")

# ----------------------------------------------------------------- stimulus cards
def mono_runs(ax, y, runs, fs):
    """Place monospace runs on one line, centred; return span map {tag:(x0,x1)}."""
    total = sum(len(t) for t, _tag, _st in runs)
    x = 0.5 - total * CW / 2
    spans = {"__all__": (x, x + total * CW)}
    for t, tag, st in runs:
        ax.text(x, y, t, transform=ax.transAxes, family=MONO, fontsize=fs,
                ha="left", va="baseline", clip_on=False, **st)
        if tag:
            spans[tag] = (x, x + len(t) * CW)
        x += len(t) * CW
    return spans


def split_runs(c, bold_last, base_color):
    """Runs for the stimulus string; token at end optionally bold+accent."""
    s, (tok, _), acc = c["stim"], c["tok_last"], accent(c)
    i0 = len(s) - len(tok)
    pre = s[:i0]
    runs = []
    if c["subsplit"]:                    # faint subword seams: Ashur|bani|pal
        cuts = [0] + c["subsplit"] + [len(s)]
        segs = [s[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
        for j, seg in enumerate(segs):
            last_seg = j == len(segs) - 1
            st = (dict(color=acc, fontweight="bold") if (bold_last and last_seg)
                  else dict(color=base_color))
            runs.append((seg, "tok" if last_seg else None, st))
            if not last_seg:
                runs.append(("·", None, dict(color="#b9bdc2")))
    else:
        runs.append((pre, None, dict(color=base_color)))
        st = (dict(color=acc, fontweight="bold") if bold_last
              else dict(color=base_color))
        runs.append((tok, "tok", st))
    return runs


for ax, c in zip(ax_stim, CARDS):
    ax.set_axis_off()
    acc = accent(c)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=tint(c),
                           edgecolor=acc, linewidth=0.7, alpha=1.0, zorder=0))

    featured = c["featured"]

    # ---- line 1 : last-token pooling -------------------------------------
    y1 = 0.60
    spans1 = mono_runs(ax, y1, split_runs(c, bold_last=True, base_color=INK), FS)
    xa, xb = spans1["tok"]
    ax.plot([xa, xb], [y1 - 0.055] * 2, transform=ax.transAxes, color=acc,
            lw=2.2, solid_capstyle="round", zorder=3)
    lab1 = c["pool_last_lab"] + ("  ▸ bars below" if featured == "last" else "")
    ax.text(xb, y1 - 0.155, lab1, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.9, color=acc,
            fontweight="bold" if featured == "last" else "normal",
            style="normal" if featured == "last" else "italic")

    # "probe reads here" arrow onto the underlined token
    xm = (xa + xb) / 2
    tx = min(max(xm - 0.16, 0.05), 0.52)
    ax.annotate("probe reads here", xy=(xm, y1 + 0.075), xytext=(tx, y1 + 0.29),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                fontsize=7.2, style="italic", color=MUT, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=0.9,
                                connectionstyle="arc3,rad=-0.25",
                                shrinkA=2, shrinkB=1))

    # ---- line 2 : mean pooling -------------------------------------------
    y2 = 0.26
    spans2 = mono_runs(ax, y2, split_runs(c, bold_last=False, base_color="#4b4f54"), FS)
    x0, x1 = spans2["__all__"]
    ax.plot([x0, x1], [y2 - 0.055] * 2, transform=ax.transAxes, color=acc,
            lw=1.1, alpha=0.55, solid_capstyle="round", zorder=3)
    lab2 = c["pool_mean_lab"] + ("  ▸ bars below" if featured == "mean" else "")
    ax.text(x1, y2 - 0.155, lab2, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.9, color=acc,
            fontweight="bold" if featured == "mean" else "normal",
            style="normal" if featured == "mean" else "italic")

# ----------------------------------------------------------------- bar panels
for ax, c in zip(ax_bars, CARDS):
    vals = c["bars"]
    arms = pick_bars(vals)
    ys = np.arange(len(arms))
    for y, a in zip(ys, arms):
        v = vals[a]
        if a == "tfidf":
            ax.barh(y, v, height=0.66, facecolor="#111111", edgecolor="none")
        elif isr(a):
            ax.barh(y, v, height=0.66, facecolor="white",
                    edgecolor=COL[a], hatch="////", linewidth=0.9)
        else:
            ax.barh(y, v, height=0.66, facecolor=COL[a], edgecolor="none")
        ax.text(v + 0.015, y, f"{v:.2f}".lstrip("0"), va="center", ha="left",
                fontsize=6.6, color="#55585c")
    ax.set_yticks(ys)
    ax.set_yticklabels([LAB[a] for a in arms], fontsize=7.3)
    for lab, a in zip(ax.get_yticklabels(), arms):
        lab.set_color("#6e7277" if (isr(a) or a == "tfidf") else INK)
    ax.set_xlim(0, 1.10)
    ax.set_ylim(-0.65, len(arms) - 0.35)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"], fontsize=6.6)
    ax.grid(axis="x", color="#000000", alpha=0.06, lw=0.7)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5, pad=1.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c9ccd0")
    ax.set_title(c["pool_tag"], fontsize=7.6, color=MUT, loc="left", pad=3)
    ax.set_xlabel("Spearman ρ  —  " + c["verdict"], fontsize=7.8,
                  style="italic", color=accent(c), labelpad=2.5)
    ax.text(0.985, 0.02, "protocol: " + c["proto"], transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.4, color=MUT, style="italic")

# ----------------------------------------------------------------- layer insets
ymin = 0.0
for ax, c in zip(ax_lay, CARDS):
    for arm in HEAD_ARMS + [CTRL_ARM]:
        nd, rho = c["layers"](arm)
        if arm == CTRL_ARM:
            ax.plot(nd, rho, color=COL[arm], lw=1.0, ls="--", alpha=0.85, zorder=2)
        else:
            lw = 1.9 if arm == "llama2_70b" else 1.5
            ax.plot(nd, rho, color=COL[arm], lw=lw, zorder=3)
            i = int(np.nanargmax(rho))
            ax.plot(nd[i], rho[i], "o", ms=3.4, color=COL[arm],
                    mec="white", mew=0.6, zorder=4)
        ymin = min(ymin, min(rho))
for ax, c in zip(ax_lay, CARDS):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(min(-0.05, ymin - 0.03), 1.0)
    ax.axhline(0, color="#9aa0a5", lw=0.6, alpha=0.6)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", ".5", "1"], fontsize=6.2)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", ".5", "1"], fontsize=6.2)
    ax.grid(color="#000000", alpha=0.05, lw=0.6)
    ax.tick_params(length=2, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("normalized depth", fontsize=6.8, color=MUT, labelpad=1.5)
    ax.set_title("ρ across layers · featured pooling", fontsize=7.4,
                 color=MUT, loc="left", pad=3)

# ----------------------------------------------------------------- top bands + cliff
fig.canvas.draw()
pA, pB = ax_head[0].get_position(), ax_head[1].get_position()
pBp, pC = ax_head[2].get_position(), ax_head[3].get_position()
band_y, band_h = 0.902, 0.023

for (x0, x1, col, txt) in [
        (pA.x0, pB.x1, TEAL, "ENTITY LEVEL — the signal survives"),
        (pBp.x0, pC.x1, WARM, "DOCUMENT LEVEL — the signal collapses")]:
    fig.add_artist(Rectangle((x0, band_y), x1 - x0, band_h,
                             transform=fig.transFigure, facecolor=col,
                             edgecolor="none", alpha=0.14))
    fig.text((x0 + x1) / 2, band_y + band_h / 2, txt, ha="center", va="center",
             fontsize=10.2, fontweight="bold", color=col)

xc = pB.x1 + 0.40 * (pBp.x0 - pB.x1)      # left of gap centre: clear of B' labels
cl = Line2D([xc, xc], [0.115, band_y + band_h], transform=fig.transFigure,
            color="#4a4d50", lw=1.1, ls=(0, (5, 4)), zorder=1)
fig.add_artist(cl)
fig.text(xc, 0.52, "the cliff", rotation=90, ha="center", va="center",
         fontsize=8.5, style="italic", color="#4a4d50",
         bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none"))

# ----------------------------------------------------------------- titles, legend, footnote
fig.text(0.062, 0.977, "Anatomy of the cliff: what the probe reads, and what it finds",
         ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
fig.text(0.062, 0.947,
         "Linear probes for “year” on hidden states: entity names — even obscure Assyrian rulers — carry a learned time "
         "signal; on fragments trained LLMs fall to their random-init twins and char-n-gram TF-IDF wins.",
         ha="left", va="top", fontsize=10.2, color=MUT)

handles = [
    Line2D([], [], color=COL["llama2_70b"], lw=1.9, label=LAB["llama2_70b"]),
    Line2D([], [], color=COL["qwen3_32b"], lw=1.5, label=LAB["qwen3_32b"]),
    Line2D([], [], color=COL["thalesian_cunei400m"], lw=1.5,
           label=LAB["thalesian_cunei400m"] + " (translation encoder)"),
    Line2D([], [], color=COL["random"], lw=1.0, ls="--",
           label="random-init Qwen3-8B (control)"),
    Patch(facecolor="white", edgecolor="#666", hatch="////",
          label="bars hatched = random-init controls"),
    Patch(facecolor="#111111", label="TF-IDF char-n-gram (text baseline)"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.052),
           ncol=6, frameon=False, fontsize=7.6, handlelength=1.7,
           columnspacing=1.5, handletextpad=0.6)

fig.text(0.062, 0.040,
         "Metric: Spearman ρ, ridge probe, target = year; bars use each card’s best layer at the featured pooling; top-7 arms by ρ plus controls. "
         "Protocols differ by level — A: held-out entities; B: entity Monte-Carlo (entity_MC); B′/C: ruler-blocked Monte-Carlo (ruler_MC_r8) — compare within, not across, cards.",
         ha="left", va="top", fontsize=7.2, color=MUT)
fig.text(0.062, 0.022,
         "Depth curves: per-layer ρ at the featured pooling (A/B′/C: held-out fits; B: entity-MC ρ, bare-name rows); dots mark each model’s best layer. "
         "* = baseline/control arms. Stimuli are abridged examples from the actual prompt sets.",
         ha="left", va="top", fontsize=7.2, color=MUT)

fig.savefig(OUT, dpi=200, facecolor="white")
print("saved", OUT)
