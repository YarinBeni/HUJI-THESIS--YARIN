#!/usr/bin/env python3
"""DESIGN 2 -- "The cliff slopegraph".

Four narrative stages (A entity-salient, B entity-obscure, B' fragment-EN-gloss,
C fragment-Akkadian). y = Delta Spearman rho vs the random-init Qwen3-8B control
measured in the identical configuration; each arm is shown at its best-scoring
pooling per stage (marker shape annotates which). Entity stages tinted teal,
fragment stages tinted warm; the regime switch is an annotated CLIFF band.
All numbers come straight from the committed TIDY table -- nothing fabricated.
"""
import csv
import os, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from _style import COL, LAB, isr
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
OUT = os.path.dirname(os.path.abspath(__file__)) + f"/slopegraph{_TAG}.png"

# ---------------------------------------------------------------- data ------
rows = list(csv.DictReader(open(TIDY)))
V = {}  # (level, salience, cleaning, pooling, arm) -> spearman (ridge)
for r in rows:
    if r["metric"] == "spearman" and r["probe"] == "ridge" and r["target"] == "year":
        V[(r["level"], r["salience"], r["cleaning"], r["pooling"], r["arm"])] = float(r["value"])

# canonical configs per stage (level, salience, cleaning, pooling)
STAGES = [
    ("A", [("entity", "salient", "historical_figure", "last"),
           ("entity", "salient", "historical_figure", "mean")]),
    ("B", [("entity", "obscure", "rows_bare", "ent_last"),
           ("entity", "obscure", "rows_bare", "ent_mean"),
           ("entity", "obscure", "rows_all", "ent_last"),
           ("entity", "obscure", "rows_all", "last")]),
    ("B'", [("fragment", "obscure", "tier0", "last"),
            ("fragment", "obscure", "tier0", "mean")]),
    ("C", [("fragment", "obscure", "maximal", "last"),
           ("fragment", "obscure", "maximal", "mean")]),
]
TFIDF_CFG = {  # pooling='text' rows per stage
    "A": [("entity", "salient", "historical_figure", "text")],
    "B": [("entity", "obscure", "rows_bare", "text"), ("entity", "obscure", "rows_all", "text")],
    "B'": [("fragment", "obscure", "tier0", "text")],
    "C": [("fragment", "obscure", "maximal", "text")],
}
ARMS = ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b",
        "llama2_7b", "llama2_13b", "llama2_70b",
        "thalesian_akk300m", "thalesian_cunei400m", "umt5_base",
        "llama2_7b_random", "llama2_13b_random", "llama2_70b_random"]

best = defaultdict(dict)   # arm -> stage -> (delta, raw, cfg)
for s, cfgs in STAGES:
    for m in ARMS:
        cands = [(V[c + (m,)], c) for c in cfgs if c + (m,) in V]
        assert len(cands) == len(cfgs), (s, m)
        raw, cfg = max(cands, key=lambda t: t[0])
        ref = V[cfg + ("random",)]
        best[m][s] = (raw - ref, raw, cfg)
    # TF-IDF: no matched 'text' config for the random arm -> reference against
    # random's best pooling within the stage (the conservative choice).
    traw, tcfg = max((V[c + ("tfidf",)], c) for c in TFIDF_CFG[s])
    rref = max(V[c + ("random",)] for c in cfgs)
    best["tfidf"][s] = (traw - rref, traw, tcfg)

S = [s for s, _ in STAGES]
X = np.arange(4.0)

# ------------------------------------------------------------- palette ------
TEAL, TEAL_D = "#0f766e", "#0b5d57"      # entity regime
WARM, WARM_D = "#c2601a", "#9a3412"      # fragment regime
INK, MUT, FAINT = "#1c1c1c", "#6d6d6d", "#9a9a9a"
GRAYS = {"llama2_7b_random": "#b0b0b0", "llama2_13b_random": "#909090",
         "llama2_70b_random": "#6f6f6f"}
P2M = {"last": "o", "mean": "s", "ent_last": "^", "ent_mean": "D"}

# ------------------------------------------------------------- figure -------
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8,
                     "text.color": INK, "axes.edgecolor": "#c9c9c9"})
fig = plt.figure(figsize=(13.4, 9.3), dpi=110)
ax = fig.add_axes([0.058, 0.335, 0.922, 0.565])

YLO, YHI = -0.175, 0.525
ax.set_xlim(-0.45, 4.62)
ax.set_ylim(YLO, YHI)

# regime tints + cliff gradient band
ax.axvspan(-0.45, 1.38, color=TEAL, alpha=0.055, lw=0, zorder=0)
ax.axvspan(1.62, 3.32, color=WARM, alpha=0.055, lw=0, zorder=0)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cliff", [TEAL, WARM])
ax.imshow(np.linspace(0, 1, 128)[None, :], extent=[1.38, 1.62, YLO, YHI],
          cmap=cmap, alpha=0.16, aspect="auto", zorder=0)
for xe in (1.38, 1.62):
    ax.axvline(xe, color="#ffffff", lw=1.2, zorder=1)

# recessive grid + stage rules (kept out of the label gutter)
GXR = 3.32
for gy in np.arange(-0.1, 0.5, 0.1):
    ax.plot([-0.45, GXR], [gy, gy], color="#000000", alpha=0.06, lw=0.7, zorder=1)
for xs in X:
    ax.axvline(xs, color="#000000", alpha=0.08, lw=0.9, zorder=1)
ax.plot([-0.45, GXR], [0, 0], color="#4d4d4d", lw=1.5, zorder=2)
ax.text(-0.41, -0.012, "Δρ = 0  ·  random-init Qwen3-8B, matched config",
        fontsize=8.2, color="#4d4d4d", va="top", ha="left", zorder=2)

ax.set_xticks([])
ax.set_yticks(np.arange(-0.1, 0.5, 0.1))
ax.set_yticklabels([f"{v:+.1f}".replace("-", "−") if abs(v) > 1e-9 else "0"
                    for v in np.arange(-0.1, 0.5, 0.1)], fontsize=9, color=MUT)
ax.set_ylabel("Δ Spearman ρ   (arm − random-init control, same configuration)",
              fontsize=10.5, color=INK, labelpad=6)
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0)

# ------------------------------------------------------------- lines --------
def series(m):
    return [best[m][s][0] for s in S]

for m in ARMS + ["tfidf"]:
    y = series(m)
    if m == "tfidf":
        ax.plot(X, y, color="#000000", lw=3.0, ls=(0, (1.5, 1.6)), zorder=4.5,
                solid_capstyle="round")
        continue
    if isr(m):
        ax.plot(X, y, color=GRAYS[m], lw=1.3, ls=(0, (5, 3)), zorder=3)
    else:
        ax.plot(X, y, color=COL[m], lw=2.1, alpha=0.95, zorder=4)
    for xi, s in zip(X, S):
        cfg = best[m][s][2]
        mk = P2M[cfg[3]]
        open_mk = (cfg[2] == "rows_all")  # name read inside a sentence
        kw = dict(marker=mk, zorder=5.2, linestyle="none")
        if isr(m):
            kw.update(ms=4.2, mfc=GRAYS[m], mec="white", mew=0.8)
            if open_mk:
                kw.update(mfc="white", mec=GRAYS[m], mew=1.1)
        else:
            kw.update(ms=7.0 if mk != "D" else 6.0, mfc=COL[m], mec="white", mew=1.0)
            if open_mk:
                kw.update(mfc="white", mec=COL[m], mew=1.5)
        ax.plot([xi], [best[m][s][0]], **kw)

# --------------------------------------------------- direct end labels ------
ends = [(m, best[m]["C"][0]) for m in ARMS + ["tfidf"]]
ends.sort(key=lambda t: t[1])
gap = 0.0255
pos, p = {}, YLO + 0.045
for m, v in ends:
    p = max(v, p)
    pos[m] = p
    p += gap
VXR = 4.585
for m, v in ends:
    yl = pos[m]
    lead_c = "#000000" if m == "tfidf" else (GRAYS[m] if isr(m) else COL[m])
    ax.plot([3.045, 3.38], [v, yl], color=lead_c, lw=1.1, alpha=0.85,
            zorder=3.5, solid_capstyle="round",
            ls="-" if not isr(m) else (0, (4, 2.5)))
    vtxt = f"{v:+.2f}".replace("-", "−")
    if m == "tfidf":
        ax.text(3.44, yl, LAB[m], fontsize=9.6, va="center", ha="left",
                color="#000000", fontweight="bold")
        ax.text(VXR, yl, vtxt, fontsize=8.8, va="center", ha="right",
                color="#000000", fontweight="bold")
    elif isr(m):
        ax.text(3.44, yl, LAB[m], fontsize=8.2, va="center", ha="left",
                color="#7a7a7a", style="italic")
        ax.text(VXR, yl, vtxt, fontsize=8.0, va="center", ha="right",
                color="#9a9a9a", style="italic")
    else:
        ax.text(3.44, yl, LAB[m], fontsize=9.2, va="center", ha="left",
                color=INK, fontweight="bold")
        ax.text(VXR, yl, vtxt, fontsize=8.6, va="center", ha="right",
                color=MUT)
ax.text(VXR, max(pos.values()) + 0.036, "Δρ at C", fontsize=7.8, ha="right",
        va="center", color=FAINT, style="italic")

# --------------------------------------------- regime + cliff headlines -----
def tracked(sr):
    return " ".join(sr)

ax.text(0.5, 0.505, tracked("ENTITY LEVEL"), fontsize=12, fontweight="bold",
        color=TEAL_D, ha="center", va="top")
ax.text(0.5, 0.468, "a name is the stimulus · linear time survives",
        fontsize=8.4, color=TEAL_D, alpha=0.85, ha="center", va="top")
ax.text(2.5, 0.505, tracked("DOCUMENT LEVEL"), fontsize=12, fontweight="bold",
        color=WARM_D, ha="center", va="top")
ax.text(2.5, 0.468, "a fragment is the stimulus · learned signal collapses",
        fontsize=8.4, color=WARM_D, alpha=0.85, ha="center", va="top")
ax.text(1.5, 0.505, tracked("THE CLIFF"), fontsize=11, fontweight="bold",
        color="#3d3d3d", ha="center", va="top")
ax.text(1.5, 0.468, "entity → document", fontsize=8.4, color="#3d3d3d",
        ha="center", va="top")
ax.annotate("", xy=(1.62, 0.428), xytext=(1.38, 0.428),
            arrowprops=dict(arrowstyle="-|>", color="#3d3d3d", lw=1.1))

# ------------------------------------------------ bottom typography ---------
W, H = fig.get_size_inches()
renderer = fig.canvas.get_renderer()
FW, FH = fig.bbox.width, fig.bbox.height

def _w(txt, size, weight="normal", style="normal"):
    t = fig.text(0.5, 0.5, txt, fontsize=size, fontweight=weight, style=style)
    w = t.get_window_extent(renderer).width
    t.remove()
    return w / FW

def strip_line(xc, y, segments, size, u_color, base_color=INK):
    """Centered piecewise text; kind 'u' underlined in regime color, 'm' muted."""
    sp = _w("a a", size) - _w("aa", size)
    widths = []
    for txt, kind in segments:
        core = txt.strip(" ")
        w = _w(core, size) if core else 0.0
        w += sp * (len(txt) - len(txt.lstrip(" "))) + sp * (len(txt) - len(txt.rstrip(" ")))
        widths.append(w)
    x = xc - sum(widths) / 2.0
    for (txt, kind), w in zip(segments, widths):
        col = {"n": base_color, "u": base_color, "m": FAINT}[kind]
        fig.text(x, y, txt, fontsize=size, ha="left", va="baseline", color=col,
                 style="italic" if kind == "m" else "normal")
        if kind == "u":
            fig.add_artist(Line2D([x, x + w], [y - 0.0075, y - 0.0075],
                                  transform=fig.transFigure, color=u_color,
                                  lw=1.4, solid_capstyle="butt", clip_on=False))
        x += w

def fx(xd):
    return ax.transData.transform((xd, 0))[0] / FW

HEAD = [("A", "salient entities · EN", "protocol: holdout", TEAL_D),
        ("B", "obscure ruler names · EN", "protocol: entity Monte-Carlo", TEAL_D),
        ("B′", "fragments · EN gloss", "protocol: ruler-grouped MC (r = 8)", WARM_D),
        ("C", "fragments · Akkadian", "protocol: ruler-grouped MC (r = 8)", WARM_D)]
STIM = {
    0: [[("“George ", "n"), ("Washington", "u"), ("”", "n")]],
    1: [[("name alone   ", "m"), ("“", "n"), ("Ashurbanipal", "u"), ("”", "n")],
        [("in sentence   ", "m"), ("“…the reign of ", "n"),
         ("Ashurbanipal", "u"), (".”", "n")]],
    2: [[("“", "n"), ("warrior smite with weapon", "u")],
        [("ox sheep … herald of", "u"), ("”", "n")]],
    3: [[("“", "n"), ("lu-qu-ra-di-šu u-ra-si-bu", "u")],
        [("ina ṣe-e-ni … bal", "u"), ("”", "n")]],
}
Y_TITLE, Y_PROTO, Y_S1, Y_S2 = 0.286, 0.264, 0.228, 0.200
for i, (code, name, proto, ccol) in enumerate(HEAD):
    xc = fx(X[i])
    u_col = TEAL_D if i < 2 else WARM_D
    fig.text(xc, Y_TITLE, f"{code} · {name}", fontsize=10.6, ha="center",
             va="baseline", color=ccol, fontweight="bold")
    fig.text(xc, Y_PROTO, proto, fontsize=7.4, ha="center", va="baseline",
             color=FAINT, style="italic")
    lines = STIM[i]
    ys = [Y_S1] if len(lines) == 1 else [Y_S1, Y_S2]
    if len(lines) == 1:
        ys = [(Y_S1 + Y_S2) / 2]
    for ln, yy in zip(lines, ys):
        strip_line(xc, yy, ln, 8.6, u_col)

# key + footnotes
kx = 0.058
fig.text(kx, 0.150,
         "marker = winning pooling per arm × stage:   ● last token    "
         "■ mean over tokens    ▲ name-token last    ◆ name-token mean    "
         "open marker = name read inside a sentence     "
         "underline = span the probe pools",
         fontsize=8.4, color="#4a4a4a", ha="left", va="baseline")
fig.text(kx, 0.124,
         f"{_PROBE_PHRASE_CAP} on hidden states · target = year · Spearman ρ.   "
         "Δρ = arm − random-init Qwen3-8B in the identical configuration; each "
         "arm is drawn at its best-scoring pooling per stage (marker shape).\n"
         "TF-IDF (char n-grams, no pooling) is referenced against random’s "
         "best pooling per stage — the conservative comparison.   * = control "
         "arm.   Random-init Llama twins are dashed grey;\n"
         "at entity level their tokenizer/architecture alone already beats the "
         "Qwen control.   Protocols differ by level — A: i.i.d. holdout · "
         "B: entity-level Monte-Carlo · B′/C: ruler-grouped\n"
         "Monte-Carlo (r = 8) — so compare Δρ within a stage; absolute ρ is "
         "not comparable across stages.",
         fontsize=7.8, color=MUT, ha="left", va="top", linespacing=1.55)

# title
fig.text(0.058, 0.962, "Linear time probes fall off a cliff at the "
         "entity → document boundary", fontsize=17.5, fontweight="bold",
         ha="left", va="baseline", color=INK)
fig.text(0.058, 0.932,
         "Δ Spearman ρ for regnal-year decoding vs a matched random-init "
         "control · Gurnee–Tegmark world-model probing extended from English "
         "entities to Neo-Assyrian documents",
         fontsize=10.2, ha="left", va="baseline", color=MUT)

_save_fig(fig, OUT)
print("saved", OUT)

# console audit of every plotted number
for m in ARMS + ["tfidf"]:
    print(m, [(s, round(best[m][s][0], 3), best[m][s][2][2] + "/" + best[m][s][2][3])
              for s in S])
