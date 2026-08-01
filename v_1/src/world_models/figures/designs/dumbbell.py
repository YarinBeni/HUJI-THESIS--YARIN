#!/usr/bin/env python3
"""DESIGN 4 -- "Trained vs its own random twin" dumbbell.

Rows = (config x model-pair). For each of the 10 canonical configs and each
trained/random-twin pair, a horizontal dumbbell: open circle = random twin,
filled circle = trained, bar green when trained>random, red when trained<random.
Entity configs get a teal regime tint, fragment configs a warm tint; TF-IDF is
a dotted vertical line per config group. All numbers come from the committed
TIDY table -- nothing fabricated.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import pandas as pd

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib')
sys.path.insert(0, SCRATCH)
from _style import COL  # per-arm hex colors (Qwen blues, Llama greens)

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


CSV = TIDY_CSV
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/dumbbell{_TAG}.png"

# ----------------------------------------------------------------- data ----
df = pd.read_csv(CSV)
d = df[(df.metric == "spearman") & (df.target == "year")]
dr = d[d.probe == "ridge"]

# canonical 10 configs, narrative order (entity -> fragment)
# (level, salience|None, cleaning, pooling, cell, title, regime)
CONFIGS = [
    ("entity", "salient", "historical_figure", "last",
     "A", "Famous historical figures (EN)", "last name token"),
    ("entity", "salient", "historical_figure", "mean",
     "A", "Famous historical figures (EN)", "mean over name tokens"),
    ("entity", "obscure", "rows_bare", "ent_last",
     "B", "Assyrian ruler names, bare (EN)", "last name token"),
    ("entity", "obscure", "rows_bare", "ent_mean",
     "B", "Assyrian ruler names, bare (EN)", "mean over name tokens"),
    ("entity", "obscure", "rows_all", "ent_last",
     "B", "Ruler name in carrier sentence (EN)", "last name token"),
    ("entity", "obscure", "rows_all", "last",
     "B", "Ruler name in carrier sentence (EN)", "last sentence token"),
    ("fragment", None, "tier0", "last",
     "B′", "English-glossed fragments (tier0)", "last token"),
    ("fragment", None, "tier0", "mean",
     "B′", "English-glossed fragments (tier0)", "mean over fragment"),
    ("fragment", None, "maximal", "last",
     "C", "Raw Akkadian fragments (maximal)", "last token"),
    ("fragment", None, "maximal", "mean",
     "C", "Raw Akkadian fragments (maximal)", "mean over fragment"),
]

# trained arm, random twin, display label -- big to small, then Qwen
PAIRS = [
    ("llama2_70b", "llama2_70b_random", "Llama-2 70B"),
    ("llama2_13b", "llama2_13b_random", "Llama-2 13B"),
    ("llama2_7b",  "llama2_7b_random",  "Llama-2 7B"),
    ("qwen3_8b",   "random",            "Qwen3 8B"),
]


def val(sub, arm):
    s = sub[sub.arm == arm]
    assert len(s) == 1, (arm, len(s), sub[["cleaning", "pooling"]].head())
    return float(s.value.iloc[0])


groups = []  # per config: dict(title, cell, regime, tfidf, rows=[(label, rnd, trn)])
for lv, sal, cl, po, cell, title, pool in CONFIGS:
    sub = dr[(dr.level == lv) & (dr.cleaning == cl) & (dr.pooling == po)]
    if sal is not None:
        sub = sub[sub.salience == sal]
    tf = dr[(dr.level == lv) & (dr.cleaning == cl) & (dr.arm == "tfidf")]
    if sal is not None:
        tf = tf[tf.salience == sal]
    rows = [(lab, val(sub, ra), val(sub, ta)) for ta, ra, lab in PAIRS]
    groups.append(dict(cell=cell, title=title, pool=pool, regime=lv,
                       tfidf=float(tf.value.iloc[0]), rows=rows))

# --------------------------------------------------------------- palette ----
TEAL, TEAL_TXT = "#0f6e64", "#0b544c"      # entity regime
WARM, WARM_TXT = "#b34a12", "#8a3a0e"      # fragment regime
GAIN, LOSS = "#1e8a3c", "#c62f21"          # dumbbell bar polarity
GRID = "#d3d3d3"

# ---------------------------------------------------------------- layout ----
HEADER_H, ROW_H, PAD, GAP, BANNER_H = 0.95, 1.0, 0.55, 1.05, 1.35

cur = 0.0
banners, layout = [], []
for gi, g in enumerate(groups):
    if gi == 0:
        banners.append((cur, cur + BANNER_H, "entity",
                        "ENTITY LEVEL — probe reads a name token"))
        cur += BANNER_H + 0.55
    if gi == 6:
        cur += 0.45
        banners.append((cur, cur + BANNER_H, "fragment",
                        "DOCUMENT LEVEL — probe reads the whole fragment"))
        cur += BANNER_H + 0.55
    top = cur
    hy = cur + 0.45
    y0 = cur + HEADER_H + PAD
    ys = [y0 + i * ROW_H for i in range(len(g["rows"]))]
    bot = ys[-1] + PAD
    layout.append(dict(top=top, hy=hy, ys=ys, bot=bot))
    cur = bot + GAP
TOTAL = cur - GAP + 0.85

# ---------------------------------------------------------------- figure ----
fig = plt.figure(figsize=(9.3, 13.9), dpi=200)
ax = fig.add_axes([0.118, 0.073, 0.802, 0.847])
ax.set_xlim(0.0, 1.0)
ax.set_ylim(TOTAL, -0.35)
ax.set_axisbelow(True)

for x in (0.2, 0.4, 0.6, 0.8):
    ax.axvline(x, color=GRID, lw=0.7, ls=":", zorder=0.4)

blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

# regime banners
for b0, b1, reg, txt in banners:
    c, ct = (TEAL, TEAL_TXT) if reg == "entity" else (WARM, WARM_TXT)
    ax.axhspan(b0, b1, color=c, alpha=0.16, lw=0, zorder=0.2)
    ax.text(0.5, (b0 + b1) / 2, txt, transform=blend, ha="center",
            va="center", fontsize=11.5, fontweight="bold", color=ct)

for g, L in zip(groups, layout):
    ent = g["regime"] == "entity"
    c, ct = (TEAL, TEAL_TXT) if ent else (WARM, WARM_TXT)

    # regime tint behind the whole config block
    ax.axhspan(L["top"], L["bot"], color=c, alpha=0.055, lw=0, zorder=0.2)
    ax.plot([0, 1], [L["top"], L["top"]], transform=blend,
            color=c, lw=1.0, alpha=0.55, zorder=0.6, solid_capstyle="butt")

    # config header (cell tag + description + pooling)
    ax.text(0.008, L["hy"], f"{g['cell']}  ·  {g['title']}",
            transform=blend, ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=ct)
    ax.text(0.992, L["hy"], g["pool"], transform=blend, ha="right",
            va="center", fontsize=8.5, style="italic", color="#666666")

    # TF-IDF floor for this config
    ax.plot([g["tfidf"]] * 2, [L["ys"][0] - 0.5, L["ys"][-1] + 0.5],
            color="black", lw=1.2, ls=(0, (1.5, 2.2)), zorder=2.5)
    ax.text(g["tfidf"], L["bot"] + 0.34, f"TF-IDF {g['tfidf']:.2f}",
            ha="center", va="center", fontsize=6.6,
            color="#555555", zorder=2.5)

    for (lab, rnd, trn), y in zip(g["rows"], L["ys"]):
        arm = [p[0] for p in PAIRS if p[2] == lab][0]
        gain = trn >= rnd
        ax.plot([rnd, trn], [y, y], color=GAIN if gain else LOSS,
                lw=3.4, solid_capstyle="round", zorder=3,
                alpha=0.95 if gain else 0.9)
        ax.plot(rnd, y, "o", ms=7.6, mfc="white", mec=COL[arm],
                mew=1.6, zorder=4)
        ax.plot(trn, y, "o", ms=7.6, mfc=COL[arm], mec="#1a1a1a",
                mew=0.7, zorder=5)
        dlt = trn - rnd
        ax.text(1.013, y, f"{dlt:+.2f}".replace("-", "−"),
                transform=blend, ha="left", va="center", fontsize=7.3,
                color=GAIN if gain else LOSS, fontweight="bold")

# y ticks = model labels per row
yticks, ylabels = [], []
for g, L in zip(groups, layout):
    for (lab, _, _), y in zip(g["rows"], L["ys"]):
        yticks.append(y)
        ylabels.append(lab)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=7.8, color="#444444")
ax.tick_params(axis="y", length=0, pad=4)
ax.tick_params(axis="x", labelsize=8.5, colors="#444444",
               top=True, labeltop=True)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel(f"Spearman ρ (held-out year decoding, {_PROBE_PHRASE})",
              fontsize=9.5, color="#222222", labelpad=4)
for s in ("left", "right"):
    ax.spines[s].set_visible(False)
for s in ("top", "bottom"):
    ax.spines[s].set_color("#999999")

# delta column header (top right, above the axes)
ax.text(1.022, 1.006, "Δρ", transform=ax.transAxes, ha="left",
        va="bottom", fontsize=9, fontweight="bold", color="#333333")

# ---------------------------------------------------- title / legend / foot --
fig.text(0.5, 0.9895,
         "The entity → document cliff: gains over random-init twins "
         "collapse at fragment level",
         ha="center", va="top", fontsize=12.5, fontweight="bold",
         color="#111111")
fig.text(0.5, 0.9745,
         f"Year decoding from frozen activations — {_PROBE_PHRASE}, Spearman ρ on held-out data; "
         "each row pairs a trained model with its own random-init twin.",
         ha="center", va="top", fontsize=8.8, color="#444444")

handles = [
    Line2D([], [], marker="o", ls="none", ms=7.5, mfc="#555555",
           mec="#1a1a1a", mew=0.7, label="trained"),
    Line2D([], [], marker="o", ls="none", ms=7.5, mfc="white",
           mec="#555555", mew=1.6, label="random-init twin"),
    Line2D([], [], color=GAIN, lw=3.4, label="trained > random"),
    Line2D([], [], color=LOSS, lw=3.4, label="trained < random"),
    Line2D([], [], color="black", lw=1.2, ls=(0, (1.5, 2.2)),
           label="char n-gram TF-IDF"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.9615),
           ncol=5, frameon=False, fontsize=8.6, handletextpad=0.5,
           columnspacing=1.4)

fig.text(0.118, 0.041,
         "Marker hue encodes model family and size (Llama-2 greens, Qwen3 blue; darker = larger); "
         "Δρ = trained − random twin.\n"
         "Protocols differ by level: A uses a holdout split over entities; B (entity) uses "
         "entity-level Monte-Carlo splits (entity_MC); fragment cells B′ and C use\n"
         "ruler-held-out Monte-Carlo with 8 rounds (ruler_MC_r8) — compare ρ within a config "
         "group, not across levels.\n"
         "Dotted line: character n-gram TF-IDF + ridge on the raw text, same protocol per "
         "config group.",
         ha="left", va="top", fontsize=6.8, color="#666666", linespacing=1.35)

fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
