#!/usr/bin/env python3
"""DESIGN 3 -- "Everything matrix".

One annotated double heatmap (ridge | PLS) over the 10 canonical configurations,
rows = 14 arms + a separated raw random-Qwen reference strip.  Cell text = raw
Spearman rho; cell colour = rho minus the column's random-init Qwen3-8B value
(RdBu_r, centered at 0).  Entity groups get teal header bands, fragment groups
warm ones; bold border = best arm per column (reference strip included).
All numbers are read live from the committed result files -- nothing hardcoded.
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from _style import COL, LAB

# The TIDY table and the output suffix are overridable so the same code renders
# both read-outs (raw = ridge everywhere; deck = the per-cell probe the thesis
# reports — PLS at fragment level, PLS-5 for obscure entities, ridge for cell A).
TIDY_CSV = os.environ.get("TIDY_CSV", "/home/user/HUJI-THESIS--YARIN/v_1/src/world_models/figures/TIDY_all_year_results.csv")
_TAG = os.environ.get("FIG_TAG", "")


ROOT = "/home/user/HUJI-THESIS--YARIN/v_1/src/world_models"
OUT = os.path.dirname(os.path.abspath(__file__)) + f"/heatmap{_TAG}.png"

# ------------------------------------------------------------------ data ----
TIDY = list(csv.DictReader(open(TIDY_CSV)))

ARMS = ["llama2_70b", "llama2_13b", "llama2_7b", "gpt_oss_120b",
        "qwen3_32b", "qwen3_8b", "qwen3_1b7",
        "thalesian_cunei400m", "thalesian_akk300m", "umt5_base",
        "llama2_70b_random", "llama2_13b_random", "llama2_7b_random",
        "tfidf"]
REF = "random"  # random-init Qwen3-8B: separate raw reference strip

# (group, level, salience, cleaning, pooling, line1, line2)
CONFIGS = [
    ("A",  "entity",   "salient", "historical_figure", "last",     "name",     "last"),
    ("A",  "entity",   "salient", "historical_figure", "mean",     "name",     "mean"),
    ("B",  "entity",   "obscure", "rows_bare",         "ent_last", "bare",     "ent-last"),
    ("B",  "entity",   "obscure", "rows_bare",         "ent_mean", "bare",     "ent-mean"),
    ("B",  "entity",   "obscure", "rows_all",          "ent_last", "sentence", "ent-last"),
    ("B",  "entity",   "obscure", "rows_all",          "last",     "sentence", "sent-last"),
    ("Bp", "fragment", None,      "tier0",             "last",     "fragment", "last"),
    ("Bp", "fragment", None,      "tier0",             "mean",     "fragment", "mean"),
    ("C",  "fragment", None,      "maximal",           "last",     "fragment", "last"),
    ("C",  "fragment", None,      "maximal",           "mean",     "fragment", "mean"),
]

def tidy_val(lvl, sal, cln, pool, arm, probe):
    for r in TIDY:
        if (r["level"] == lvl and r["cleaning"] == cln and r["probe"] == probe
                and r["metric"] == "spearman" and r["arm"] == arm
                and r["target"] == "year"):
            if sal and r["salience"] != sal:
                continue
            if arm == "tfidf":
                if r["pooling"] != "text":
                    continue
            elif r["pooling"] != pool:
                continue
            return float(r["value"])
    return None

def pls_best_k(path):
    """max over k of test_spearman inside pls_at_best_layer."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    kk = d.get("pls_at_best_layer") or {}
    vv = [v["test_spearman"] for v in kk.values() if v.get("test_spearman") is not None]
    return max(vv) if vv else None

def get_value(panel, cfg, arm):
    g, lvl, sal, cln, pool, _, _ = cfg
    if panel == "ridge":
        return tidy_val(lvl, sal, cln, pool, arm, "ridge")
    # PLS panel
    if g == "A":
        return pls_best_k(f"{ROOT}/results/eng_pls/{arm}/historical_figure.{pool}.json")
    if g == "B":
        return tidy_val(lvl, sal, cln, pool, arm, "pls5")
    var = "eng_tier0" if cln == "tier0" else "akk_maximal"
    return pls_best_k(f"{ROOT}/akkadian/results/layers_pls/{arm}/{var}.year.{pool}.json")

def build(panel):
    M = np.full((len(ARMS), len(CONFIGS)), np.nan)
    R = np.full(len(CONFIGS), np.nan)
    for j, cfg in enumerate(CONFIGS):
        for i, a in enumerate(ARMS):
            v = get_value(panel, cfg, a)
            if v is not None:
                M[i, j] = v
        rv = get_value(panel, cfg, REF)
        if rv is not None:
            R[j] = rv
    return M, R

M_R, REF_R = build("ridge")
M_P, REF_P = build("pls")

# --------------------------------------------------------------- geometry ---
GAP = 0.6
col_x, x = [], 0.0
group_of = [c[0] for c in CONFIGS]
for j, g in enumerate(group_of):
    if j > 0 and g != group_of[j - 1]:
        x += GAP
    col_x.append(x)
    x += 1.0
col_x = np.array(col_x)
XMAX = col_x[-1] + 0.5          # 11.3
n_rows = len(ARMS)
Y_REF = n_rows + 0.65           # reference strip center (gap of 0.65 above)
YLIM = (Y_REF + 0.62, -4.85)    # inverted y

GROUPS = {
    "A":  dict(color="#0b6e62", title="A · EN figures"),
    "B":  dict(color="#178a7a", title="B · Assyrian ruler names (EN)"),
    "Bp": dict(color="#b4551e", title="B′ · EN gloss"),
    "C":  dict(color="#8c2f1b", title="C · Akkadian"),
}
TEAL, WARM = "#0b6e62", "#8c2f1b"
CLIFF_X = 0.5 * (col_x[5] + 0.5 + col_x[6] - 0.5)   # middle of the B|B' gap

VLIM = 0.56
CMAP = plt.get_cmap("RdBu_r").copy()
NORM = TwoSlopeNorm(vmin=-VLIM, vcenter=0.0, vmax=VLIM)

def fmt(v):
    s = f"{v:.2f}"
    return s.replace("-0.", "−.").replace("0.", ".")

# ----------------------------------------------------------------- figure ---
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.0})
FIG_W, FIG_H = 14.9, 7.9
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200, facecolor="white")

AX_W, AX_H, AX_Y = 0.388, 0.765, 0.115
ax1 = fig.add_axes([0.100, AX_Y, AX_W, AX_H])
ax2 = fig.add_axes([0.594, AX_Y, AX_W, AX_H])

def draw_panel(ax, M, RREF, labels_left=True, side_labels=False):
    ax.set_xlim(-0.5, XMAX)
    ax.set_ylim(*YLIM)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # ---- level brackets -------------------------------------------------
    ent_lo, ent_hi = col_x[0] - 0.5, col_x[5] + 0.5
    doc_lo, doc_hi = col_x[6] - 0.5, col_x[9] + 0.5
    ax.text(0.5 * (ent_lo + ent_hi), -4.28, "ENTITY LEVEL  —  probe a name",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=TEAL)
    ax.text(0.5 * (doc_lo + doc_hi), -4.28, "DOCUMENT LEVEL  —  probe a fragment",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=WARM)
    ax.plot([ent_lo, ent_hi], [-3.82, -3.82], color=TEAL, lw=1.6,
            solid_capstyle="butt", clip_on=False)
    ax.plot([doc_lo, doc_hi], [-3.82, -3.82], color=WARM, lw=1.6,
            solid_capstyle="butt", clip_on=False)

    # ---- group header bands --------------------------------------------
    for g in ["A", "B", "Bp", "C"]:
        idx = [j for j, gg in enumerate(group_of) if gg == g]
        lo, hi = col_x[idx[0]] - 0.5, col_x[idx[-1]] + 0.5
        ax.add_patch(Rectangle((lo, -3.52), hi - lo, 0.92,
                               facecolor=GROUPS[g]["color"], edgecolor="none",
                               clip_on=False))
        ax.text(0.5 * (lo + hi), -3.06, GROUPS[g]["title"], ha="center",
                va="center", fontsize=7.3, fontweight="bold", color="white")

    # ---- column labels (two lines) -------------------------------------
    for j, cfg in enumerate(CONFIGS):
        ax.text(col_x[j], -1.98, cfg[5], ha="center", va="center",
                fontsize=6.4, color="#444444")
        ax.text(col_x[j], -1.38, cfg[6], ha="center", va="center",
                fontsize=6.4, color="#111111", fontweight="bold")
    ax.text(-0.62, -1.98, "context:", ha="right", va="center",
            fontsize=6.2, color="#888888", style="italic")
    ax.text(-0.62, -1.38, "pooling:", ha="right", va="center",
            fontsize=6.2, color="#888888", style="italic")

    # ---- matrix cells ---------------------------------------------------
    best = {}
    for j in range(len(CONFIGS)):
        cand = [(M[i, j], ("row", i)) for i in range(n_rows) if np.isfinite(M[i, j])]
        if np.isfinite(RREF[j]):
            cand.append((RREF[j], ("ref", 0)))
        if cand:
            best[j] = max(cand, key=lambda t: t[0])[1]

    for i in range(n_rows):
        for j in range(len(CONFIGS)):
            xc, yc = col_x[j], i
            v = M[i, j]
            if not np.isfinite(v):
                ax.add_patch(Rectangle((xc - 0.5, yc - 0.5), 1, 1,
                                       facecolor="#f2f2f2", edgecolor="white", lw=0.6))
                ax.text(xc, yc, "–", ha="center", va="center",
                        fontsize=6.2, color="#aaaaaa")
                continue
            d = v - RREF[j]
            rgba = CMAP(NORM(d))
            ax.add_patch(Rectangle((xc - 0.5, yc - 0.5), 1, 1,
                                   facecolor=rgba, edgecolor="white", lw=0.6))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(xc, yc, fmt(v), ha="center", va="center", fontsize=6.3,
                    color="white" if lum < 0.46 else "#111111")

    # ---- reference strip ------------------------------------------------
    for j in range(len(CONFIGS)):
        xc = col_x[j]
        ax.add_patch(Rectangle((xc - 0.5, Y_REF - 0.5), 1, 1,
                               facecolor="#e9e9e9", edgecolor="white", lw=0.6))
        if np.isfinite(RREF[j]):
            ax.text(xc, Y_REF, fmt(RREF[j]), ha="center", va="center",
                    fontsize=6.3, color="#333333")

    # ---- best-per-column bold border ------------------------------------
    for j, (kind, i) in best.items():
        yc = i if kind == "row" else Y_REF
        ax.add_patch(Rectangle((col_x[j] - 0.5, yc - 0.5), 1, 1, facecolor="none",
                               edgecolor="black", lw=1.7, zorder=6))

    # ---- family separators (wider white gaps) ---------------------------
    for ysep in (6.5, 9.5, 12.5):
        ax.plot([-0.5, XMAX], [ysep, ysep], color="white", lw=3.0, zorder=4,
                solid_capstyle="butt")

    # ---- the cliff line -------------------------------------------------
    ax.plot([CLIFF_X, CLIFF_X], [-3.72, Y_REF + 0.55], color="#555555",
            lw=1.1, ls=(0, (4, 3)), clip_on=False, zorder=5)
    ax.text(CLIFF_X, 5.0, "entity → document cliff", ha="center", va="center",
            fontsize=6.8, color="#555555", style="italic", rotation=90,
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    # ---- row labels + colour chips --------------------------------------
    if labels_left:
        for i, a in enumerate(ARMS):
            ax.scatter([-0.78], [i], marker="s", s=34, color=COL[a],
                       edgecolor="#666666", lw=0.4, clip_on=False, zorder=6)
            ax.text(-1.02, i, LAB[a], ha="right", va="center", fontsize=7.0,
                    color="#111111")
        ax.scatter([-0.78], [Y_REF], marker="s", s=34, color=COL[REF],
                   edgecolor="#666666", lw=0.4, clip_on=False, zorder=6)
        ax.text(-1.02, Y_REF, "random Qwen* (ref.)",
                ha="right", va="center", fontsize=7.0, color="#111111",
                style="italic")

    # ---- family block labels on the right side --------------------------
    if side_labels:
        for (y0, y1, lab) in [(-0.5, 6.5, "trained LLMs"),
                              (6.5, 9.5, "encoders"),
                              (9.5, 12.5, "random twins")]:
            ax.text(XMAX + 0.28, 0.5 * (y0 + y1), lab, ha="center", va="center",
                    fontsize=6.6, color="#777777", rotation=270, clip_on=False)
            ax.plot([XMAX + 0.06, XMAX + 0.06], [y0 + 0.12, y1 - 0.12],
                    color="#bbbbbb", lw=1.0, clip_on=False,
                    solid_capstyle="butt")

draw_panel(ax1, M_R, REF_R, labels_left=True)
draw_panel(ax2, M_P, REF_P, labels_left=True, side_labels=True)

# panel titles at figure level (above the axes, below the subtitle)
for ax, ttl in ((ax1, "RIDGE PROBE"), (ax2, "PLS PROBE")):
    bb = ax.get_position()
    fig.text(0.5 * (bb.x0 + bb.x1), 0.898, ttl, ha="center", va="bottom",
             fontsize=10.0, fontweight="bold", color="#111111")

# ------------------------------------------------------------ title block ---
fig.text(0.012, 0.975, "The entity → document cliff, in one matrix",
         ha="left", va="top", fontsize=13.5, fontweight="bold", color="#111111")
fig.text(0.012, 0.938,
         "Spearman ρ for probing YEAR from hidden states — cell text: raw ρ;  "
         "cell colour: ρ − ρ(random-init Qwen3-8B) in the same column (grey strip, bottom);  "
         "black frame: best arm per column.",
         ha="left", va="top", fontsize=8.2, color="#444444")

# ------------------------------------------------------------- colourbar ----
cax = fig.add_axes([0.100, 0.052, 0.185, 0.020])
sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
cb.ax.tick_params(labelsize=6.5, length=2.5, pad=1.5)
cb.outline.set_linewidth(0.4)
cax.set_title("Δρ vs random-init Qwen3-8B (blue = below random)",
              fontsize=6.8, color="#444444", pad=3)

# -------------------------------------------------------------- footnotes ---
fig.text(0.318, 0.094,
         "Protocols differ by level: A = holdout;  B = entity-level MC (entity_MC);  "
         "B′/C = leave-rulers-out MC, r = 8 (ruler_MC_r8) — ρ comparable within, not across, groups.",
         ha="left", va="top", fontsize=6.5, color="#555555")
fig.text(0.318, 0.068,
         "PLS panel: A, B′, C = best-k PLS at best layer (layer sweeps);  B = PLS(k = 5), "
         "same protocol as ridge;  “–” = not run (70B random twin in A; TF-IDF outside B).",
         ha="left", va="top", fontsize=6.5, color="#555555")
fig.text(0.318, 0.042,
         "* = untrained / non-neural baselines.   Pooling “ent-” = over name tokens only.   "
         "Target = year;  metric = Spearman ρ.",
         ha="left", va="top", fontsize=6.5, color="#555555")

fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT, os.path.getsize(OUT), "bytes")
