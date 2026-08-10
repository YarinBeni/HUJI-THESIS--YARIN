"""Paper-grade figure suite for phase 2 — every number is read from committed
result files; nothing is typed in by hand.

Figure grammar follows Gurnee & Tegmark 2023 (layerwise probe curves,
predicted-vs-actual scatters, small multiples) and El-Shangiti et al. 2025
(steering curves vs alpha), adapted to the phase-2 story:

  fig1  the dissociation      — E1 pairwise accuracy per arm vs floor, akk/eng
  fig2  orthogonal axes       — E3 cos(name-axis, doc-axis) vs chance band
  fig3  lens spectroscopy     — F21 z-heatmaps, cell-A vs doc direction
  fig4  entity-gated features — F8/F11/F22 firing rates across populations
  fig5  feature causality     — F23 amplify curves + the bridge
  fig6  it's all form         — F15/F16/F17/F27 decomposition panels
  fig7  the ignition null     — F26 name-span clamps, treated vs control

    python make_figures.py            # writes figs/out/*.png (300 dpi)
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_P2 = os.path.abspath(os.path.join(_HERE, ".."))
OUT = os.path.join(_HERE, "out")
os.makedirs(OUT, exist_ok=True)

# ---- palette (dataviz reference instance; roles fixed, never cycled) -------
BLUE = "#2a78d6"      # trained model 1 (olmo)
ORANGE = "#eb6834"    # trained model 2 (qwen)
AQUA = "#1baf7a"      # highlight / treatment
GRAY = "#52514e"      # twins / controls
INK = "#0b0b0b"
MUT = "#52514e"
SURF = "#fcfcfb"
GRID = "#e3e2de"
DIVERGE = LinearSegmentedColormap.from_list(
    "bwo", [ORANGE, "#f6f5f2", BLUE])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "font.size": 9.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "DejaVu Sans",
})


def J(*parts):
    with open(os.path.join(_P2, *parts)) as f:
        return json.load(f)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


ARM_LABEL = {
    "olmo2_7b": "OLMo-2 7B", "qwen3_8b": "Qwen3 8B",
    "llama2_7b": "Llama-2 7B", "llama2_13b": "Llama-2 13B",
    "llama2_70b": "Llama-2 70B", "qwen3_1b7": "Qwen3 1.7B",
    "qwen3_32b": "Qwen3 32B", "gpt_oss_120b": "gpt-oss 120B",
    "olmo2_7b_random": "OLMo twin (random)",
    "llama2_13b_random": "Llama-13B twin", "llama2_70b_random":
    "Llama-70B twin", "llama2_7b_random": "Llama-7B twin",
    "random": "random (qwen init)", "tfidf_char": "char n-gram floor",
}


# ============================ fig 1 — dissociation ==========================
def fig1():
    sp = pd.read_csv(os.path.join(_P2, "pairs", "results",
                                  "summary_probes.csv"))
    sp = sp[(sp.site == "mean") & (sp.m == 21)]     # headline config (F3's
    # m=100 robustness rows would duplicate arms)
    inf = {v: J("pairs", "results", "inference", f"{v}.json")
           for v in ("akk_maximal", "eng_tier0")}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharex=True)
    for ax, var, title in zip(
            axes, ("akk_maximal", "eng_tier0"),
            ("Akkadian (raw transliteration)", "English gloss")):
        d = sp[sp.variant == var].sort_values("macro_acc")
        floor = float(d[d.method == "tfidf_char"].macro_acc.iloc[0])
        ax.axvline(floor, color=INK, lw=1.1, ls=(0, (4, 3)))
        ax.axvline(0.5, color=GRID, lw=1)
        ys = np.arange(len(d))
        for y, (_, r) in zip(ys, d.iterrows()):
            twin = "random" in r.method
            trained = r.method in ("olmo2_7b", "qwen3_8b")
            c = BLUE if r.method == "olmo2_7b" else \
                ORANGE if r.method == "qwen3_8b" else \
                INK if r.method == "tfidf_char" else GRAY
            ax.errorbar(r.macro_acc, y, xerr=r.macro_sd, fmt="o",
                        ms=6.5 if trained else 5.5, color=c,
                        mfc="white" if twin else c, mec=c, mew=1.4,
                        elinewidth=1.1, capsize=0, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([ARM_LABEL.get(m, m) for m in d.method])
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_xlabel("pairwise ordering accuracy (macro over ruler pairs)")
        # permutation p annotations from the E8 files
        for m, col in (("olmo2_7b", BLUE), ("qwen3_8b", ORANGE),
                       ("tfidf_char", INK)):
            a = inf[var]["arms"].get(m)
            if a is None or m not in list(d.method):
                continue
            y = float(np.where(d.method == m)[0][0])
            p = a["permutation"]["p_value"]
            ax.annotate(f"p={p:.3g}", (float(
                d[d.method == m].macro_acc.iloc[0]) +
                float(d[d.method == m].macro_sd.iloc[0]) + .004, y),
                va="center", fontsize=8, color=col)
        style_ax(ax)
    for ax, var in zip(axes, ("akk_maximal", "eng_tier0")):
        fl = float(sp[(sp.variant == var) &
                      (sp.method == 'tfidf_char')].macro_acc.iloc[0])
        ax.annotate("floor", (fl, ax.get_ylim()[1] + .1), fontsize=8,
                    color=INK, ha="center", annotation_clip=False)
    hollow = Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY,
                    color=GRAY, label="hollow = random-weight twin")
    fig.legend(handles=[hollow], loc="upper right", frameon=False,
               fontsize=8, bbox_to_anchor=(0.995, 0.955))
    fig.suptitle("E1/E8 — ordering fragments: no model beats the surface floor"
                 " in Akkadian; trained models carry the only significant"
                 " signal in English", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_dissociation.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ========================= fig 2 — orthogonal axes ==========================
def fig2():
    rows = []
    for f in glob.glob(os.path.join(_P2, "transfer", "results",
                                    "*.mean.json")):
        d = json.load(open(f))
        cos = d.get("cosine_vs_pairwise_direction", {})
        cosv = [abs(v["cosine"]) for v in cos.values()
                if isinstance(v, dict) and "cosine" in v]
        rows.append({"method": d["method"], "variant": d["variant"],
                     "cos": max(cosv) if cosv else np.nan,
                     "frozen_rho": d["frozen"]["spearman"],
                     "frozen_pw": d["frozen"]["pairwise_macro"]})
    t = pd.DataFrame(rows)
    t = t[t.method != "olmo2_7b_random"]
    chance = 1 / np.sqrt(4096)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    t1 = t.sort_values(["method", "variant"]).reset_index(drop=True)
    ys = np.arange(len(t1))
    ax.axvspan(0, chance, color=GRID, alpha=.7, lw=0)
    for y, (_, r) in zip(ys, t1.iterrows()):
        c = BLUE if r.method == "olmo2_7b" else \
            ORANGE if r.method == "qwen3_8b" else GRAY
        ax.plot([0, r.cos], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.plot(r.cos, y, "o", ms=6, color=c)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{ARM_LABEL.get(r.method, r.method)} · "
                        f"{'akk' if 'akk' in r.variant else 'eng'}"
                        for _, r in t1.iterrows()], fontsize=8.5)
    ax.annotate(f"chance for random directions in d=4096 (1/√d = "
                f"{chance:.3f})", (chance + .0005, len(t1) - .5), fontsize=8,
                color=MUT)
    ax.set_xlabel("|cos(entity time axis, document order axis)|")
    ax.set_title("The two axes are orthogonal", fontsize=10.5)
    style_ax(ax)
    ax = axes[1]
    ax.axhline(0, color=GRID, lw=1)
    for i, (_, r) in enumerate(t1.iterrows()):
        c = BLUE if r.method == "olmo2_7b" else \
            ORANGE if r.method == "qwen3_8b" else GRAY
        ax.bar(i, r.frozen_rho, width=.62, color=c)
    ax.set_xticks(range(len(t1)))
    ax.set_xticklabels([f"{r.method.split('_')[0]}\n"
                        f"{'akk' if 'akk' in r.variant else 'eng'}"
                        for _, r in t1.iterrows()], fontsize=8)
    ax.set_ylabel("Spearman ρ of frozen transfer")
    ax.set_ylim(-1, 1)
    ax.set_title("Frozen cell-A probe transfers nothing", fontsize=10.5)
    style_ax(ax)
    fig.suptitle("E3 — the entity time axis does not order documents:"
                 " transfer ≈ 0 and the axes are numerically orthogonal",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_orthogonal.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ========================= fig 3 — lens spectroscopy ========================
def fig3():
    methods = ["olmo2_7b", "llama2_7b", "qwen3_8b"]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.6), sharex=True,
                             sharey=True)
    cats = None
    for j, m in enumerate(methods):
        d = J("traces", "results", f"spectroscopy.{m}.json")
        cats = d["cats"]
        for i, (dname, dlabel) in enumerate(
                (("cellA", "entity time axis (cell A)"),
                 ("pairwise_doc", "document order axis (E1)"))):
            ax = axes[i, j]
            z = np.array(d["directions"][dname]["cos"]["z_scores"]).T
            im = ax.imshow(z, cmap=DIVERGE, vmin=-7, vmax=7,
                           aspect="auto", interpolation="nearest")
            sig = np.argwhere(np.abs(z) >= 3.35)
            if len(sig):
                ax.scatter(sig[:, 1], sig[:, 0], s=14, facecolors="none",
                           edgecolors=INK, linewidths=1.1)
            if i == 0:
                ax.set_title(ARM_LABEL.get(m, m), fontsize=10)
            if j == 0:
                ax.set_ylabel(dlabel, fontsize=9)
                ax.set_yticks(range(len(cats)))
                ax.set_yticklabels([c.replace("_", " ") for c in cats],
                                   fontsize=7.5)
            ax.set_xticks([0, 4, 9])
            ax.set_xticklabels(["1\n(early end)", "5", "10\n(late end)"],
                               fontsize=7.5)
            ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=.75, pad=.015)
    cb.set_label("z vs 50 random directions (cos variant)", fontsize=8.5)
    fig.suptitle("F21 — whole-vocabulary spectroscopy: the entity axis is"
                 " enriched in ancient vocabulary at its early end in every"
                 " model (○ = |z| ≥ 3.35, Bonferroni); the document"
                 " axis is spectrum-flat", fontsize=11, y=0.99)
    fig.savefig(os.path.join(OUT, "fig3_spectroscopy.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ======================== fig 4 — entity-gated features =====================
def fig4():
    tf1 = J("sae", "results", "token_firing.layer24.json")[
        "median_fired_anywhere"]
    fh1 = pd.read_csv(os.path.join(_P2, "sae", "results",
                                   "feature_hunt.layer24.csv"))
    p2 = J("sae2", "results", "pipeline.json")
    tf2 = {k: v["median_fired_anywhere"] for k, v in p2["step4"].items()}
    last1 = {"cellA_entities": float(fh1.fire_cellA.median()),
             "eng_tier0_frags": float(fh1.fire_eng_tier0.median()),
             "akk_maximal_frags": float(fh1.fire_akk_maximal.median())}
    pops = ["cellA_entities", "eng_tier0_frags", "akk_maximal_frags"]
    plabels = ["entity\nprompts", "English\nglosses", "Akkadian\nfragments"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7))
    panels = [
        ("last-token firing\n(Qwen-Scope, layer 24)", last1),
        ("fired anywhere in text\n(Qwen-Scope, layer 24)", tf1),
        ("fired anywhere in text\n(Karvonen 65k, layer 9)", tf2),
    ]
    for ax, (title, data) in zip(axes, panels):
        vals = [100 * float(data.get(p, np.nan)) for p in pops]
        bars = ax.bar(range(3), vals, width=.6,
                      color=[AQUA, BLUE, ORANGE])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                txt = f"{v:.2f}%" if v < 1 else f"{v:.1f}%"
                ax.annotate(txt, (b.get_x() + b.get_width() / 2,
                            v + 1.2), ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(plabels, fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=9.5)
        style_ax(ax)
    axes[0].set_ylabel("median firing rate of top-50 year features (%)")
    fig.suptitle("F8/F11/F22 — the year features are entity-gated: they fire"
                 " inside English documents but barely at the read-out token,"
                 " and (in a sparse basis) never engage Akkadian",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_gating.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ========================= fig 5 — feature causality ========================
def fig5():
    st = J("sae2", "results", "steer.layer9.json")
    hunt = pd.read_csv(sorted(glob.glob(os.path.join(
        _P2, "sae2", "results", "feature_hunt2.layer*.csv")))[-1])
    rho = dict(zip(hunt.feature.astype(int), hunt.rho_year))
    alphas = [-8, -4, -2, 0, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ctrl_curves = np.array([[st["runs"][f"ctrl:{f}"]["amplify"][str(a)]
                             for a in alphas] for f in st["ctrl"]])
    ax.fill_between(alphas, ctrl_curves.min(0), ctrl_curves.max(0),
                    color=GRAY, alpha=.22, lw=0,
                    label="rate-matched control band")
    for f in st["treat"]:
        cur = [st["runs"][f"treat:{f}"]["amplify"][str(a)] for a in alphas]
        c = BLUE if rho.get(int(f), 0) > 0 else ORANGE
        ax.plot(alphas, cur, "-o", ms=4, lw=1.8, color=c)
        ax.annotate(str(f), (alphas[-1] + .25, cur[-1]), fontsize=7.5,
                    color=c, va="center")
    ax.set_xlabel("clamp strength α (× act95)")
    ax.set_ylabel("frozen year probe read-out (sd of death year)")
    ax.set_title("Amplify at entity prompts: read-out moves in the sign"
                 " of each feature's ρ", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_ax(ax)
    ax = axes[1]
    groups = [("treat", "onomastic features", AQUA),
              ("ctrl", "matched controls", GRAY)]
    for gi, (g, lab, c) in enumerate(groups):
        f0 = [st["runs"][f"{g}:{f}"]["bridge"]["0"]["fire_last"]
              for f in st[g]]
        f4 = [st["runs"][f"{g}:{f}"]["bridge"]["4"]["fire_last"]
              for f in st[g]]
        x = np.arange(len(f0)) + gi * (len(f0) + 1)
        ax.bar(x - .19, 100 * np.array(f0), width=.38, color=c, alpha=.45,
               label=f"{lab}: α=0" if gi < 2 else None)
        ax.bar(x + .19, 100 * np.array(f4), width=.38, color=c,
               label=f"{lab}: mid-text clamp α=4" if gi < 2 else None)
    ax.set_xticks([2, 8])
    ax.set_xticklabels(["treated\n(top-5 year features)",
                        "controls\n(rate-matched)"], fontsize=8.5)
    ax.set_ylabel("% of glosses firing at the LAST token")
    ax.set_title("The bridge: clamping mid-text changes last-token firing"
                 " not at all", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), ncol=1)
    ax.set_ylim(0, 47)
    style_ax(ax)
    fig.suptitle("F23 — features causally feed the entity read-out, but no"
                 " intervention makes the signal propagate to the document"
                 " read-out", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_causal.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# =========================== fig 6 — it's all form ==========================
def fig6():
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.8))
    # (a) Esarhaddon
    ax = axes[0]
    arms = [("olmo2_7b", "eng_tier0"), ("olmo2_7b_random", "eng_tier0"),
            ("qwen3_8b", "eng_tier0"), ("qwen3_8b", "akk_maximal")]
    raw, part, names = [], [], []
    base = None
    for m, v in arms:
        d = J("esarhaddon", "results", f"{m}.{v}.json")
        lc = d["length_control"]
        raw.append(d["probe"]["oof_spearman"])
        part.append(lc["probe_partial_rho_length_out"])
        base = lc["length_only_baseline_rho"]
        names.append(f"{'OLMo' if 'olmo' in m else 'Qwen'}"
                     f"{' twin' if 'random' in m else ''}\n"
                     f"{'eng' if 'eng' in v else 'akk'}")
    x = np.arange(len(arms))
    ax.bar(x - .2, raw, width=.4, color=BLUE, label="raw probe ρ")
    ax.bar(x + .2, part, width=.4, color=GRAY,
           label="partial ρ (length out)")
    ax.axhline(base, color=INK, ls=(0, (4, 3)), lw=1.1)
    ax.annotate(f"length-only baseline ({base:.2f})", (len(arms) - .4,
                base + .012), fontsize=7.5, ha="right", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("within-Esarhaddon year ρ")
    ax.set_title("F15: 'identity-free chronology'\nis length encoding",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    style_ax(ax)
    # (b) seriation
    ax = axes[1]
    arms = ["olmo2_7b", "qwen3_8b", "olmo2_7b_random", "tfidf_char"]
    ry, rl = [], []
    for m in arms:
        d = J("seriation", "results", f"{m}.eng_tier0.json")
        ry.append(d["rho_year_abs"])
        rl.append(d["rho_length_abs"])
    x = np.arange(len(arms))
    ax.bar(x - .2, ry, width=.4, color=BLUE, label="|ρ| vs year")
    ax.bar(x + .2, rl, width=.4, color=GRAY, label="|ρ| vs length")
    ax.set_xticks(x)
    ax.set_xticklabels(["OLMo", "Qwen", "twin", "floor"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("F16: the cloud's natural order\ntracks length, not year",
                 fontsize=9)
    ax.set_ylabel("|Spearman ρ| of Fiedler order")
    ax.legend(frameon=False, fontsize=7)
    style_ax(ax)
    # (c) LEACE slopes
    ax = axes[2]
    for m, c in (("olmo2_7b", BLUE), ("qwen3_8b", ORANGE),
                 ("tfidf_char", INK)):
        d = J("erasure", "results", f"{m}.eng_tier0.json")
        ax.plot([0, 1], [d["raw"]["pairwise_macro"],
                         d["erased"]["pairwise_macro"]], "-o", ms=5,
                color=c, lw=1.8, label=ARM_LABEL.get(m, m))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "after LEACE\n(provenance+length)"],
                       fontsize=8)
    ax.set_ylabel("eng pairwise accuracy")
    ax.set_title("F17: most of the English signal\nrides on find-spot+length",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    style_ax(ax)
    # (d) nonlinear heads
    ax = axes[3]
    arms = ["olmo2_7b", "qwen3_8b", "olmo2_7b_random", "tfidf_char"]
    heads = ["ridge", "krr_rbf", "mlp"]
    marks = {"ridge": "o", "krr_rbf": "s", "mlp": "^"}
    for i, m in enumerate(arms):
        d = J("erasure", "results", f"nl.{m}.akk_maximal.json")["heads"]
        c = BLUE if m == "olmo2_7b" else ORANGE if m == "qwen3_8b" else \
            GRAY if "random" in m else INK
        for h in heads:
            ax.plot(i, d[h]["pairwise_macro"], marks[h], ms=6, color=c,
                    mfc="white" if h != "mlp" else c, mew=1.3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["OLMo", "Qwen", "twin", "floor"], fontsize=8)
    ax.set_ylabel("akk pairwise accuracy")
    handles = [Line2D([], [], marker=marks[h], ls="", color=INK,
                      mfc="white" if h != "mlp" else INK, label=h)
               for h in heads]
    ax.legend(handles=handles, frameon=False, fontsize=7)
    ax.set_title("F27: the MLP lifts twin & floor too\n= capacity finds form",
                 fontsize=9)
    style_ax(ax)
    fig.suptitle("Every candidate 'document time' signal decomposes into"
                 " text form (length, find-spot), in every instrument",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_form.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ========================== fig 7 — the ignition null =======================
def fig7():
    ig = J("steering", "results", "ignite.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, lang, title in zip(
            axes, ("eng_namespan", "akk_allbutlast"),
            ("English glosses — clamp at the ruler-NAME tokens",
             "Akkadian — clamp at every position but the last")):
        arm = ig["arms"][lang]
        base = arm[f"feat:treat:{ig['treat'][0]}:a0"]["probe"]
        rows, colors, labels = [], [], []
        for g, c in (("treat", AQUA), ("ctrl", GRAY)):
            for f in ig[g]:
                for a in (4, 8):
                    rows.append(arm[f"feat:{g}:{f}:a{a}"]["probe"] - base)
                    colors.append(c)
                    labels.append(f"{f}@{a}")
        y = np.arange(len(rows))
        ax.barh(y, rows, color=colors, height=.72)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Δ probe read-out vs baseline (sd units)")
        ax.set_title(title, fontsize=9.5)
        style_ax(ax)
    handles = [Line2D([], [], marker="s", ls="", color=AQUA,
                      label="treated (top-5 year features)"),
               Line2D([], [], marker="s", ls="", color=GRAY,
                      label="rate-matched controls")]
    fig.legend(handles=handles, frameon=False, fontsize=8,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("F26 — igniting the anchor inside documents fails with"
                 " controls: every treated shift lies inside the control"
                 " band", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_ignition.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        fn()
        print(f"[fig] {fn.__name__} done", flush=True)
    print(f"[done] -> {OUT}", flush=True)
