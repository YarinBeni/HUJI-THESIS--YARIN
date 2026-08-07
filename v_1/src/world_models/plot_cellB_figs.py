"""Cell-B deck figures: the entity-level layer sweep and the entity-level
PLS-k sweep, on the bare names (the read-out the deck's entity table leads
with: rows='bare', last/mean over the name's own tokens, entity-level MC).

Same house style as the cell-A and cell-C figures (plot_cellA_figs.py): one
hue per model family, controls purple and dashed, TF-IDF a dotted floor, star
on each arm's best point, so all three cells of the deck read identically.

Four panels each: YEAR (34 obscure rulers -> reign year, Spearman) under both
name poolings, and PLACE (25 find-spots -> lat/lon, R2) under both name
poolings. The PLS-k sweep runs k = 1..64 at each arm's best layer; arms whose
sweep is still running on the cluster are skipped and listed on stdout.

    python plot_cellB_figs.py       # -> results/figs/fig_cellB_{layers,plsk}.png
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cellA_figs import COLORS, IS_CTRL, LABEL, ORDER, _legend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AKK = os.path.join(HERE, "akkadian", "results")
FIGS = os.path.join(HERE, "results", "figs")
ROWS = "bare"   # the name alone; the in-sentence variants live in the table

# (entity_type, site, metric, panel title)
PANELS = [
    ("assyrian_ruler", "ent_last", "mc_rho",
     "YEAR · ruler name last token · Spearman $\\rho$"),
    ("assyrian_ruler", "ent_mean", "mc_rho",
     "YEAR · ruler name average · Spearman $\\rho$"),
    ("mesopotamian_place", "ent_last", "mc_r2",
     "PLACE · place name last token · R$^2$"),
    ("mesopotamian_place", "ent_mean", "mc_r2",
     "PLACE · place name average · R$^2$"),
]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
})


def load(arm, et, site):
    p = os.path.join(AKK, "probes_entity", arm, f"{et}.{site}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def tfidf_floors():
    """The deck table's TF-IDF row: char n-grams on the name text, ridge,
    entity-level MC. One number per entity type and metric."""
    out = {}
    with open(os.path.join(AKK, "summary_entity_best.csv")) as f:
        for r in csv.DictReader(f):
            if r["arm"] == "tfidf" and r["rows"] == ROWS:
                out[(r["entity_type"], "mc_rho")] = float(r["ridge_mc_rho"])
                out[(r["entity_type"], "mc_r2")] = float(r["ridge_mc_r2"])
    return out


def _style(ax, metric, xlabel, title):
    ax.set_title(title, pad=9, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test Spearman $\\rho$" if metric == "mc_rho" else "test R$^2$")
    ax.grid(alpha=0.22, lw=0.7)
    ax.axhline(0, color="#999", lw=0.8, zorder=0)
    if metric == "mc_r2":
        ax.set_yscale("symlog", linthresh=0.1, linscale=0.7)
        ax.set_yticks([-10, -1, 0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["-10", "-1", "0", ".25", ".5", ".75", "1"])
        # several arms dive far below zero; clip the view so the failure is
        # visible without letting the worst excursion own the panel.
        ax.set_ylim(-1.8, 1.25)
    else:
        ax.set_ylim(-0.1, 0.95)


def _draw(ax, curves, floor, metric, xlabel, title, logx=False, floor_curve=None):
    # On the k panels the floor is itself a function of k, so draw it as a curve;
    # a flat line there would compare every arm's swept best against the floor's
    # single fixed-k score, which is not the same quantity.
    if floor_curve is not None:
        fx, fy = floor_curve
        ax.plot(fx, fy, color="k", lw=2.0, ls=(0, (1, 1.6)), marker="o", ms=4.0,
                label=LABEL["tfidf"], zorder=4)
    elif floor is not None:
        ax.axhline(floor, color="k", lw=1.8, ls=(0, (1, 1.6)),
                   label=LABEL["tfidf"], zorder=4)
    for arm, xs, ys in curves:
        ctrl = arm in IS_CTRL
        ax.plot(xs, ys, color=COLORS[arm], ls=(0, (5, 2)) if ctrl else "-",
                lw=1.7 if ctrl else 2.8, alpha=0.75 if ctrl else 1.0,
                marker="o" if logx else None, ms=4.5,
                label=LABEL[arm], zorder=2 if ctrl else 3)
        b = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot(xs[b], ys[b], marker="*", ms=22 if not ctrl else 15,
                color=COLORS[arm], mec="#111", mew=1.3,
                zorder=7 if not ctrl else 5, clip_on=False)
    if logx:
        ax.set_xscale("log", base=2)
        # The grid is 1..64, but PLS needs k < min(n_samples, n_features) and cell B
        # holds only a few dozen entities, so the high ks are never fitted. Stop the
        # axis at the largest k anyone actually reached — an axis running to 64 with
        # empty space past 24 reads as missing runs rather than as a hard limit.
        kmax = max([x for _, xs, _ in curves for x in xs] +
                   ([] if floor_curve is None else list(floor_curve[0])) or [1])
        ticks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= max(kmax, 2)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_xlim(0.85, kmax * 1.12)
    _style(ax, metric, xlabel, title)


def fig_layers():
    floors = tfidf_floors()
    fig, axes = plt.subplots(2, 2, figsize=(19, 11.5), layout="constrained")
    for ax, (et, site, metric, title) in zip(axes.ravel(), PANELS):
        curves = []
        for arm in ORDER:
            if arm == "tfidf":
                continue
            d = load(arm, et, site)
            if not d or len(d["layers"]) < 2:
                continue
            layers = sorted(int(l) for l in d["layers"])
            top = max(layers)
            curves.append((arm, [l / top for l in layers],
                           [d["layers"][str(l)][ROWS]["ridge_mc"][metric]
                            for l in layers]))
        _draw(ax, curves, floors.get((et, metric)), metric,
              "depth (layer / total layers)", title)
    _legend(fig, axes)
    fig.suptitle("Obscure entities (cell B): where the ruler's year and the "
                 "find-spot live, per-layer ridge probe on the bare name",
                 fontsize=20, fontweight="bold")
    out = os.path.join(FIGS, "fig_cellB_layers.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[write] {out}")


def _ksweep(arm, et, site):
    """The k = 1..64 sweep at the arm's best layer. Prefer the WBk sweep inside
    the entity probe file; fall back to the older standalone sweep files."""
    d = load(arm, et, site)
    if d:
        rec = d["layers"][str(d["best_layer"])][ROWS]
        if "pls_per_k" in rec:
            return rec["pls_per_k"]
    p = os.path.join(AKK, "probes_entity_pls", arm, f"{et}.{site}.json")
    if os.path.exists(p):
        return json.load(open(p))["rows"][ROWS]["pls_by_k"]
    return None


def fig_plsk():
    floors = tfidf_floors()
    missing = set()
    fig, axes = plt.subplots(2, 2, figsize=(19, 11.5), layout="constrained")
    for ax, (et, site, metric, title) in zip(axes.ravel(), PANELS):
        curves = []
        for arm in ORDER:
            if arm == "tfidf":
                continue
            by_k = _ksweep(arm, et, site)
            if not by_k:
                missing.add(arm)
                continue
            ks = sorted(int(k) for k in by_k)
            curves.append((arm, ks, [by_k[str(k)][metric] for k in ks]))
        fby_k = _ksweep("tfidf", et, "text")
        fcurve = None
        if fby_k:
            fks = sorted(int(k) for k in fby_k)
            fcurve = (fks, [fby_k[str(k)][metric] for k in fks])
        else:
            missing.add("tfidf")
        _draw(ax, curves, floors.get((et, metric)), metric,
              "PLS components k", title, logx=True, floor_curve=fcurve)
        ax.axvline(5, color="#c62828", lw=1.2, ls="-.", alpha=0.5, zorder=1)
    _legend(fig, axes)
    fig.suptitle("Obscure entities (cell B): how many PLS directions the signal needs "
                 "— grid k = 1..64, capped by sample size — at each arm's best layer",
                 fontsize=20, fontweight="bold")
    out = os.path.join(FIGS, "fig_cellB_plsk.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[write] {out}")
    if missing:
        print("[skip] no k-sweep yet for: " + ", ".join(sorted(missing)))


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_layers()
    fig_plsk()
