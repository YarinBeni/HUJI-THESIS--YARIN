#!/usr/bin/env python3
"""Per-model fragment-level success/failure — what does each model UNIQUELY
get right or wrong vs the others?

Reads predictions.csv (per-fragment OOF predicted year for each model) and, at
the fragment level:
  - flags each model correct/wrong (|pred - true| <= tol)
  - per model: UNIQUE wins  (this model right, ALL others wrong)
               UNIQUE losses (this model wrong, ALL others right)
  - a directional disagreement matrix: cell[A,B] = # fragments A right & B wrong
  - characterizes each model's unique sets by period / sub_genre / year

Outputs (under predictions/):
  - fragment_scoreboard.csv   one row per fragment: year, metadata, per-model
                              err + correct flag + n_correct  (sortable for digs)
  - per_model_uniqueness.csv  unique-win / unique-loss counts + where they cluster
  - disagreement_matrix.png   A-right-B-wrong heatmap
  - unique_wins_losses.png    bar of unique wins vs losses per model

Usage:
    python analyze_per_model.py --pred-csv .../predictions.csv --tol 100
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

META = ["ruler", "period", "provenance", "domain", "sub_genre"]

# Okabe-Ito colorblind-safe palette — consistent model->color across every plot.
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
            "#56B4E9", "#999999", "#F0E442"]


def _setup(plt):
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.edgecolor": "#666666",
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "legend.frameon": False, "xtick.color": "#333333", "ytick.color": "#333333",
    })


def _cmap(models):
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}


def group_analysis(rows, models, meta_cols, out, tol, plt, min_n=10, top_k=14):
    """Same error analysis as the fragment level, but aggregated per metadata
    group: a scoreboard CSV + frac-correct heatmap + mean-abs-error heatmap for
    each label (period / sub_genre / ...)."""
    import numpy as np
    for label in meta_cols:
        groups, counts = np.unique([r[label] for r in rows], return_counts=True)
        keep = [g for g, c in sorted(zip(groups, counts), key=lambda x: -x[1])
                if c >= min_n][:top_k]
        if not keep:
            continue
        fracM = np.full((len(keep), len(models)), np.nan)
        errM = np.full((len(keep), len(models)), np.nan)
        sb = out / f"group_scoreboard_{label}.csv"
        with open(sb, "w", newline="") as f:
            w = csv.writer(f)
            head = [label, "n"]
            for m in models:
                head += [f"frac_correct_{m}", f"mean_err_{m}"]
            w.writerow(head)
            for gi, g in enumerate(keep):
                grp = [r for r in rows if r[label] == g]
                line = [g, len(grp)]
                for mi, m in enumerate(models):
                    oks = [r[f"_ok_{m}"] for r in grp]
                    errs = [r[f"_err_{m}"] for r in grp if not np.isnan(r[f"_err_{m}"])]
                    fc = float(np.mean(oks)) if oks else np.nan
                    me = float(np.mean(errs)) if errs else np.nan
                    fracM[gi, mi], errM[gi, mi] = fc, me
                    line += [f"{fc:.3f}", f"{me:.0f}" if not np.isnan(me) else ""]
                w.writerow(line)
        print(f"[ok] {sb.name} ({len(keep)} groups)")

        def heat(M, title, fname, cmap, vmin, vmax, fmt):
            fig, ax = plt.subplots(figsize=(1.4 * len(models) + 3, 0.45 * len(keep) + 2))
            im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(keep)))
            ax.set_yticklabels([f"{g[:24]} (n={int((np.array([r[label] for r in rows])==g).sum())})"
                                for g in keep], fontsize=8)
            for gi in range(len(keep)):
                for mi in range(len(models)):
                    if not np.isnan(M[gi, mi]):
                        ax.text(mi, gi, fmt.format(M[gi, mi]), ha="center", va="center", fontsize=7)
            ax.set_title(title); fig.colorbar(im); fig.tight_layout()
            fig.savefig(out / fname, dpi=150); plt.close(fig)
            print(f"[ok] {fname}")

        heat(fracM, f"frac correct by {label} (±{tol:.0f} yr)",
             f"permodel_frac_{label}.png", "RdYlGn", 0, 1, "{:.2f}")
        heat(errM, f"mean abs error (yr) by {label}",
             f"permodel_mae_{label}.png", "RdYlGn_r", 0, float(np.nanmax(errM)), "{:.0f}")


def grouped_bars(rows, models, labels, out, tol, plt, min_n=10, top_k=14):
    """Grouped bar charts: x = metadata value, one colored bar per model.
    Two per label: fraction-correct and mean-abs-error. Reads the error
    distribution differences across models at a glance (softer than unique sets)."""
    import numpy as np
    _setup(plt)
    cm = _cmap(models)
    for label in labels:
        vals = np.array([r.get(label, "") for r in rows])
        groups, counts = np.unique(vals, return_counts=True)
        keep = [g for g, c in sorted(zip(groups, counts), key=lambda x: -x[1])
                if c >= min_n and g != ""][:top_k]
        if label == "century":
            keep = sorted(keep, key=lambda g: int(g))   # chronological, not by count
        if not keep:
            continue
        nfrac = np.full((len(keep), len(models)), np.nan)
        nmae = np.full((len(keep), len(models)), np.nan)
        for gi, g in enumerate(keep):
            grp = [r for r in rows if r.get(label, "") == g]
            for mi, m in enumerate(models):
                oks = [r[f"_ok_{m}"] for r in grp]
                errs = [r[f"_err_{m}"] for r in grp if not np.isnan(r[f"_err_{m}"])]
                nfrac[gi, mi] = np.mean(oks) if oks else np.nan
                nmae[gi, mi] = np.mean(errs) if errs else np.nan

        def bars(M, ylab, title, fname, ylim=None):
            x = np.arange(len(keep)); w = 0.82 / len(models)
            fig, ax = plt.subplots(figsize=(max(7, 0.85 * len(keep)) + 1.5, 4.8))
            for mi, m in enumerate(models):
                ax.bar(x + (mi - (len(models) - 1) / 2) * w, M[:, mi], w,
                       color=cm[m], edgecolor="white", linewidth=0.6, label=m)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{str(g)[:18]}\n(n={int((vals == g).sum())})" for g in keep],
                               rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylab); ax.set_title(title)
            ax.grid(axis="x", visible=False)
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend(fontsize=8, ncol=len(models), loc="upper center",
                      bbox_to_anchor=(0.5, 1.0))
            fig.tight_layout(); fig.savefig(out / fname); plt.close(fig)
            print(f"[ok] {fname}")

        nice = {"ruler": "ruler", "period": "period", "provenance": "provenance",
                "domain": "domain", "sub_genre": "object type", "century": "century (BCE)"}.get(label, label)
        bars(nfrac, f"fraction correct (±{tol:.0f} yr)",
             f"Dating accuracy by {nice}", f"bars_frac_{label}.png", ylim=(0, 1))
        bars(nmae, "mean abs error (years)",
             f"Dating error by {nice}", f"bars_mae_{label}.png")


def year_line(rows, models, out, tol, plt, window=90):
    """Dating accuracy through time, honestly: x = fragments ordered oldest->
    newest (rank), so every x is a real fragment and there are no fake flat
    bridges across the corpus's big empty year-stretches. Year labels mark a few
    rank positions; clustered labels reveal where the corpus actually sits."""
    import numpy as np
    _setup(plt)
    cm = _cmap(models)
    order = np.argsort([-float(r["year_true"]) for r in rows])   # oldest first
    yrs = np.array([float(rows[i]["year_true"]) for i in order])
    half = window // 2
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(11, 5))
    for m in models:
        ok = np.array([rows[i][f"_ok_{m}"] for i in order], dtype=float)
        roll = np.array([ok[max(0, p - half):p + half].mean() for p in range(len(ok))])
        ax.plot(x, roll, lw=2.2, color=cm[m], label=m)
    ticks = np.linspace(0, len(order) - 1, 8).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(yrs[t])}" for t in ticks])
    ax.set_xlabel("fragments ordered oldest → newest   (tick = year BCE at that point)")
    ax.set_ylabel(f"fraction correct (±{tol:.0f} yr)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Dating accuracy through the corpus (rolling {window}-fragment window)")
    ax.legend(ncol=len(models), loc="lower center")
    fig.tight_layout(); fig.savefig(out / "year_accuracy_line.png"); plt.close(fig)
    print("[ok] year_accuracy_line.png")


def error_densities(rows, models, out, tol, plt):
    """Overlaid smooth densities (KDE) of each model's error — the 'are the
    distributions different like two gaussians' view, which the JS/Wasserstein
    numbers quantify. Left: signed error (bias); right: absolute error."""
    import numpy as np
    from scipy.stats import gaussian_kde
    _setup(plt)
    cm = _cmap(models)

    serr = {m: np.array([r[f"_serr_{m}"] for r in rows if not np.isnan(r[f"_serr_{m}"])])
            for m in models}
    aerr = {m: np.array([r[f"_err_{m}"] for r in rows if not np.isnan(r[f"_err_{m}"])])
            for m in models}
    # readable window (trim long tails): symmetric for signed, [0,hi] for abs
    shi = np.percentile(np.concatenate(list(serr.values())), 96)
    slo = np.percentile(np.concatenate(list(serr.values())), 4)
    s_lim = max(abs(slo), abs(shi))
    a_hi = np.percentile(np.concatenate(list(aerr.values())), 97)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs = np.linspace(-s_lim, s_lim, 500)
    for m in models:
        d = gaussian_kde(serr[m])(xs)
        axes[0].plot(xs, d, color=cm[m], lw=2.2, label=m)
        axes[0].fill_between(xs, d, color=cm[m], alpha=0.10)
    axes[0].axvline(0, color="#888888", ls="--", lw=1)
    axes[0].set_xlabel("signed error: predicted − true (years)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Bias — under- vs over-dating")
    axes[0].legend()

    xa = np.linspace(0, a_hi, 500)
    for m in models:
        d = gaussian_kde(aerr[m])(xa)
        axes[1].plot(xa, d, color=cm[m], lw=2.2, label=m)
        axes[1].fill_between(xa, d, color=cm[m], alpha=0.10)
    axes[1].axvline(tol, color="#888888", ls=":", lw=1.2)
    axes[1].text(tol, axes[1].get_ylim()[1] * 0.92, f" ±{tol:.0f}yr", fontsize=8, color="#666")
    axes[1].set_xlabel("absolute error (years)")
    axes[1].set_title("Error magnitude")
    axes[1].legend()
    fig.suptitle("Per-model error distributions (overlaid) — the picture behind "
                 "the JS / Wasserstein distances", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(out / "error_densities.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] error_densities.png")


def distance_and_map(rows, models, out, tol, plt):
    """Quantify how different the models' error distributions are, and draw a
    map of the models from those distances.

    - JS divergence  (symmetric, bounded version of KL; raw KL is asymmetric and
      blows up on empty bins) between abs-error histograms, pairwise.
    - Wasserstein-1  (earth-mover, in YEARS — interpretable) pairwise.
    - MDS embeds the models in 2D from the JS distances: close points = similar
      error distributions. (With 4 models the map is simple; it gets genuinely
      useful as we add qwen3_1b7/8b/random/etc.)
    """
    import numpy as np
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import wasserstein_distance
    _setup(plt)

    errs = {m: np.array([r[f"_err_{m}"] for r in rows if not np.isnan(r[f"_err_{m}"])])
            for m in models}
    hi = max(e.max() for e in errs.values())
    bins = np.linspace(0, hi, 31)
    hist = {m: np.histogram(errs[m], bins=bins)[0].astype(float) + 1e-9 for m in models}

    n = len(models)
    JS = np.zeros((n, n)); W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            JS[i, j] = jensenshannon(hist[models[i]], hist[models[j]])     # in [0,1]
            W[i, j] = wasserstein_distance(errs[models[i]], errs[models[j]])

    # heatmaps + CSV
    def heat(M, title, fname, fmt):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(M, cmap="magma")
        ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(models, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        color="w" if M[i, j] < M.max() * 0.6 else "k", fontsize=8)
        ax.set_title(title); fig.colorbar(im); fig.tight_layout()
        fig.savefig(out / fname, dpi=150); plt.close(fig)
        print(f"[ok] {fname}")

    heat(JS, "JS divergence of |error| dists", "dist_js.png", "{:.3f}")
    heat(W, "Wasserstein dist of |error| (yr)", "dist_wasserstein.png", "{:.0f}")

    with open(out / "distribution_distances.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pair", "js_divergence", "wasserstein_yr"])
        for i in range(n):
            for j in range(i + 1, n):
                wr.writerow([f"{models[i]}__vs__{models[j]}",
                             f"{JS[i, j]:.4f}", f"{W[i, j]:.1f}"])
    print("[ok] distribution_distances.csv")

    # Node-edge graph: nodes positioned by MDS on Wasserstein (so on-page
    # distance ~ years), every pair connected by an edge LABELED with the
    # Wasserstein distance; thicker/darker edge = closer (more similar errors).
    cm = _cmap(models)
    try:
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                  n_init=12, normalized_stress="auto")
        xy = mds.fit_transform(W)
        fig, ax = plt.subplots(figsize=(7, 6))
        wmax = W.max()
        for i in range(n):
            for j in range(i + 1, n):
                close = 1 - W[i, j] / wmax           # 0..1, 1 = closest
                ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                        color=str(0.85 - 0.6 * close), lw=1 + 5 * close, zorder=1)
                mx, my = (xy[i, 0] + xy[j, 0]) / 2, (xy[i, 1] + xy[j, 1]) / 2
                ax.text(mx, my, f"{W[i, j]:.1f}y", fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                        zorder=2)
        for i, m in enumerate(models):
            ax.scatter(xy[i, 0], xy[i, 1], s=420, color=cm[m],
                       edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(m, (xy[i, 0], xy[i, 1]), textcoords="offset points",
                        xytext=(0, 16), ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Model similarity graph", fontsize=13, pad=18)
        ax.text(0.5, 1.005, "edge = Wasserstein distance of error distributions (yr) · "
                "thicker = more alike", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, color="#555555")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.margins(0.22)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.savefig(out / "model_graph.png", bbox_inches="tight"); plt.close(fig)
        print(f"[ok] model_graph.png  (MDS stress={mds.stress_:.4f})")
    except Exception as e:
        print(f"[warn] graph skipped: {type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=100.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--examples", type=int, default=8,
                    help="how many example fragment_ids to print per set")
    ap.add_argument("--min-n", type=int, default=10, help="min fragments per metadata group")
    ap.add_argument("--top-k", type=int, default=14, help="max groups per metadata label")
    ap.add_argument("--year-bin", type=int, default=50, help="bin width (yr) for the year line")
    args = ap.parse_args()
    out = args.out_dir or args.pred_csv.parent
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.pred_csv)))
    models = [c[5:] for c in rows[0] if c.startswith("pred_")]
    meta_cols = [c for c in META if c in rows[0]]

    # per-fragment correctness
    for r in rows:
        yt = float(r["year_true"])
        for m in models:
            v = r[f"pred_{m}"]
            se = (float(v) - yt) if v not in ("", "nan") else np.nan   # signed
            e = abs(se)
            r[f"_serr_{m}"] = se
            r[f"_err_{m}"] = e
            r[f"_ok_{m}"] = (not np.isnan(e)) and e <= args.tol
        r["_nc"] = sum(r[f"_ok_{m}"] for m in models)

    # ---- fragment scoreboard (full detail, sortable) ----
    sb = out / "fragment_scoreboard.csv"
    with open(sb, "w", newline="") as f:
        w = csv.writer(f)
        head = ["fragment_id", "year_true"] + meta_cols + ["n_correct"]
        for m in models:
            head += [f"err_{m}", f"ok_{m}"]
        w.writerow(head)
        for r in sorted(rows, key=lambda r: r["_nc"]):  # hardest first
            line = [r["fragment_id"], r["year_true"]] + [r.get(c, "") for c in meta_cols] + [r["_nc"]]
            for m in models:
                e = r[f"_err_{m}"]
                line += ["" if np.isnan(e) else f"{e:.0f}", int(r[f"_ok_{m}"])]
            w.writerow(line)
    print(f"[ok] {sb.name}")

    # ---- per-model unique wins / losses ----
    def cluster(frags, col):
        c = Counter(r[col] for r in frags if r.get(col))
        return "; ".join(f"{k}:{v}" for k, v in c.most_common(3))

    uniq = out / "per_model_uniqueness.csv"
    wins, losses = {}, {}
    with open(uniq, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "frac_correct", "unique_win", "unique_loss",
                    "win_top_period", "win_top_subgenre", "loss_top_period", "loss_top_subgenre"])
        for m in models:
            others = [o for o in models if o != m]
            uw = [r for r in rows if r[f"_ok_{m}"] and not any(r[f"_ok_{o}"] for o in others)]
            ul = [r for r in rows if not r[f"_ok_{m}"] and all(r[f"_ok_{o}"] for o in others)]
            wins[m], losses[m] = uw, ul
            fc = np.mean([r[f"_ok_{m}"] for r in rows])
            w.writerow([m, f"{fc:.3f}", len(uw), len(ul),
                        cluster(uw, "period"), cluster(uw, "sub_genre"),
                        cluster(ul, "period"), cluster(ul, "sub_genre")])
            print(f"\n=== {m}  (frac_correct={fc:.3f}) ===")
            print(f"  UNIQUE WINS  (only {m} right): {len(uw)}   "
                  f"period[{cluster(uw,'period')}]  genre[{cluster(uw,'sub_genre')}]")
            for r in uw[:args.examples]:
                print(f"     + {r['fragment_id']}  {int(float(r['year_true']))}BCE  {r.get('period','')}/{r.get('sub_genre','')}")
            print(f"  UNIQUE LOSSES (only {m} wrong): {len(ul)}   "
                  f"period[{cluster(ul,'period')}]  genre[{cluster(ul,'sub_genre')}]")
            for r in ul[:args.examples]:
                print(f"     - {r['fragment_id']}  {int(float(r['year_true']))}BCE  {r.get('period','')}/{r.get('sub_genre','')}")
    print(f"\n[ok] {uniq.name}")

    # ---- directional disagreement matrix + unique bars ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(models)
    D = np.zeros((n, n), dtype=int)   # D[i,j] = # frags  i right & j wrong
    for i, j in product(range(n), range(n)):
        if i != j:
            D[i, j] = sum(r[f"_ok_{models[i]}"] and not r[f"_ok_{models[j]}"] for r in rows)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(D, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(models, fontsize=8)
    ax.set_ylabel("this model RIGHT"); ax.set_xlabel("that model WRONG")
    for i, j in product(range(n), range(n)):
        ax.text(j, i, D[i, j], ha="center", va="center",
                color="w" if D[i, j] > D.max() * 0.6 else "k", fontsize=8)
    ax.set_title(f"Row right & Column wrong (±{args.tol:.0f} yr)")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out / "disagreement_matrix.png", dpi=150); plt.close(fig)
    print("[ok] disagreement_matrix.png")

    fig, ax = plt.subplots(figsize=(1.5 * n + 2, 4.5))
    x = np.arange(n); w = 0.38
    ax.bar(x - w / 2, [len(wins[m]) for m in models], w, color="#1a7a1a", label="unique wins")
    ax.bar(x + w / 2, [len(losses[m]) for m in models], w, color="#8B0000", label="unique losses")
    for i, m in enumerate(models):
        ax.text(i - w / 2, len(wins[m]), len(wins[m]), ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, len(losses[m]), len(losses[m]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel(f"# fragments (vs the other {n-1} models)")
    ax.set_title(f"What each model UNIQUELY gets right / wrong (±{args.tol:.0f} yr)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "unique_wins_losses.png", dpi=150); plt.close(fig)
    print("[ok] unique_wins_losses.png")

    # ---- same error analysis, aggregated per metadata group ----
    print("\n--- per-metadata-group scoreboards ---")
    group_analysis(rows, models, meta_cols, out, args.tol, plt,
                   min_n=args.min_n, top_k=args.top_k)

    # year binned to century, so it can be a grouped-bar x-axis too
    for r in rows:
        r["century"] = f"{int(float(r['year_true']) // 100) * 100}"

    print("\n--- grouped bars (one color per model) per metadata value ---")
    grouped_bars(rows, models, meta_cols, out, args.tol, plt,
                 min_n=args.min_n, top_k=args.top_k)

    print("\n--- accuracy-vs-year (rolling) + error-distribution densities ---")
    year_line(rows, models, out, args.tol, plt)
    error_densities(rows, models, out, args.tol, plt)

    print("\n--- model error-distribution distances + MDS map ---")
    distance_and_map(rows, models, out, args.tol, plt)


if __name__ == "__main__":
    main()
