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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=100.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--examples", type=int, default=8,
                    help="how many example fragment_ids to print per set")
    ap.add_argument("--min-n", type=int, default=10, help="min fragments per metadata group")
    ap.add_argument("--top-k", type=int, default=14, help="max groups per metadata label")
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
            e = abs(float(v) - yt) if v not in ("", "nan") else np.nan
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


if __name__ == "__main__":
    main()
