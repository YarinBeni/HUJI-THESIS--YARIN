"""J9b — P7 k-sparse localization, v2 (CPU).

Extends probe_p7.py on two axes the first pass hardcoded away:
  * --cleaning {tier0, maximal, maxking}  (mean-pool activations, all on disk)
  * TWO probe families per layer x k (k = number of NEURONS, {1,2,4,8,16,32,64}):
      - cls: the original Haystack-style sparse probe — binarize year at the median,
        select top-k neurons by ANOVA F (f_classif) on the training folds, fit
        LogisticRegression on those k dims -> macro-F1 / accuracy.
      - reg: the k-sparse YEAR-REGRESSION variant — select top-k neurons by
        f_regression against the continuous year on the training folds, fit
        Ridge(alpha=1.0) on those k dims -> Spearman(pred year, true) / MAE.
  * GroupKFold-by-ruler (5 splits) for both, matching the original protocol.

Emits results/v2/p7_v2__<method>__<cleaning>.json with the full per-layer x per-k
curves for both families (the input to plot_p7_curves.py).

Usage:  python probe_p7_v2.py --method qwen3_8b --cleaning maxking
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from geo_loader import find_acts_dir, load_layer, available_layers   # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
KS = [1, 2, 4, 8, 16, 32, 64]


def _labels():
    df = pd.read_parquet(CORPUS)
    mask = df["year"].notna().to_numpy()
    y = df["year"].to_numpy(dtype=float)[mask]
    g = df["ruler"].astype(str).to_numpy()[mask]
    ybin = (y < np.median(y)).astype(int)
    return mask, y, ybin, g


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(set(np.asarray(a).tolist())) < 2:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r) if r == r else float("nan")


def _cv_both(X, y, ybin, g, k, n_splits=5):
    from sklearn.feature_selection import f_classif, f_regression
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score, accuracy_score
    gkf = GroupKFold(n_splits=n_splits)
    f1s, accs, sps, maes = [], [], [], []
    for tr, te in gkf.split(X, ybin, g):
        if len(set(ybin[tr])) < 2 or len(set(ybin[te])) < 2:
            continue
        # cls: F-score on the binary target
        F, _ = f_classif(X[tr], ybin[tr])
        top = np.argsort(np.nan_to_num(F))[::-1][:k]
        clf = LogisticRegression(max_iter=1000).fit(X[tr][:, top], ybin[tr])
        pred = clf.predict(X[te][:, top])
        f1s.append(f1_score(ybin[te], pred, average="macro"))
        accs.append(accuracy_score(ybin[te], pred))
        # reg: F-score on the continuous year
        Fr, _ = f_regression(X[tr], y[tr])
        topr = np.argsort(np.nan_to_num(Fr))[::-1][:k]
        reg = Ridge(alpha=1.0).fit(X[tr][:, topr], y[tr])
        yp = reg.predict(X[te][:, topr])
        sps.append(_spearman(y[te], yp))
        maes.append(float(np.mean(np.abs(yp - y[te]))))
    def m(v): return float(np.nanmean(v)) if v else float("nan")
    return {"macro_f1": m(f1s), "acc": m(accs), "reg_spearman": m(sps), "reg_mae": m(maes)}


def run(args):
    mask, y, ybin, g = _labels()
    d = find_acts_dir(args.method, args.cleaning, "mean")
    out = {"method": args.method, "cleaning": args.cleaning, "n": int(mask.sum()),
           "chance_acc": float(max(np.mean(ybin), 1 - np.mean(ybin))), "ks": KS}
    if d is None:
        out["missing"] = True
    else:
        per = {}
        for L in available_layers(d):
            X = load_layer(d, L)[mask]
            rowmask = np.isfinite(X).all(axis=1)
            Xc, yc, ybc, gc = X[rowmask], y[rowmask], ybin[rowmask], g[rowmask]
            per[str(L)] = {str(k): _cv_both(Xc, yc, ybc, gc, k)
                           for k in KS if k < Xc.shape[1]}
        out["per_layer"] = per
        kmax = str(max(KS))
        bestL_cls = max(per, key=lambda L: per[L][kmax]["macro_f1"])
        bestL_reg = max(per, key=lambda L: per[L][kmax]["reg_spearman"]
                        if per[L][kmax]["reg_spearman"] == per[L][kmax]["reg_spearman"] else -9)
        out["best_layer_cls"] = bestL_cls
        out["best_layer_reg"] = bestL_reg
        print(f"  {args.method} {args.cleaning}: cls bestL{bestL_cls} "
              f"fullk_F1={per[bestL_cls][kmax]['macro_f1']:.3f} | "
              f"reg bestL{bestL_reg} fullk_sp={per[bestL_reg][kmax]['reg_spearman']:.3f}")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p7_v2__{args.method}__{args.cleaning}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--cleaning", required=True, choices=["tier0", "maximal", "maxking"])
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "v2"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
