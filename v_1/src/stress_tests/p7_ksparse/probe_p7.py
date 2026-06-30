"""J9 — P7 k-sparse localization probe (CPU).

"Finding Neurons in a Haystack": how many neurons does it take to recover the
date? Binarize to before/after the median year (clean two-class target), then for
k in {1,2,4,8,16,32,64} select the top-k neurons by univariate F-score and fit a
logistic probe (GroupKFold-by-ruler). Small-k success => localized "time neurons"
(the Gurnee-Tegmark claim); only-large-k => distributed; never => absent.

Framed as a localization/capacity diagnostic (a small-k failure is NOT proof of
absence). Per method x layer (mean, tier0). Emits results/p7_ksparse__<method>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import warnings
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
    y = df["year"].to_numpy()[mask]
    g = df["ruler"].astype(str).to_numpy()[mask]
    binary = (y < np.median(y)).astype(int)   # 1 = more recent (smaller BCE)
    return mask, binary, g


def _ksparse_cv(X, y, g, k, n_splits=5):
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score, accuracy_score
    gkf = GroupKFold(n_splits=n_splits)
    f1s, accs = [], []
    for tr, te in gkf.split(X, y, g):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        F, _ = f_classif(X[tr], y[tr])
        top = np.argsort(np.nan_to_num(F))[::-1][:k]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr][:, top], y[tr])
        pred = clf.predict(X[te][:, top])
        f1s.append(f1_score(y[te], pred, average="macro"))
        accs.append(accuracy_score(y[te], pred))
    return (float(np.mean(f1s)) if f1s else float("nan"),
            float(np.mean(accs)) if accs else float("nan"))


def run(args):
    mask, ybin, g = _labels()
    d = find_acts_dir(args.method, "tier0", "mean")
    out = {"method": args.method, "n": int(mask.sum()),
           "chance_acc": float(max(np.mean(ybin), 1 - np.mean(ybin)))}
    if d is None:
        out["missing"] = True
    else:
        layers = available_layers(d)
        per = {}
        for L in layers:
            X = load_layer(d, L)[mask]
            curve = {k: _ksparse_cv(X, ybin, g, k) for k in KS if k < X.shape[1]}
            per[L] = {str(k): {"macro_f1": v[0], "acc": v[1]} for k, v in curve.items()}
        # best layer by full-k (largest) macro-f1
        kmax = str(max(int(k) for k in next(iter(per.values()))))
        bestL = max(per, key=lambda L: per[L][kmax]["macro_f1"])
        out["best_layer"] = bestL
        out["per_layer"] = per
        # localization summary at best layer: smallest k reaching 90% of full-k f1
        cur = per[bestL]
        full = cur[kmax]["macro_f1"]
        k_at_90 = next((k for k in sorted(cur, key=lambda z: int(z))
                        if cur[k]["macro_f1"] >= 0.9 * full), None)
        out["localization"] = {"full_k_macro_f1": full, "k_reaching_90pct": k_at_90}
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p7_ksparse__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not out.get("missing"):
        loc = out["localization"]
        print(f"  {args.method}: best L{out['best_layer']} full-k F1={loc['full_k_macro_f1']:.3f} "
              f"(chance acc {out['chance_acc']:.2f}); k@90%={loc['k_reaching_90pct']}")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
