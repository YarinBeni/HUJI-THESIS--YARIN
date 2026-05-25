#!/usr/bin/env python3
"""Local balanced-MC TF-IDF dating experiment: unmasked vs m- name-masked.

Question: does TF-IDF date Akkadian texts via explicit royal names, or via
period orthography that survives masking those names?

TF-IDF is CPU-trivial, so this runs locally (no cluster). Uses the SAME balanced
draws (200 × 168 frags, 8 rulers × 21) as the cluster MC so numbers are
comparable. For each draw × cleaning × {unmasked, masked}:
  - year:  Ridge GroupKFold(ruler) -> Spearman/MAE (raw-year)
  - ruler: logistic StratifiedKFold -> Macro-F1
Aggregates mean ± std across draws.

Masking: a whitespace token starting with 'm-' is the Akkadian personal-name
determinative -> replaced with [PN]. Conservative / language-principled.
"""
import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "v_1/src/linear_probing"))
from pls_utils import fit_ridge_year_groupkfold          # noqa: E402
from cls_utils import fit_cls_cv                          # noqa: E402

BASE   = ROOT / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
PARQUET = ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 5))
PN = re.compile(r'(?<!\S)m-[^\s]+')


def mask_names(t: str) -> str:
    return PN.sub("[PN]", str(t))


def tfidf(texts):
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X = normalize(vec.fit_transform(texts), norm="l2")
    return X.toarray().astype(np.float32)


def main():
    n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    df = pd.read_parquet(PARQUET)
    fo = json.load(open(BASE / "corpus_fragment_order.json"))
    dm = np.load(BASE / "draws_matrix.npy")
    assert list(df.index) == list(range(len(df))) and len(fo) == len(df)

    # Pre-build masked text columns.
    for c in ("tier0", "maximal"):
        df[f"text_{c}_masked"] = df[f"text_{c}"].apply(mask_names)

    conditions = ["unmasked", "masked"]
    cleanings  = ["tier0", "maximal"]
    # accumulators: results[cond][cleaning] -> {"year_sp":[], "year_mae":[], "ruler_f1":[]}
    acc = {cond: {cl: {"year_sp": [], "year_mae": [], "ruler_f1": []}
                  for cl in cleanings} for cond in conditions}

    for di in range(n_draws):
        pos = np.where(dm[di])[0]
        sub = df.iloc[pos]
        y_raw = sub["year"].astype(float).values
        y_log = np.log(y_raw)
        y_rul = sub["ruler"].astype(str).values
        for cond in conditions:
            suffix = "_masked" if cond == "masked" else ""
            for cl in cleanings:
                texts = sub[f"text_{cl}{suffix}"].fillna("").astype(str).tolist()
                X = tfidf(texts)
                ridge = fit_ridge_year_groupkfold(X, y_raw, y_log, y_rul, n_splits=5)
                r = ridge["raw"]
                if not np.isnan(r.get("spearman_mean", np.nan)):
                    acc[cond][cl]["year_sp"].append(r["spearman_mean"])
                    acc[cond][cl]["year_mae"].append(r["mae_mean"])
                cls = fit_cls_cv(X, y_rul, cv_strategy="stratified", n_splits=5)
                if not np.isnan(cls.get("macro_f1_mean", np.nan)):
                    acc[cond][cl]["ruler_f1"].append(cls["macro_f1_mean"])
        if (di + 1) % 25 == 0:
            print(f"  draw {di+1}/{n_draws} done", flush=True)

    def ms(lst):
        return (float(np.mean(lst)), float(np.std(lst)), len(lst)) if lst else (float("nan"),)*2 + (0,)

    print("\n========== TF-IDF balanced MC: unmasked vs m- name-masked ==========")
    print(f"(n_draws={n_draws}, 168 frags/draw)\n")
    hdr = f"{'cleaning':<9} {'condition':<9} {'YEAR Spearman':<20} {'YEAR MAE':<14} {'RULER Macro-F1':<18}"
    print(hdr); print("-"*len(hdr))
    out = {}
    for cl in cleanings:
        for cond in conditions:
            sp = ms(acc[cond][cl]["year_sp"])
            mae = ms(acc[cond][cl]["year_mae"])
            f1 = ms(acc[cond][cl]["ruler_f1"])
            print(f"{cl:<9} {cond:<9} {sp[0]:.3f} ± {sp[1]:.3f}        "
                  f"{mae[0]:.1f} ± {mae[1]:.1f}    {f1[0]:.3f} ± {f1[1]:.3f}")
            out[f"{cl}__{cond}"] = {"year_sp": sp, "year_mae": mae, "ruler_f1": f1}
        # deltas
        d_sp = (ms(acc['unmasked'][cl]['year_sp'])[0] - ms(acc['masked'][cl]['year_sp'])[0])
        d_f1 = (ms(acc['unmasked'][cl]['ruler_f1'])[0] - ms(acc['masked'][cl]['ruler_f1'])[0])
        print(f"  → {cl} masking drop:  year Spearman {d_sp:+.3f}   ruler Macro-F1 {d_f1:+.3f}\n")

    outdir = ROOT / "v_1/src/linear_probing/results/orcc_round2_phase0"
    (outdir / "tfidf_namemask_results.json").write_text(json.dumps(out, indent=2))
    print(f"Saved → {outdir/'tfidf_namemask_results.json'}")


if __name__ == "__main__":
    main()
