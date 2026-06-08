#!/usr/bin/env python3
"""Balanced + maximal OOF per-fragment year predictions for error analysis.

Companion to dump_oof_predictions.py, but in the CLASS-BALANCED regime: instead
of one imbalanced GroupKFold over the full corpus, we re-fit PLS inside each of
the 200 Monte-Carlo balanced draws (8 rulers x 21 frags, from
balanced_subset/draws_matrix.npy) and average a fragment's out-of-fold predicted
year over every draw it appears in.

Pipeline per draw (mirrors run_mc_probes.run_*_pls exactly):
  - features X = TF-IDF(text_maximal)  OR  L2-normalized maximal activations
    at the model's best maximal layer.
  - sweep k in PLS_K_VALUES = [1,2,3,5]; pick best_k by mean OOF Spearman
    (year-raw) via fit_pls_groupkfold (the same selector run_mc_probes uses).
  - refit GroupKFold-by-ruler at best_k to get one OOF predicted year per
    fragment in the draw.
Aggregate: pred_year[frag] = mean over draws it appeared in; n_draws[frag] = #.

Output: predictions.csv with the same schema as predictions_maximal/ (one row
per fragment, metadata slice cols + pred_<model>), a drop-in for error_map.py /
analyze_per_model.py / compare_anchor.py. Covers only the ~1076 fragments of the
8 balanced rulers (the only ones the balanced regime ever samples).

Validation: TF-IDF needs no activations, so this runs end-to-end on a laptop for
--models tfidf; the neural models need maximal activations (cluster only).

Usage (cluster, all 3):
    python dump_oof_predictions_balanced.py \
        --out-dir v_1/src/geodesic/fig1_followups/error_overlap/predictions_maximal_balanced \
        --models tfidf,thalesian_cunei400m,qwen3_32b
Usage (laptop sanity, tfidf only):
    python dump_oof_predictions_balanced.py --out-dir /tmp/bal_tfidf --models tfidf
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[5]                       # .../lititure-review
_PHASE0 = _REPO_ROOT / "v_1/src/linear_probing/round2_phase0"
_PROBES = _REPO_ROOT / "v_1/src/linear_probing"
for p in (_PHASE0, _PROBES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_mc_probes as rmp                          # noqa: E402  activation loader
from pls_utils import l2_normalize, fit_pls_groupkfold  # noqa: E402

ORCC_PARQUET = _REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO_ROOT / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
DRAWS_MATRIX = BAL / "draws_matrix.npy"
FRAGMENT_ORDER = BAL / "corpus_fragment_order.json"

PLS_K_VALUES = [1, 2, 3, 5]      # identical grid to run_mc_probes
N_SPLITS = 5
TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 5))

# best layer per (cleaning, model) from T1 (same as dump_oof_predictions.py).
LAYER_BY_CLEANING = {
    "tier0":   {"thalesian_cunei400m": 12, "qwen3_32b": 9, "tfidf": 0},
    "maximal": {"thalesian_cunei400m": 9,  "qwen3_32b": 7, "tfidf": 0},
}
POOLING = {"thalesian_cunei400m": "mean", "qwen3_32b": "mean", "tfidf": "na"}
_CLEANING = "maximal"                       # set by main() from --cleaning
_LAYER = LAYER_BY_CLEANING[_CLEANING]


def _tfidf_draw(texts: list[str]) -> np.ndarray:
    """TF-IDF fit on this draw's texts only (matches run_mc_probes per-draw fit)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    X = TfidfVectorizer(**TFIDF_PARAMS).fit_transform(texts)
    return normalize(X, norm="l2").toarray().astype(np.float32)


def _activations_full(model: str, acts_base: Path) -> np.ndarray | None:
    """Full-corpus L2-normalized activations for the active cleaning (row order = parquet)."""
    rmp._CLEANING, rmp._POOLING = _CLEANING, POOLING[model]
    X = rmp._load_orcc_activations(model, _LAYER[model], acts_base)
    return None if X is None else l2_normalize(X)


def _oof_at_k(X, y, groups, k) -> np.ndarray:
    """GroupKFold OOF predicted year (raw) at fixed n_components=k."""
    pred = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        kk = min(k, X[tr].shape[0] - 1, X.shape[1])
        pls = PLSRegression(n_components=max(1, kk))
        pls.fit(X[tr], y[tr])
        pred[va] = pls.predict(X[va]).ravel()
    return pred


def main() -> None:
    global _CLEANING, _LAYER
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="tfidf,thalesian_cunei400m,qwen3_32b",
                    help="comma-sep subset of {tfidf,thalesian_cunei400m,qwen3_32b}")
    ap.add_argument("--cleaning", default="maximal", choices=["tier0", "maximal"],
                    help="text/activation cleaning (default maximal)")
    ap.add_argument("--draw-range", default="0-199", help="inclusive, e.g. 0-199")
    ap.add_argument("--activations-base", type=Path,
                    default=_REPO_ROOT / "v_1/src/linear_probing/results")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _CLEANING = args.cleaning
    _LAYER = LAYER_BY_CLEANING[_CLEANING]
    print(f"[cleaning] {_CLEANING}  layers={_LAYER}")
    models = [m for m in args.models.split(",") if m]
    lo, hi = (int(x) for x in args.draw_range.split("-"))
    draw_ids = list(range(lo, hi + 1))

    df = pd.read_parquet(ORCC_PARQUET).reset_index(drop=True)
    frag_order = json.load(open(FRAGMENT_ORDER))
    assert len(frag_order) == len(df), "fragment_order length != corpus length"
    draws = np.load(DRAWS_MATRIX)
    print(f"[data] corpus={len(df)}  draws_matrix={draws.shape}  draws={lo}-{hi}  models={models}")

    # accumulate per (model, full-corpus-position): sum of OOF preds + count
    sums = {m: np.zeros(len(df)) for m in models}
    cnts = {m: np.zeros(len(df), dtype=int) for m in models}
    best_k_log = {m: [] for m in models}

    # neural features: full-corpus activations, built once and indexed per draw.
    # tfidf: re-fit per draw on the draw's 168 texts (run_mc_probes convention).
    feats = {}
    for m in models:
        if m == "tfidf":
            feats[m] = "tfidf"                       # sentinel: built in the loop
            print(f"[feat] tfidf: per-draw fit on text_{_CLEANING}")
            continue
        X = _activations_full(m, args.activations_base)
        if X is None:
            print(f"[skip] {m}: no {_CLEANING} activations at L{_LAYER[m]:02d}")
            continue
        feats[m] = X
        print(f"[feat] {m}: X={X.shape} (L{_LAYER[m]}, {POOLING[m]})")

    for di in draw_ids:
        pos, y_raw, _, y_ruler, _ = rmp._draw_subset(df, frag_order, draws, di)
        for m, X in feats.items():
            if m == "tfidf":
                texts = df.iloc[pos][f"text_{_CLEANING}"].fillna("").astype(str).tolist()
                Xd = _tfidf_draw(texts)
            else:
                Xd = X[pos]
            # best k by mean OOF Spearman (year-raw) — run_mc_probes' selector
            best_k, best_sp = PLS_K_VALUES[0], -np.inf
            for k in PLS_K_VALUES:
                try:
                    sp = fit_pls_groupkfold(Xd, y_raw, y_ruler, n_components=k,
                                            n_splits=N_SPLITS)["spearman_mean"]
                except Exception:
                    sp = np.nan
                if np.isfinite(sp) and sp > best_sp:
                    best_sp, best_k = sp, k
            best_k_log[m].append(best_k)
            pred = _oof_at_k(Xd, y_raw, y_ruler, best_k)
            ok = ~np.isnan(pred)
            sums[m][pos[ok]] += pred[ok]
            cnts[m][pos[ok]] += 1
        if di % 25 == 0:
            print(f"  draw {di:3d} done", flush=True)

    # aggregate: mean OOF year per fragment, over draws it appeared in
    slice_cols = [c for c in ("ruler", "period", "provenance", "domain", "sub_genre")
                  if c in df.columns]
    covered = np.zeros(len(df), dtype=bool)
    for m in feats:
        covered |= cnts[m] > 0
    idx = np.where(covered)[0]
    print(f"[agg] {len(idx)} fragments covered by >=1 draw")

    merged = args.out_dir / "predictions.csv"
    with open(merged, "w", newline="") as f:
        w = csv.writer(f)
        head = (["fragment_id", "year_true"] + slice_cols
                + [f"pred_{m}" for m in feats] + [f"ndraws_{m}" for m in feats])
        w.writerow(head)
        for i in idx:
            row = [frag_order[i], float(df.iloc[i]["year"])]
            row += [str(df.iloc[i][c]) for c in slice_cols]
            for m in feats:
                row.append(sums[m][i] / cnts[m][i] if cnts[m][i] else "")
            for m in feats:
                row.append(int(cnts[m][i]))
            w.writerow(row)
    print(f"[ok] -> {merged}")

    summ = {m: {"best_k_hist": {int(k): int(np.sum(np.array(best_k_log[m]) == k))
                                for k in PLS_K_VALUES},
                "n_draws": len(best_k_log[m])} for m in feats}
    json.dump(summ, open(args.out_dir / "best_k_summary.json", "w"), indent=2)
    print("[ok] best-k histogram:", summ)


if __name__ == "__main__":
    main()
