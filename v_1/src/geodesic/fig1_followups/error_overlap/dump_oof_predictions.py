#!/usr/bin/env python3
"""Task 3 — dump out-of-fold per-fragment year predictions (CLUSTER).

For each of the 4 Fig-1A models, runs imbalanced GroupKFold-by-ruler PLS year
regression over the FULL labeled ORCC corpus and saves one out-of-fold
predicted year per fragment. This is the only regime where every fragment gets
a prediction (balanced only ever samples the 8 well-attested rulers).

Output: one JSON per model with a list of
    {fragment_id, ruler, year_true, year_pred}
plus a merged predictions.csv for the local analysis step.

Reuses the activation loader + path map from run_mc_probes.py so it reads the
exact same cluster activation tensors.

Usage:
    python dump_oof_predictions.py \
        --out-dir v_1/src/geodesic/fig1_followups/error_overlap/predictions \
        --k 5 --n-splits 5
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
_REPO_ROOT = _THIS.parents[5]            # .../lititure-review
_PHASE0 = _REPO_ROOT / "v_1/src/linear_probing/round2_phase0"
_PROBES = _REPO_ROOT / "v_1/src/linear_probing"
for p in (_PHASE0, _PROBES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_mc_probes as rmp            # noqa: E402  (activation loader + path map)
from pls_utils import l2_normalize     # noqa: E402

ORCC_PARQUET = _REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
FRAGMENT_ORDER = (_REPO_ROOT / "v_1/src/linear_probing/results/orcc_round2_phase0"
                  / "balanced_subset/corpus_fragment_order.json")
TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 5))

# best layer per (cleaning, model) from T1. mlm has no maximal activations
# (tier0-only) -> it will skip gracefully on --cleaning maximal.
LAYERS_BY_CLEANING = {
    "tier0":   {"mlm": 1,  "thalesian_cunei400m": 12, "qwen3_32b": 9, "tfidf": 0},
    "maximal": {"mlm": 1,  "thalesian_cunei400m": 9,  "qwen3_32b": 7, "tfidf": 0},
}
POOLING = {"mlm": "mean", "thalesian_cunei400m": "mean", "qwen3_32b": "mean", "tfidf": "na"}


def build_models(cleaning: str) -> dict:
    """model -> (method, layer, cleaning, pooling) for the requested cleaning."""
    layers = LAYERS_BY_CLEANING[cleaning]
    return {m: (m, layers[m], cleaning, POOLING[m]) for m in layers}


def _features(model: str, layer: int, cleaning: str, pooling: str,
              df: pd.DataFrame, acts_base: Path) -> np.ndarray | None:
    """Return the L2-normalized feature matrix for the labeled rows in df."""
    if model == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        texts = df[f"text_{cleaning}"].fillna("").astype(str).tolist()
        X = TfidfVectorizer(**TFIDF_PARAMS).fit_transform(texts)
        return normalize(X, norm="l2").toarray().astype(np.float32)
    # activation-based: set the loader's runtime cleaning/pooling, then index
    rmp._CLEANING, rmp._POOLING = cleaning, pooling
    X_full = rmp._load_orcc_activations(model, layer, acts_base)
    if X_full is None:
        return None
    return l2_normalize(X_full[df["_pos"].values])


def oof_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                k: int, n_splits: int) -> np.ndarray:
    """GroupKFold OOF predicted year (raw)."""
    pred = np.full(len(y), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(X, y, groups):
        pls = PLSRegression(n_components=min(k, X[tr].shape[0] - 1, X.shape[1]))
        pls.fit(X[tr], y[tr])
        pred[va] = pls.predict(X[va]).ravel()
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--cleaning", default="tier0", choices=["tier0", "maximal"],
                    help="text/activation cleaning variant (default tier0)")
    ap.add_argument("--activations-base", type=Path,
                    default=_REPO_ROOT / "v_1/src/linear_probing/results")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models_cfg = build_models(args.cleaning)
    print(f"[cleaning] {args.cleaning}  models={list(models_cfg)}")

    df = pd.read_parquet(ORCC_PARQUET).reset_index(drop=True)
    df["_pos"] = np.arange(len(df))
    frag_order = json.load(open(FRAGMENT_ORDER))
    assert len(frag_order) == len(df), "fragment_order length != corpus length"
    df["_fid"] = frag_order
    labeled = df[df["year"].notna()].copy()
    print(f"[data] {len(labeled)} labeled fragments / {labeled['ruler'].nunique()} rulers")

    y = labeled["year"].astype(float).values
    groups = labeled["ruler"].astype(str).values

    all_preds: dict[str, np.ndarray] = {}
    for name, (method, layer, cleaning, pooling) in models_cfg.items():
        X = _features(method, layer, cleaning, pooling, labeled, args.activations_base)
        if X is None:
            print(f"[skip] {name}: no activations at L{layer:02d}")
            continue
        pred = oof_predict(X, y, groups, args.k, args.n_splits)
        all_preds[name] = pred
        recs = [{"fragment_id": fid, "ruler": r, "year_true": float(yt),
                 "year_pred": (None if np.isnan(yp) else float(yp))}
                for fid, r, yt, yp in zip(labeled["_fid"], groups, y, pred)]
        out = args.out_dir / f"oof_{name}.json"
        json.dump({"model": name, "layer": layer, "k": args.k,
                   "regime": "imbalanced_groupkfold_ruler", "records": recs},
                  open(out, "w"), indent=2, ensure_ascii=False)
        mae = np.nanmean(np.abs(pred - y))
        print(f"[ok] {name}: MAE={mae:.1f} yr  -> {out.name}")

    # merged CSV for the local analysis step. Carry the metadata-label columns
    # that exist so analyze_overlap.py can slice the error overlap by each
    # (ruler / period / provenance / domain) and see whether shared errors
    # cluster on any label.
    if all_preds:
        # corpus/word_language/genre are single-valued and sub_provenance is all
        # null in ORCC, so they are deliberately excluded as slice labels.
        slice_cols = [c for c in ("ruler", "period", "provenance", "domain",
                                  "sub_genre")
                      if c in labeled.columns]
        merged = args.out_dir / "predictions.csv"
        with open(merged, "w", newline="") as f:
            w = csv.writer(f)
            cols = list(all_preds)
            w.writerow(["fragment_id", "year_true"] + slice_cols + [f"pred_{m}" for m in cols])
            meta = labeled[slice_cols].astype(str).values
            for i, (fid, yt) in enumerate(zip(labeled["_fid"], y)):
                row = [fid, yt] + list(meta[i]) + [all_preds[m][i] for m in cols]
                w.writerow(row)
        print(f"[ok] merged -> {merged}  (slice labels: {slice_cols})")


if __name__ == "__main__":
    main()
