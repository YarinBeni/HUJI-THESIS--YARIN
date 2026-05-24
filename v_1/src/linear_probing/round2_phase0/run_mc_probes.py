"""run_mc_probes.py — Round 2 Phase 0: Monte-Carlo balanced re-run of the 8 Round-1 probes.

Loops the 8 Round-1 probes over N (default 200) balanced sub-draws produced by
`build_balanced_subset.py`:

    tfidf_pls,  tfidf_cls    (no activations needed — pure text)
    mlm_pls,    mlm_cls      (mean-pooled Akkadian MLM, 17 layers, hidden=384)
    qwen_pls,   qwen_cls     (Qwen 2.5-7B mean-pooled, 29 layers, hidden=3584)
    random_pls, random_cls   (random-init Qwen activations, 29 layers, hidden=3584)

The `random` method is the same architecture+tokenizer as Qwen but with
randomly-initialized weights (see `01b_extract_random_baseline.py`). Activations
are precomputed and live alongside the pretrained Qwen ones in `orcc__embed`.
This is the same "random" baseline used in Round-1 (`05_compute_pls.py:125`,
`05_compute_cls.py:78` accept `--method {qwen,random}`).

------------------------------------------------------------------------------
Design choice (subprocess vs. import)
------------------------------------------------------------------------------
None of the existing `05_compute_*.py` probe scripts accept a fragment-id
filter; they all hard-code `ORCC_PARQUET` at module level, load the *full*
labeled set (893 fragments / 38 rulers), and index activations by parquet row
order. Adding a filter flag to all 6 would touch hundreds of lines and risk
silent schema drift.

Instead this wrapper IMPORTS the shared utility modules (`pls_utils`, `cls_utils`)
that the 6 scripts already delegate to, then re-drives each probe in-process
per draw. That keeps the 05_compute_*.py scripts untouched and reuses the
exact same statistical machinery (same `fit_plsda_stratified_kfold`, same
`fit_cls_cv`, same TF-IDF vectorizer config, same L2-normalization).

The per-draw output JSON schema mirrors the Round-1 `pls_results_*.json` /
`cls_results_*.json` files, with two extra top-level keys:
    "method_tag": "mc_balanced"
    "draw_idx":   <int 0..N-1>

------------------------------------------------------------------------------
Output layout
------------------------------------------------------------------------------
{out_dir}/
    {probe}__mc_balanced__draw{NNN}.json    # per-draw, N files per probe
    {probe}__mc_balanced__summary.json      # aggregated over all draws

Resumability: if `{probe}__mc_balanced__draw{NNN}.json` already exists for a
(probe, draw) pair it is SKIPPED (and logged). The summary is rebuilt on every
invocation from whatever per-draw files are currently present.

------------------------------------------------------------------------------
CLI
------------------------------------------------------------------------------
    python run_mc_probes.py \
        --draws-matrix .../balanced_subset/draws_matrix.npy \
        --fragment-order .../balanced_subset/corpus_fragment_order.json \
        --output-dir .../probes \
        --probes tfidf_pls,tfidf_cls,mlm_pls,mlm_cls,qwen_pls,qwen_cls \
        [--draws-range 0-199] \
        [--method-tag mc_balanced]

Both `--draws-matrix` / `--draws_matrix` (and underscore variants of every
flag) are accepted to keep parity with the spec and W2.D's sbatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: make sibling modules (pls_utils, cls_utils) importable
# ---------------------------------------------------------------------------

_THIS_FILE   = Path(__file__).resolve()
_PHASE0_DIR  = _THIS_FILE.parent                  # .../linear_probing/round2_phase0
_PROBES_DIR  = _PHASE0_DIR.parent                 # .../linear_probing
_REPO_ROOT   = _PROBES_DIR.parents[2]             # .../lititure-review

if str(_PROBES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBES_DIR))

# Shared utilities (the 05_compute_*.py scripts delegate to these too)
from pls_utils import (                            # noqa: E402
    l2_normalize,
    fit_plsda_stratified_kfold,
    fit_pls_groupkfold,
    fit_ridge_year_groupkfold,
)
from cls_utils import fit_cls_cv                   # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PROBES = [
    "tfidf_pls", "tfidf_cls",
    "mlm_pls",   "mlm_cls",
    "qwen_pls",  "qwen_cls",
    "random_pls", "random_cls",
]

ORCC_PARQUET = _REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
SEAL_PARQUET = _REPO_ROOT / "v_1/data/evaluation/corpora/seal_corpus.parquet"

# Activation roots (gitignored, present on cluster). Override via --activations-base.
ACTS_BASE_DEFAULT = _PROBES_DIR / "results"
QWEN_ORCC_DIR     = "qwen_tier0_mean"      # matches Round-1 layout
MLM_ORCC_DIR      = "mlm_tier0"
# Random-init Qwen activations (extract_random_tier0.sh writes to this dir).
RANDOM_ORCC_DIR   = "random_tier0_mean"

QWEN_N_LAYERS   = 29       # L00..L28
MLM_N_LAYERS    = 17       # L00..L16
RANDOM_N_LAYERS = 29       # same arch as Qwen → same layer count

# Thalesian (Phase 3 — Akkadian-finetuned UMT5 encoders). Activations were
# extracted to a different subdir (orcc__embed/activations/) than the renamed
# orcc_round1/activations/ Round-1 path. Path resolution per-method below.
THALESIAN_AKK300M_N_LAYERS  = 9        # 8 encoder layers + embedding (L00..L08)
THALESIAN_CUNEI400M_N_LAYERS = 13      # 12 encoder layers + embedding (L00..L12)

# Phase E1 (Round 3) — Qwen3 scale sweep.
# Counts = transformer layers + 1 embedding layer (L00 = embedding).
QWEN3_1B7_N_LAYERS  = 29   # Qwen3-1.7B:  28 transformer + embedding (L00..L28)
QWEN3_8B_N_LAYERS   = 37   # Qwen3-8B:    36 transformer + embedding (L00..L36)
QWEN3_32B_N_LAYERS  = 65   # Qwen3-32B:   64 transformer + embedding (L00..L64)

# Per-method (subdir under acts_base, activation-dir name).
# Used by _load_orcc_activations to dispatch per method.
ACT_PATH_MAP: dict[str, tuple[str, str]] = {
    "qwen":                ("orcc_round1/activations", QWEN_ORCC_DIR),
    "mlm":                 ("orcc_round1/activations", MLM_ORCC_DIR),
    "random":              ("orcc_round1/activations", RANDOM_ORCC_DIR),
    "thalesian_akk300m":   ("orcc__embed/activations", "thalesian_akk300m_tier0_mean"),
    "thalesian_cunei400m": ("orcc__embed/activations", "thalesian_cunei400m_tier0_mean"),
    # Phase E1: Qwen3 scale sweep — tier0/mean only (MC standard)
    "qwen3_1b7":           ("orcc__embed/activations", "qwen3_1b7_tier0_mean"),
    "qwen3_8b":            ("orcc__embed/activations", "qwen3_8b_tier0_mean"),
    "qwen3_32b":          ("orcc__embed/activations", "qwen3_32b_tier0_mean"),
}

# PLS hyper-params: mirror Round 1 sweep (1,2,3,5). With N=168 a 5-split CV is
# easy. Year transforms: raw,log to match Round 1.
PLS_K_VALUES   = [1, 2, 3, 5]
YEAR_TRANSFORMS = ["raw", "log"]
N_SPLITS        = 5

# TF-IDF — match Round 1 char_wb (2,5)
TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 5))


# ---------------------------------------------------------------------------
# CLI parsing (accept both --foo-bar and --foo_bar)
# ---------------------------------------------------------------------------

def _add_dual(p: argparse.ArgumentParser, dashed: str, **kw):
    """Add a flag accepting both --foo-bar and --foo_bar spellings."""
    underscored = dashed.replace("-", "_")
    dest = kw.pop("dest", underscored.lstrip("-"))
    p.add_argument(f"--{dashed}", f"--{underscored}", dest=dest, **kw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_dual(p, "draws-matrix", type=Path, required=True,
              help="Path to draws_matrix.npy (shape (N_draws, N_orcc_full))")
    # accept --fragment-order OR --corpus-order (spec uses corpus_order, sbatch uses fragment-order)
    p.add_argument("--fragment-order", "--fragment_order", "--corpus-order", "--corpus_order",
                   dest="fragment_order", type=Path, required=True,
                   help="Path to corpus_fragment_order.json")
    # accept --output-dir OR --out-dir OR --out_dir
    p.add_argument("--output-dir", "--output_dir", "--out-dir", "--out_dir",
                   dest="output_dir", type=Path, required=True,
                   help="Where to write per-draw + summary JSONs")
    _add_dual(p, "probes", type=str, default=",".join(DEFAULT_PROBES),
              help=f"Comma-separated probe names (default: {','.join(DEFAULT_PROBES)})")
    _add_dual(p, "draws-range", type=str, default=None,
              help="Inclusive draw index range 'A-B' (default: all rows of draws_matrix)")
    _add_dual(p, "method-tag", type=str, default="mc_balanced",
              help="Tag embedded in output filenames / JSON (default: mc_balanced)")
    _add_dual(p, "corpus", type=Path, default=ORCC_PARQUET,
              help=f"Path to ORCC parquet (default: {ORCC_PARQUET})")
    _add_dual(p, "activations-base", type=Path, default=ACTS_BASE_DEFAULT,
              help=f"Activations root (default: {ACTS_BASE_DEFAULT})")
    # Layer subset for activation-based probes (mlm/qwen/random). TF-IDF ignores this.
    # Default matches Phase 1b's 6-layer subset {0,4,10,15,22,28} so the two phases
    # produce apples-to-apples comparable per-layer numbers. Use "all" to scan every
    # layer (full Round-1 grid, much slower for qwen/random).
    _add_dual(p, "layers", type=str, default="0,4,10,15,22,28",
              help="Comma-separated layer indices for mlm/qwen/random probes "
                   "(default: '0,4,10,15,22,28'). Pass 'all' to scan every layer.")
    return p.parse_args()


def _parse_range(s: str | None, n: int) -> list[int]:
    if s is None:
        return list(range(n))
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


# ---------------------------------------------------------------------------
# Activation loading (cached across draws)
# ---------------------------------------------------------------------------

_ACT_CACHE: dict[tuple[str, int], np.ndarray] = {}

# Optional layer subset for activation-based probes (mlm/qwen/random). Set by
# main() from --layers. None means "all layers".
_LAYER_SUBSET: list[int] | None = None


def _load_orcc_activations(method: str, layer: int, acts_base: Path) -> np.ndarray | None:
    """Load full-ORCC activations for a method+layer. Returns None if missing."""
    key = (method, layer)
    if key in _ACT_CACHE:
        return _ACT_CACHE[key]
    if method not in ACT_PATH_MAP:
        raise ValueError(f"Unknown method {method}; known: {list(ACT_PATH_MAP)}")
    subpath, dirname = ACT_PATH_MAP[method]
    # Try the canonical path first. For Round-1 methods (qwen/mlm/random) some
    # clusters keep them at the legacy orcc__embed/ path instead of the renamed
    # orcc_round1/ — try that as a fallback.
    candidates = [acts_base / subpath / dirname / f"layer_{layer:02d}.npz"]
    if subpath.startswith("orcc_round1"):
        candidates.append(acts_base / "orcc__embed" / "activations" / dirname / f"layer_{layer:02d}.npz")
    for npz in candidates:
        if npz.exists():
            arr = np.load(npz)["activations"].astype(np.float32)
            _ACT_CACHE[key] = arr
            return arr
    return None


# ---------------------------------------------------------------------------
# Per-draw drivers (one function per probe). Each returns a "results" dict
# matching the Round-1 schema for that probe.
# ---------------------------------------------------------------------------

def _draw_subset(
    orcc_df: pd.DataFrame,
    fragment_order: list[str],
    draws_matrix: np.ndarray,
    draw_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (orcc_positions, y_raw, y_log, y_ruler, fragment_ids) for one draw.

    `orcc_positions` are integer positions into the full `orcc_df` (parquet row order),
    which is also the row order of the full activations matrices.
    """
    row = draws_matrix[draw_idx]
    # Two supported encodings: boolean mask (N_orcc,) OR int indices (k*n_rulers,)
    if row.dtype == bool:
        orcc_positions = np.where(row)[0]
    else:
        orcc_positions = row.astype(int)

    # Sanity: fragment_order is the parquet row order; orcc_df must match length.
    assert len(fragment_order) == len(orcc_df), (
        f"fragment_order length {len(fragment_order)} != orcc_df length {len(orcc_df)}"
    )
    fragment_ids = [fragment_order[i] for i in orcc_positions]

    sub = orcc_df.iloc[orcc_positions]
    y_raw   = sub["year"].astype(float).values
    y_ruler = sub["ruler"].astype(str).values
    # All 168 draw fragments are non-null-year by construction, so np.log is safe.
    y_log   = np.log(y_raw)
    return orcc_positions, y_raw, y_log, y_ruler, fragment_ids


# ---- TF-IDF probes --------------------------------------------------------

def _build_tfidf_matrix(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X = vec.fit_transform(texts)            # sparse (N, V)
    X = normalize(X, norm="l2")
    return X.toarray().astype(np.float32)


def run_tfidf_pls(orcc_df, orcc_positions, y_raw, y_log, y_ruler, fragment_ids) -> dict:
    results: dict[str, Any] = {}
    sub = orcc_df.iloc[orcc_positions]
    for cleaning in ("tier0", "maximal"):
        col = f"text_{cleaning}"
        texts = sub[col].fillna("").astype(str).tolist()
        X = _build_tfidf_matrix(texts)

        # ── year ───────────────────────────────────────────────────────────
        for yt in YEAR_TRANSFORMS:
            y = y_raw if yt == "raw" else y_log
            metrics_per_k = {}
            for k in PLS_K_VALUES:
                metrics_per_k[str(k)] = fit_pls_groupkfold(
                    X, y, y_ruler, n_components=k, n_splits=N_SPLITS)
            best_sp = max(PLS_K_VALUES, key=lambda k: metrics_per_k[str(k)]["spearman_mean"])
            best_r2 = max(PLS_K_VALUES, key=lambda k: metrics_per_k[str(k)]["r2_mean"])
            results[f"tfidf__{cleaning}__na__L00__year-{yt}"] = {
                "method": "tfidf", "cleaning": cleaning, "pooling": "na",
                "layer": 0, "year_transform": yt,
                "n_labeled": int(X.shape[0]), "n_groups": int(len(np.unique(y_ruler))),
                "metrics_per_k": metrics_per_k,
                "best_k_by_spearman": best_sp, "best_k_by_r2": best_r2,
            }

        # ── ruler ─────────────────────────────────────────────────────────
        metrics_per_k = {}
        for k in PLS_K_VALUES:
            metrics_per_k[str(k)] = fit_plsda_stratified_kfold(
                X, y_ruler, n_components=k, n_splits=N_SPLITS)
        best_k = max(PLS_K_VALUES, key=lambda k: metrics_per_k[str(k)]["macro_f1_mean"])
        results[f"tfidf__{cleaning}__na__L00__ruler"] = {
            "method": "tfidf", "cleaning": cleaning, "pooling": "na",
            "layer": 0, "target": "ruler",
            "n_labeled": int(X.shape[0]),
            "metrics_per_k": metrics_per_k,
            "best_k_by_macro_f1": best_k,
        }
    return results


def run_tfidf_cls_numeric(orcc_df, orcc_positions, y_raw, y_log, y_ruler, fragment_ids) -> dict:
    """Ridge year regression on TF-IDF features (cls_numeric probe)."""
    results: dict[str, Any] = {}
    sub = orcc_df.iloc[orcc_positions]
    for cleaning in ("tier0", "maximal"):
        col = f"text_{cleaning}"
        texts = sub[col].fillna("").astype(str).tolist()
        X = _build_tfidf_matrix(texts)
        try:
            ridge_res = fit_ridge_year_groupkfold(
                X, y_raw, y_log, y_ruler, n_splits=N_SPLITS)
        except Exception as e:
            print(f"    [ridge-tfidf-skip] {cleaning}: {type(e).__name__}: {e}", flush=True)
            ridge_res = {yt: {"spearman_mean": float("nan"), "mae_mean": float("nan"),
                               "r2_mean": float("nan"), "skipped": True}
                         for yt in YEAR_TRANSFORMS}
        for yt in YEAR_TRANSFORMS:
            r = ridge_res[yt]
            results[f"tfidf__{cleaning}__na__L00__year-{yt}"] = {
                "method": "tfidf", "cleaning": cleaning, "pooling": "na",
                "layer": 0, "probe": "ridge", "year_transform": yt,
                "n_labeled": int(X.shape[0]), "n_groups": int(len(np.unique(y_ruler))),
                **r,
            }
    return results


def run_tfidf_cls(orcc_df, orcc_positions, y_raw, y_log, y_ruler, fragment_ids) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    results: dict[str, Any] = {}
    sub = orcc_df.iloc[orcc_positions]
    for cleaning in ("tier0", "maximal"):
        col = f"text_{cleaning}"
        texts = sub[col].fillna("").astype(str).tolist()
        vec = TfidfVectorizer(**TFIDF_PARAMS)
        X_sparse = vec.fit_transform(texts)
        X_sparse = normalize(X_sparse, norm="l2")
        X = X_sparse.toarray().astype(np.float32)

        # ruler
        m = fit_cls_cv(X, y_ruler, cv_strategy="stratified", n_splits=N_SPLITS)
        results[f"tfidf__{cleaning}__na__L00__ruler"] = {
            "method": "tfidf", "cleaning": cleaning, "pooling": "na",
            "layer": 0, "task": "ruler", "n_dropped": 0, **m,
        }
        # year (treated as categorical, matches Round-1 cls behavior)
        y_year_str = np.array([str(int(y)) for y in y_raw])
        m = fit_cls_cv(X, y_year_str, cv_strategy="stratified", n_splits=N_SPLITS)
        results[f"tfidf__{cleaning}__na__L00__year"] = {
            "method": "tfidf", "cleaning": cleaning, "pooling": "na",
            "layer": 0, "task": "year", "n_dropped": 0, **m,
        }
    return results


# ---- MLM / Qwen PLS probes ------------------------------------------------

def _run_acts_pls(method: str, n_layers: int,
                  orcc_positions, y_raw, y_log, y_ruler,
                  acts_base: Path) -> dict:
    """Generic PLS sweep over layers for an activation-based method."""
    results: dict[str, Any] = {}
    pooling   = "mean"
    cleaning  = "tier0"
    layer_iter = _LAYER_SUBSET if _LAYER_SUBSET is not None else range(n_layers)
    for layer in layer_iter:
        if layer >= n_layers:
            continue
        X_full = _load_orcc_activations(method, layer, acts_base)
        if X_full is None:
            print(f"    [{method}] L{layer:02d} — no activations, skip")
            continue
        X = l2_normalize(X_full[orcc_positions])

        # ── year ── (defensive try/except — L0 can be rank-deficient)
        nan_year = {"spearman_mean": float("nan"), "spearman_std": float("nan"),
                    "mae_mean": float("nan"), "mae_std": float("nan"),
                    "r2_mean": float("nan"), "r2_std": float("nan"),
                    "skipped": True}
        for yt in YEAR_TRANSFORMS:
            y = y_raw if yt == "raw" else y_log
            metrics_per_k = {}
            for k in PLS_K_VALUES:
                try:
                    metrics_per_k[str(k)] = fit_pls_groupkfold(
                        X, y, y_ruler, n_components=k, n_splits=N_SPLITS)
                except Exception as e:
                    print(f"    [pls-skip] {method} L{layer:02d} k={k} year-{yt}: {type(e).__name__}: {e}", flush=True)
                    metrics_per_k[str(k)] = {**nan_year, "error": f"{type(e).__name__}: {e}"}
            valid_sp = [k for k in PLS_K_VALUES
                        if not (isinstance(metrics_per_k[str(k)].get("spearman_mean"), float)
                                and np.isnan(metrics_per_k[str(k)]["spearman_mean"]))]
            valid_r2 = [k for k in PLS_K_VALUES
                        if not (isinstance(metrics_per_k[str(k)].get("r2_mean"), float)
                                and np.isnan(metrics_per_k[str(k)]["r2_mean"]))]
            best_sp = (max(valid_sp, key=lambda k: metrics_per_k[str(k)]["spearman_mean"])
                       if valid_sp else PLS_K_VALUES[0])
            best_r2 = (max(valid_r2, key=lambda k: metrics_per_k[str(k)]["r2_mean"])
                       if valid_r2 else PLS_K_VALUES[0])
            results[f"{method}__{cleaning}__{pooling}__L{layer:02d}__year-{yt}"] = {
                "method": method, "cleaning": cleaning, "pooling": pooling,
                "layer": layer, "year_transform": yt,
                "n_labeled": int(X.shape[0]), "n_groups": int(len(np.unique(y_ruler))),
                "metrics_per_k": metrics_per_k,
                "best_k_by_spearman": best_sp, "best_k_by_r2": best_r2,
            }

        # ── ruler ── (defensive try/except — same rank-deficient L0 case)
        nan_ruler = {"accuracy_mean": float("nan"), "accuracy_std": float("nan"),
                     "macro_f1_mean": float("nan"), "macro_f1_std": float("nan"),
                     "skipped": True}
        metrics_per_k = {}
        for k in PLS_K_VALUES:
            try:
                metrics_per_k[str(k)] = fit_plsda_stratified_kfold(
                    X, y_ruler, n_components=k, n_splits=N_SPLITS)
            except Exception as e:
                print(f"    [plsda-skip] {method} L{layer:02d} k={k} ruler: {type(e).__name__}: {e}", flush=True)
                metrics_per_k[str(k)] = {**nan_ruler, "error": f"{type(e).__name__}: {e}"}
        valid_k = [k for k in PLS_K_VALUES
                   if not (isinstance(metrics_per_k[str(k)].get("macro_f1_mean"), float)
                           and np.isnan(metrics_per_k[str(k)]["macro_f1_mean"]))]
        best_k = (max(valid_k, key=lambda k: metrics_per_k[str(k)]["macro_f1_mean"])
                  if valid_k else PLS_K_VALUES[0])
        results[f"{method}__{cleaning}__{pooling}__L{layer:02d}__ruler"] = {
            "method": method, "cleaning": cleaning, "pooling": pooling,
            "layer": layer, "target": "ruler",
            "n_labeled": int(X.shape[0]),
            "metrics_per_k": metrics_per_k,
            "best_k_by_macro_f1": best_k,
        }
    return results


def _run_acts_cls(method: str, n_layers: int,
                  orcc_positions, y_raw, y_log, y_ruler,
                  acts_base: Path) -> dict:
    results: dict[str, Any] = {}
    pooling   = "mean"
    cleaning  = "tier0"
    y_year_str = np.array([str(int(y)) for y in y_raw])
    layer_iter = _LAYER_SUBSET if _LAYER_SUBSET is not None else range(n_layers)
    for layer in layer_iter:
        if layer >= n_layers:
            continue
        X_full = _load_orcc_activations(method, layer, acts_base)
        if X_full is None:
            print(f"    [{method}] L{layer:02d} — no activations, skip")
            continue
        X = l2_normalize(X_full[orcc_positions])

        for task, y in (("ruler", y_ruler), ("year", y_year_str)):
            m = fit_cls_cv(X, y, cv_strategy="stratified", n_splits=N_SPLITS)
            results[f"{method}__{cleaning}__{pooling}__L{layer:02d}__{task}"] = {
                "method": method, "cleaning": cleaning, "pooling": pooling,
                "layer": layer, "task": task, "n_dropped": 0, **m,
            }
    return results


def _run_acts_ridge_year(method: str, n_layers: int,
                         orcc_positions, y_raw, y_log, y_ruler,
                         acts_base: Path) -> dict:
    """Ridge regression for year (cls_numeric probe). GroupKFold by ruler."""
    results: dict[str, Any] = {}
    pooling  = "mean"
    cleaning = "tier0"
    layer_iter = _LAYER_SUBSET if _LAYER_SUBSET is not None else range(n_layers)
    for layer in layer_iter:
        if layer >= n_layers:
            continue
        X_full = _load_orcc_activations(method, layer, acts_base)
        if X_full is None:
            print(f"    [{method}] L{layer:02d} — no activations, skip")
            continue
        X = l2_normalize(X_full[orcc_positions])
        try:
            ridge_res = fit_ridge_year_groupkfold(
                X, y_raw, y_log, y_ruler, n_splits=N_SPLITS)
        except Exception as e:
            print(f"    [ridge-skip] {method} L{layer:02d}: {type(e).__name__}: {e}", flush=True)
            ridge_res = {yt: {"spearman_mean": float("nan"), "mae_mean": float("nan"),
                               "r2_mean": float("nan"), "skipped": True}
                         for yt in YEAR_TRANSFORMS}
        for yt in YEAR_TRANSFORMS:
            r = ridge_res[yt]
            results[f"{method}__{cleaning}__{pooling}__L{layer:02d}__year-{yt}"] = {
                "method": method, "cleaning": cleaning, "pooling": pooling,
                "layer": layer, "probe": "ridge", "year_transform": yt,
                "n_labeled": int(len(orcc_positions)),
                "n_groups": int(len(np.unique(y_ruler))),
                **r,
            }
    return results


# ---- Dispatch table -------------------------------------------------------

PROBE_DISPATCH: dict[str, Any] = {
    "tfidf_pls":         ("pls",         run_tfidf_pls),
    "tfidf_cls":         ("cls",         run_tfidf_cls),
    "tfidf_cls_numeric": ("cls_numeric", run_tfidf_cls_numeric),
    "mlm_pls":           ("pls",         lambda *a, **kw: _run_acts_pls("mlm",    MLM_N_LAYERS,    *a, **kw)),
    "mlm_cls":           ("cls",         lambda *a, **kw: _run_acts_cls("mlm",    MLM_N_LAYERS,    *a, **kw)),
    "mlm_cls_numeric":   ("cls_numeric", lambda *a, **kw: _run_acts_ridge_year("mlm",    MLM_N_LAYERS,    *a, **kw)),
    "qwen_pls":          ("pls",         lambda *a, **kw: _run_acts_pls("qwen",   QWEN_N_LAYERS,   *a, **kw)),
    "qwen_cls":          ("cls",         lambda *a, **kw: _run_acts_cls("qwen",   QWEN_N_LAYERS,   *a, **kw)),
    "qwen_cls_numeric":  ("cls_numeric", lambda *a, **kw: _run_acts_ridge_year("qwen",   QWEN_N_LAYERS,   *a, **kw)),
    "random_pls":        ("pls",         lambda *a, **kw: _run_acts_pls("random", RANDOM_N_LAYERS, *a, **kw)),
    "random_cls":        ("cls",         lambda *a, **kw: _run_acts_cls("random", RANDOM_N_LAYERS, *a, **kw)),
    "random_cls_numeric":("cls_numeric", lambda *a, **kw: _run_acts_ridge_year("random", RANDOM_N_LAYERS, *a, **kw)),
    # Phase 3: Thalesian (Akkadian-finetuned UMT5) encoder activations.
    "thalesian_akk300m_pls":          ("pls",         lambda *a, **kw: _run_acts_pls("thalesian_akk300m",   THALESIAN_AKK300M_N_LAYERS,   *a, **kw)),
    "thalesian_akk300m_cls":          ("cls",         lambda *a, **kw: _run_acts_cls("thalesian_akk300m",   THALESIAN_AKK300M_N_LAYERS,   *a, **kw)),
    "thalesian_akk300m_cls_numeric":  ("cls_numeric", lambda *a, **kw: _run_acts_ridge_year("thalesian_akk300m",   THALESIAN_AKK300M_N_LAYERS,   *a, **kw)),
    "thalesian_cunei400m_pls":        ("pls",         lambda *a, **kw: _run_acts_pls("thalesian_cunei400m", THALESIAN_CUNEI400M_N_LAYERS, *a, **kw)),
    "thalesian_cunei400m_cls":        ("cls",         lambda *a, **kw: _run_acts_cls("thalesian_cunei400m", THALESIAN_CUNEI400M_N_LAYERS, *a, **kw)),
    "thalesian_cunei400m_cls_numeric":("cls_numeric", lambda *a, **kw: _run_acts_ridge_year("thalesian_cunei400m", THALESIAN_CUNEI400M_N_LAYERS, *a, **kw)),
    # Phase E1: Qwen3 scale sweep (tier0/mean, same draws as Phase 0).
    "qwen3_1b7_pls":         ("pls",         lambda *a, **kw: _run_acts_pls("qwen3_1b7",  QWEN3_1B7_N_LAYERS,  *a, **kw)),
    "qwen3_1b7_cls":         ("cls",         lambda *a, **kw: _run_acts_cls("qwen3_1b7",  QWEN3_1B7_N_LAYERS,  *a, **kw)),
    "qwen3_1b7_cls_numeric": ("cls_numeric", lambda *a, **kw: _run_acts_ridge_year("qwen3_1b7",  QWEN3_1B7_N_LAYERS,  *a, **kw)),
    "qwen3_8b_pls":          ("pls",         lambda *a, **kw: _run_acts_pls("qwen3_8b",   QWEN3_8B_N_LAYERS,   *a, **kw)),
    "qwen3_8b_cls":          ("cls",         lambda *a, **kw: _run_acts_cls("qwen3_8b",   QWEN3_8B_N_LAYERS,   *a, **kw)),
    "qwen3_8b_cls_numeric":  ("cls_numeric", lambda *a, **kw: _run_acts_ridge_year("qwen3_8b",   QWEN3_8B_N_LAYERS,   *a, **kw)),
    "qwen3_32b_pls":        ("pls",         lambda *a, **kw: _run_acts_pls("qwen3_32b", QWEN3_32B_N_LAYERS, *a, **kw)),
    "qwen3_32b_cls":        ("cls",         lambda *a, **kw: _run_acts_cls("qwen3_32b", QWEN3_32B_N_LAYERS, *a, **kw)),
    "qwen3_32b_cls_numeric":("cls_numeric", lambda *a, **kw: _run_acts_ridge_year("qwen3_32b", QWEN3_32B_N_LAYERS, *a, **kw)),
}


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def _aggregate_summary(out_dir: Path, probe: str, method_tag: str) -> dict:
    """Read all draw JSONs for one probe; aggregate macro_f1 / spearman / r2 across draws."""
    pattern = f"{probe}__{method_tag}__draw*.json"
    files = sorted(out_dir.glob(pattern))
    if not files:
        return {"probe": probe, "method_tag": method_tag, "n_draws": 0}

    # Collect: per config_key, list of (macro_f1_mean, accuracy_mean, spearman_mean, r2_mean)
    per_key: dict[str, dict[str, list[float]]] = {}
    for fp in files:
        with open(fp) as f:
            doc = json.load(f)
        results = doc.get("results", {})
        for cfg_key, rec in results.items():
            slot = per_key.setdefault(cfg_key, {
                "macro_f1": [], "accuracy": [], "spearman": [], "r2": [],
            })
            # cls-style schema
            if "macro_f1_mean" in rec:
                slot["macro_f1"].append(rec["macro_f1_mean"])
                slot["accuracy"].append(rec.get("accuracy_mean", float("nan")))
            # pls-style ruler schema (has best_k_by_macro_f1 + metrics_per_k)
            if "best_k_by_macro_f1" in rec:
                bk = str(rec["best_k_by_macro_f1"])
                mpk = rec["metrics_per_k"][bk]
                slot["macro_f1"].append(mpk["macro_f1_mean"])
                slot["accuracy"].append(mpk.get("accuracy_mean", float("nan")))
            # cls_numeric (Ridge) schema — spearman_mean directly at top level
            if "spearman_mean" in rec and "metrics_per_k" not in rec:
                sp = rec.get("spearman_mean")
                r2 = rec.get("r2_mean")
                if sp is not None and not (isinstance(sp, float) and np.isnan(sp)):
                    slot["spearman"].append(sp)
                if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)):
                    slot["r2"].append(r2)
            # pls year schema
            if "best_k_by_spearman" in rec:
                bk_sp = str(rec["best_k_by_spearman"])
                bk_r2 = str(rec["best_k_by_r2"])
                slot["spearman"].append(rec["metrics_per_k"][bk_sp]["spearman_mean"])
                slot["r2"].append(rec["metrics_per_k"][bk_r2]["r2_mean"])

    summary_per_key: dict[str, Any] = {}
    for cfg_key, slot in per_key.items():
        agg: dict[str, Any] = {"n_draws": len(files)}
        for metric, vals in slot.items():
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            if not vals:
                continue
            agg[f"{metric}_mean"]   = float(np.mean(vals))
            agg[f"{metric}_std"]    = float(np.std(vals))
            agg[f"{metric}_median"] = float(np.median(vals))
            agg[f"{metric}_n"]      = len(vals)
        summary_per_key[cfg_key] = agg

    return {
        "probe": probe,
        "method_tag": method_tag,
        "n_draws": len(files),
        "per_config": summary_per_key,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _LAYER_SUBSET
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve layer subset (used by mlm/qwen/random probes only)
    lay_arg = getattr(args, "layers", "all")
    if lay_arg and str(lay_arg).strip().lower() != "all":
        _LAYER_SUBSET = sorted({int(x.strip()) for x in str(lay_arg).split(",") if x.strip()})
        print(f"[layers] activation probes restricted to layers: {_LAYER_SUBSET}")
    else:
        _LAYER_SUBSET = None
        print("[layers] activation probes will scan ALL layers (slow for qwen/random)")

    probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    for p in probes:
        if p not in PROBE_DISPATCH:
            raise SystemExit(f"Unknown probe '{p}'. Valid: {sorted(PROBE_DISPATCH)}")

    # Load draws matrix + fragment order + corpus
    draws_matrix = np.load(args.draws_matrix)
    if draws_matrix.ndim != 2:
        raise SystemExit(f"draws_matrix must be 2D, got shape {draws_matrix.shape}")
    n_draws_total = draws_matrix.shape[0]

    with open(args.fragment_order) as f:
        fragment_order = json.load(f)

    orcc_df = pd.read_parquet(args.corpus)
    if len(orcc_df) != len(fragment_order):
        raise SystemExit(
            f"Corpus length {len(orcc_df)} != fragment_order length {len(fragment_order)}. "
            "Was the corpus parquet rebuilt after balanced_subset was created?"
        )

    draw_indices = _parse_range(args.draws_range, n_draws_total)
    print(f"=== run_mc_probes.py ===")
    print(f"  probes        : {probes}")
    print(f"  draws_matrix  : {args.draws_matrix}  shape={draws_matrix.shape} dtype={draws_matrix.dtype}")
    print(f"  draws to run  : {len(draw_indices)} (range: {draw_indices[0]}..{draw_indices[-1]})")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  method_tag    : {args.method_tag}")
    print(f"  activations   : {args.activations_base}")
    print()

    t_overall = time.time()

    for probe in probes:
        kind, fn = PROBE_DISPATCH[probe]
        print(f"--- probe: {probe} ({kind}) ---")
        t_probe = time.time()
        n_done = n_skipped = n_failed = 0

        for di in draw_indices:
            out_path = args.output_dir / f"{probe}__{args.method_tag}__draw{di:03d}.json"
            if out_path.exists():
                n_skipped += 1
                if n_skipped <= 3 or n_skipped % 50 == 0:
                    print(f"  draw {di:3d}: SKIP (already exists)")
                continue

            orcc_positions, y_raw, y_log, y_ruler, fragment_ids = _draw_subset(
                orcc_df, fragment_order, draws_matrix, di)

            try:
                if probe.startswith("tfidf"):
                    results = fn(orcc_df, orcc_positions, y_raw, y_log, y_ruler, fragment_ids)
                else:
                    results = fn(orcc_positions, y_raw, y_log, y_ruler, args.activations_base)
            except Exception as e:
                print(f"  draw {di:3d}: FAIL — {type(e).__name__}: {e}")
                n_failed += 1
                continue

            doc = {
                "probe":        probe,
                "method_tag":   args.method_tag,
                "draw_idx":     int(di),
                "n_fragments":  int(len(orcc_positions)),
                "fragment_ids": fragment_ids,
                "results":      results,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            n_done += 1
            if n_done <= 3 or n_done % 25 == 0:
                # report headline metric where available
                ruler_keys = [k for k in results if k.endswith("__ruler")]
                head = ""
                if ruler_keys:
                    rec = results[ruler_keys[0]]
                    if "macro_f1_mean" in rec:
                        head = f" macro_f1={rec['macro_f1_mean']:.3f}"
                    elif "best_k_by_macro_f1" in rec:
                        bk = str(rec["best_k_by_macro_f1"])
                        head = f" macro_f1={rec['metrics_per_k'][bk]['macro_f1_mean']:.3f}"
                print(f"  draw {di:3d}: ok{head}  ({len(results)} configs)")

        # rebuild summary from disk (so partial re-runs stay consistent)
        summary = _aggregate_summary(args.output_dir, probe, args.method_tag)
        summary_path = args.output_dir / f"{probe}__{args.method_tag}__summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        dt = (time.time() - t_probe) / 60
        print(f"  {probe}: done={n_done} skipped={n_skipped} failed={n_failed} "
              f"summary_n_draws={summary['n_draws']} ({dt:.1f} min)")
        print()

    print(f"=== ALL PROBES DONE in {(time.time() - t_overall) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
