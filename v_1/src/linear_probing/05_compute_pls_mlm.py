#!/usr/bin/env python3
"""
05_compute_pls_mlm.py — PLS regression sweep over all MLM layers.

Mirrors 05_compute_pls.py (Qwen/Random) but targets the Akkadian MLM baseline:
  - Only tier0 cleaning (MLM text column is text_tier0 / "text"; no maximal variant)
  - Only mean pooling (last-token not extracted)
  - 17 layers: L00 (embedding) through L16 (final transformer block)

Reads pre-extracted activations from:
  results/seal_round4/activations/mlm_tier0/layer_NN.npz
  results/orcc_round1/activations/mlm_tier0/layer_NN.npz

Outputs (in results/orcc_round1/pls/):
  pls_results_mlm.json      — CV metrics per config
  pls_projections_mlm.json  — 5-component projections for all 1586 fragments

Config key format:
  mlm__tier0__mean__L{NN:02d}__year-{raw|log}

Usage (from repo root):
  python v_1/src/linear_probing/05_compute_pls_mlm.py
  python v_1/src/linear_probing/05_compute_pls_mlm.py --cleaning tier0 --pooling mean
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from utils import RESULTS_DIR

from pls_utils import (  # noqa: E402
    l2_normalize,
    fit_pls_groupkfold,  # CV for a single n_components value
    fit_pls_full,        # refit on full labeled set → PLSRegression
    project,             # project(model, X) → (N, n_components)
)

# ---------------------------------------------------------------------------
ORCC_PARQUET = Path("v_1/data/evaluation/corpora/orcc_corpus.parquet")
SEAL_PARQUET = Path("v_1/data/evaluation/corpora/seal_corpus.parquet")

SEAL_ACTS_DIR = RESULTS_DIR / "seal_round4" / "activations" / "mlm_tier0"
ORCC_ACTS_DIR = RESULTS_DIR / "orcc_round1" / "activations" / "mlm_tier0"
OUT_DIR       = RESULTS_DIR / "orcc_round1" / "pls"

N_SEAL = 384
N_ORCC = 1202      # total ORCC rows; 893 have non-null year
ALL_LAYERS = list(range(17))   # L00–L16

N_COMPONENTS_LIST = [1, 2, 3, 5]
YEAR_TRANSFORMS   = ["raw", "log"]
REFIT_K           = 5    # always refit with 5 components for projections
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="PLS sweep over all MLM layers (tier0, mean pooling)")
    p.add_argument("--cleaning", choices=["tier0"], default="tier0",
                   help="Only 'tier0' is supported for MLM")
    p.add_argument("--pooling", choices=["mean"], default="mean",
                   help="Only 'mean' is supported for MLM (last not extracted)")
    p.add_argument("--layers", default="all",
                   help="'all' (default) or comma-separated layer indices, e.g. 0,4,8")
    p.add_argument("--year-transforms", default="raw,log",
                   help="Comma-separated transforms: raw,log (default: raw,log)")
    p.add_argument("--n-components", default="1,2,3,5",
                   help="Comma-separated PLS component counts (default: 1,2,3,5)")
    p.add_argument("--output-dir", default=str(OUT_DIR),
                   help="Output directory for pls_results_mlm.json + pls_projections_mlm.json")
    return p.parse_args()


def load_activations(acts_dir: Path, layer: int) -> np.ndarray:
    """Load (N, hidden_dim) float32 array from layer_NN.npz."""
    npz_path = acts_dir / f"layer_{layer:02d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Activation file not found: {npz_path}\n"
            "Run extract_mlm_all_layers.sh first."
        )
    return np.load(npz_path)["activations"].astype(np.float32)


def load_fragment_ids(acts_dir: Path) -> list:
    """Read fragment_ids from metadata.json; fall back to integer strings."""
    meta_path = acts_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)["fragment_ids"]
    return None


def validate_layer_availability(acts_dir: Path, layers: list) -> None:
    missing = [l for l in layers
               if not (acts_dir / f"layer_{l:02d}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing layers in {acts_dir}: {missing}\n"
            "Run extract_mlm_all_layers.sh first."
        )


def main():
    args = parse_args()

    # ── Validate args ────────────────────────────────────────────────────────
    if args.cleaning != "tier0":
        print("ERROR: MLM only supports --cleaning tier0 "
              "(no maximal variant was extracted).", file=sys.stderr)
        sys.exit(1)
    if args.pooling != "mean":
        print("ERROR: MLM only supports --pooling mean "
              "(last-token activations were not extracted).", file=sys.stderr)
        sys.exit(1)

    layers = (ALL_LAYERS if args.layers == "all"
              else [int(x) for x in args.layers.split(",")])
    year_transforms   = [t.strip() for t in args.year_transforms.split(",")]
    n_components_list = [int(x) for x in args.n_components.split(",")]
    out_dir           = Path(args.output_dir)

    print(f"Layers:          L{layers[0]:02d}–L{layers[-1]:02d} ({len(layers)} total)")
    print(f"Year transforms: {year_transforms}")
    print(f"PLS components:  {n_components_list}")
    print(f"Output dir:      {out_dir}")

    # ── Validate activation files exist ─────────────────────────────────────
    validate_layer_availability(SEAL_ACTS_DIR, layers)
    validate_layer_availability(ORCC_ACTS_DIR, layers)

    # ── Load corpora ─────────────────────────────────────────────────────────
    seal_df = pd.read_parquet(SEAL_PARQUET)
    orcc_df = pd.read_parquet(ORCC_PARQUET)

    seal_ids = (seal_df["fragment_id"].astype(str).tolist()
                if "fragment_id" in seal_df.columns
                else [str(i) for i in range(len(seal_df))])
    orcc_ids = (orcc_df["fragment_id"].astype(str).tolist()
                if "fragment_id" in orcc_df.columns
                else [str(i) for i in range(len(orcc_df))])

    fragment_ids_all = seal_ids + orcc_ids   # SEAL first, then ORCC
    N_all = len(fragment_ids_all)            # 1586 = 384 + 1202

    # ORCC labels (year, ruler) — indices into the combined [SEAL + ORCC] array
    orcc_offset = len(seal_df)              # 384
    year_series  = orcc_df["year"]
    ruler_series = orcc_df["ruler"]

    labeled_mask_orcc = year_series.notna().values   # (N_ORCC,) bool
    labeled_orcc_idx  = np.where(labeled_mask_orcc)[0]     # indices into orcc_df
    labeled_all_idx   = labeled_orcc_idx + orcc_offset     # indices into combined array

    y_raw = year_series.values[labeled_orcc_idx].astype(float)
    y_log = np.log(y_raw)
    groups = ruler_series.values[labeled_orcc_idx].astype(str)

    n_labeled = len(labeled_all_idx)
    n_groups  = int(pd.Series(groups).nunique())
    print(f"\nCorpus: {len(seal_df)} SEAL + {len(orcc_df)} ORCC = {N_all} total")
    print(f"Labeled: {n_labeled} ORCC fragments with non-null year")
    print(f"Groups (rulers): {n_groups} unique")

    # ── Output structures ────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    pls_results: dict = {}
    pls_embeddings: dict = {}

    method   = "mlm"
    cleaning = "tier0"
    pooling  = "mean"

    n_configs = len(layers) * len(year_transforms)
    done = 0

    for layer in layers:
        layer_tag = f"L{layer:02d}"
        print(f"\n{'='*60}")
        print(f"Layer {layer_tag}  [{done+1}–{done+len(year_transforms)}/{n_configs}]")
        print(f"{'='*60}")

        # Load activations for this layer (SEAL + ORCC concatenated)
        X_seal = load_activations(SEAL_ACTS_DIR, layer)
        X_orcc = load_activations(ORCC_ACTS_DIR, layer)
        X_all  = np.concatenate([X_seal, X_orcc], axis=0)   # (1586, 384)

        assert X_all.shape[0] == N_all, (
            f"Expected {N_all} rows, got {X_all.shape[0]}"
        )

        # L2-normalize all rows
        X_all = l2_normalize(X_all)

        X_labeled = X_all[labeled_all_idx]    # (893, 384)

        for year_transform in year_transforms:
            y = y_raw if year_transform == "raw" else y_log
            config_key = f"{method}__{cleaning}__{pooling}__{layer_tag}__year-{year_transform}"
            print(f"\n  Config: {config_key}")

            # CV sweep: one call to fit_pls_groupkfold per k value
            metrics_per_k = {}
            for k in n_components_list:
                metrics_per_k[str(k)] = fit_pls_groupkfold(
                    X_labeled, y, groups, n_components=k
                )

            # Pick best k by spearman and r2 (using mean across folds)
            best_k_by_spearman = max(
                n_components_list,
                key=lambda k: metrics_per_k[str(k)]["spearman_mean"],
            )
            best_k_by_r2 = max(
                n_components_list,
                key=lambda k: metrics_per_k[str(k)]["r2_mean"],
            )

            pls_results[config_key] = {
                "method":             method,
                "cleaning":           cleaning,
                "pooling":            pooling,
                "layer":              layer,
                "year_transform":     year_transform,
                "n_labeled":          n_labeled,
                "n_groups":           n_groups,
                "metrics_per_k":      metrics_per_k,
                "best_k_by_spearman": best_k_by_spearman,
                "best_k_by_r2":       best_k_by_r2,
            }

            best_sp = metrics_per_k[str(best_k_by_spearman)]["spearman_mean"]
            print(f"  best_k_spearman={best_k_by_spearman} (rho={best_sp:.3f})  "
                  f"best_k_r2={best_k_by_r2}")

            # Refit on full labeled set with REFIT_K=5 components; project all rows
            pls_model = fit_pls_full(X_labeled, y, n_components=REFIT_K)
            proj = project(pls_model, X_all)   # (N_all, REFIT_K=5)

            embed_prefix = f"{method}__{cleaning}__{layer_tag}"
            pls_embeddings[f"{embed_prefix}__pls12-{year_transform}"] = (
                proj[:, [0, 1]].tolist()
            )
            pls_embeddings[f"{embed_prefix}__pls23-{year_transform}"] = (
                proj[:, [1, 2]].tolist()
            )
            pls_embeddings[f"{embed_prefix}__pls34-{year_transform}"] = (
                proj[:, [2, 3]].tolist()
            )

            done += 1

    # ── Save outputs ─────────────────────────────────────────────────────────
    results_path     = out_dir / "pls_results_mlm.json"
    projections_path = out_dir / "pls_projections_mlm.json"

    with open(results_path, "w") as f:
        json.dump(pls_results, f, indent=2)
    print(f"\nResults saved → {results_path}  ({len(pls_results)} configs)")

    projections_out = {
        "fragment_ids": fragment_ids_all,
        "embeddings":   pls_embeddings,
    }
    with open(projections_path, "w") as f:
        json.dump(projections_out, f, indent=2)
    print(f"Projections saved → {projections_path}  ({len(pls_embeddings)} embedding keys)")
    print("✅ Done")


if __name__ == "__main__":
    main()
