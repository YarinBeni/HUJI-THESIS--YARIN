"""
Step 5 — Compute PLS regression coords for SEAL+ORCC fragments.

Loads activations for one (method, cleaning, pooling) config, runs GroupKFold
PLS regression against ORCC fragment dates, and projects all 1586 SEAL+ORCC
fragments onto the first 5 PLS components.

CLI
---
  --method {qwen,random}
  --cleaning {tier0,maximal}
  --pooling {mean,last}
  --activations-base   path to results/ root (default: auto from script location)
  --output-dir         (default: results/orcc_round1/pls)
  --year-transforms    raw,log   (comma-separated; default: both)
  --n-components       1,2,3,5  (comma-separated)
  --layers             all | 0,5,10,15,...

Outputs (merged with existing files — never clobbers existing keys):
  pls_results_{method}.json
  pls_projections_{method}.json
"""

import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

_THIS_DIR   = Path(__file__).resolve().parent
_RESULTS_DIR = _THIS_DIR / 'results'

SEAL_PARQUET = _THIS_DIR / '../../data/evaluation/corpora/seal_corpus.parquet'
ORCC_PARQUET = _THIS_DIR / '../../data/evaluation/corpora/orcc_corpus.parquet'

N_LAYERS = 29   # L00..L28


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_seal_dir(base: Path, method: str, cleaning: str, pooling: str) -> Path:
    """SEAL activation dir: mean has no suffix, last has _last."""
    root = base / 'seal_round4' / 'activations'
    if pooling == 'mean':
        return root / f'{method}_{cleaning}'
    return root / f'{method}_{cleaning}_last'


def get_orcc_dir(base: Path, method: str, cleaning: str, pooling: str) -> Path:
    """ORCC activation dir: always carries _{pooling} suffix."""
    root = base / 'orcc_round1' / 'activations'
    return root / f'{method}_{cleaning}_{pooling}'


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_npz(path: Path) -> np.ndarray:
    return np.load(path)['activations'].astype(np.float32)


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def merge_projections(existing: dict, fragment_ids: list,
                      new_embs: dict) -> dict:
    if existing:
        assert existing.get('fragment_ids') == fragment_ids, (
            "Fragment ID mismatch between runs — cannot merge projections!"
        )
        embs = dict(existing.get('embeddings', {}))
    else:
        embs = {}
    embs.update(new_embs)
    return {'fragment_ids': fragment_ids, 'embeddings': embs}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    if not results:
        return
    cols = f"{'Config key':<58} {'sp_k':>4} {'r2_k':>4} {'spearman':>8} {'r2':>7}"
    sep  = '-' * 85
    print(f'\n{"="*85}\nSUMMARY\n{cols}\n{sep}')
    for key in sorted(results):
        rec    = results[key]
        bk_sp  = rec['best_k_by_spearman']
        bk_r2  = rec['best_k_by_r2']
        sp_val = rec['metrics_per_k'][str(bk_sp)]['spearman_mean']
        r2_val = rec['metrics_per_k'][str(bk_r2)]['r2_mean']
        print(f"{key:<58} {bk_sp:>4} {bk_r2:>4} {sp_val:>8.3f} {r2_val:>7.3f}")
    print('=' * 85)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Step 5: Compute PLS coords')
    p.add_argument('--method',    required=True, choices=['qwen', 'random'])
    p.add_argument('--cleaning',  required=True, choices=['tier0', 'maximal'])
    p.add_argument('--pooling',   required=True, choices=['mean', 'last'])
    p.add_argument('--activations-base', type=str, default=None,
                   help='Path to results/ root (default: auto-detected from script location)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output dir (default: results/orcc_round1/pls)')
    p.add_argument('--year-transforms', type=str, default='raw,log',
                   help='Comma-separated year transforms (default: raw,log)')
    p.add_argument('--n-components', type=str, default='1,2,3,5',
                   help='Comma-separated n_components values (default: 1,2,3,5)')
    p.add_argument('--layers', type=str, default='all',
                   help='"all" for L00-L28, or comma-separated indices (default: all)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from pls_utils import l2_normalize, fit_pls_groupkfold, fit_pls_full, project

    args = parse_args()

    base    = Path(args.activations_base) if args.activations_base else _RESULTS_DIR
    out_dir = Path(args.output_dir) if args.output_dir else _RESULTS_DIR / 'orcc_round1' / 'pls'

    year_transforms = [yt.strip() for yt in args.year_transforms.split(',')]
    n_components    = [int(k.strip()) for k in args.n_components.split(',')]
    layers = list(range(N_LAYERS)) if args.layers == 'all' \
             else [int(l.strip()) for l in args.layers.split(',')]

    seal_dir = get_seal_dir(base, args.method, args.cleaning, args.pooling)
    orcc_dir = get_orcc_dir(base, args.method, args.cleaning, args.pooling)

    print(f"=== PLS Pipeline ===")
    print(f"  method={args.method}  cleaning={args.cleaning}  pooling={args.pooling}")
    print(f"  layers={layers}")
    print(f"  year_transforms={year_transforms}  n_components={n_components}")
    print(f"  SEAL acts : {seal_dir}")
    print(f"  ORCC acts : {orcc_dir}")
    print(f"  output    : {out_dir}")

    # ── Load parquets ──────────────────────────────────────────────────────────
    if not SEAL_PARQUET.exists() or not ORCC_PARQUET.exists():
        print("ERROR: Parquet files not found:")
        if not SEAL_PARQUET.exists():
            print(f"  {SEAL_PARQUET}")
        if not ORCC_PARQUET.exists():
            print(f"  {ORCC_PARQUET}")
        sys.exit(1)

    seal_df = pd.read_parquet(SEAL_PARQUET)
    orcc_df = pd.read_parquet(ORCC_PARQUET)
    n_seal  = len(seal_df)
    n_orcc  = len(orcc_df)
    print(f"  SEAL rows: {n_seal}  ORCC rows: {n_orcc}  total: {n_seal + n_orcc}")

    fragment_ids = (
        [str(fid) for fid in seal_df['fragment_id'].tolist()] +
        [str(fid) for fid in orcc_df['fragment_id'].tolist()]
    )

    # Labeled ORCC rows (non-null year)
    labeled_mask    = ~orcc_df['year'].isna()
    labeled_orcc_idx = np.where(labeled_mask)[0]       # indices into orcc_df
    labeled_all_idx  = n_seal + labeled_orcc_idx        # indices into X_all
    n_labeled        = len(labeled_orcc_idx)

    y_raw  = orcc_df['year'].values[labeled_orcc_idx].astype(float)
    y_log  = np.log(y_raw)
    groups = orcc_df['ruler'].values[labeled_orcc_idx].astype(str)
    n_groups = int(len(np.unique(groups)))

    print(f"  n_labeled={n_labeled}  n_groups={n_groups}")

    # ── Load existing outputs for merge ────────────────────────────────────────
    results_path  = out_dir / f'pls_results_{args.method}.json'
    proj_path     = out_dir / f'pls_projections_{args.method}.json'
    results       = load_json(results_path)          # merged in-place below
    proj_existing = load_json(proj_path)
    new_projections: dict = {}

    pooling_infix = '__last' if args.pooling == 'last' else ''
    t_start = time.time()

    # ── Main layer loop ────────────────────────────────────────────────────────
    any_processed = False
    for layer in layers:
        seal_npz = seal_dir / f'layer_{layer:02d}.npz'
        orcc_npz = orcc_dir / f'layer_{layer:02d}.npz'

        missing = [str(p) for p in (seal_npz, orcc_npz) if not p.exists()]
        if missing:
            print(f"  WARNING: Layer {layer:02d} — skipping (not found):")
            for p in missing:
                print(f"    {p}")
            continue

        t_layer = time.time()
        X_seal = load_npz(seal_npz)   # (n_seal, hidden_dim)
        X_orcc = load_npz(orcc_npz)   # (n_orcc, hidden_dim)

        assert X_seal.shape[0] == n_seal, (
            f"Layer {layer:02d} SEAL shape mismatch: {X_seal.shape[0]} vs {n_seal}")
        assert X_orcc.shape[0] == n_orcc, (
            f"Layer {layer:02d} ORCC shape mismatch: {X_orcc.shape[0]} vs {n_orcc}")

        X_all  = np.concatenate([X_seal, X_orcc], axis=0)   # (n_seal+n_orcc, H)
        X_norm = l2_normalize(X_all)                          # (n_seal+n_orcc, H)
        X_lab  = X_norm[labeled_all_idx]                      # (n_labeled, H)

        print(f"\n  Layer {layer:02d}  hidden_dim={X_all.shape[1]}")

        for yt in year_transforms:
            y = y_raw if yt == 'raw' else y_log

            # GroupKFold CV for each k
            metrics_per_k: dict = {}
            for k in n_components:
                mk = fit_pls_groupkfold(X_lab, y, groups, k)
                metrics_per_k[str(k)] = mk

            best_sp = max(n_components,
                          key=lambda k: metrics_per_k[str(k)]['spearman_mean'])
            best_r2 = max(n_components,
                          key=lambda k: metrics_per_k[str(k)]['r2_mean'])

            config_key = (f'{args.method}__{args.cleaning}__{args.pooling}'
                          f'__L{layer:02d}__year-{yt}')
            results[config_key] = {
                'method':            args.method,
                'cleaning':          args.cleaning,
                'pooling':           args.pooling,
                'layer':             layer,
                'year_transform':    yt,
                'n_labeled':         n_labeled,
                'n_groups':          n_groups,
                'metrics_per_k':     metrics_per_k,
                'best_k_by_spearman': best_sp,
                'best_k_by_r2':       best_r2,
            }

            # Full refit with n_components=5, project all fragments
            model5 = fit_pls_full(X_lab, y, n_components=5)
            comps  = project(model5, X_norm)   # (n_seal+n_orcc, 5)

            pk = f'{args.method}__{args.cleaning}__L{layer:02d}{pooling_infix}'
            new_projections[f'{pk}__pls12-{yt}'] = comps[:, 0:2].tolist()
            new_projections[f'{pk}__pls23-{yt}'] = comps[:, 1:3].tolist()
            new_projections[f'{pk}__pls34-{yt}'] = comps[:, 2:4].tolist()

            sp_val = metrics_per_k[str(best_sp)]['spearman_mean']
            r2_val = metrics_per_k[str(best_r2)]['r2_mean']
            print(f"    year={yt}  best_k_sp={best_sp} sp={sp_val:.3f}  "
                  f"best_k_r2={best_r2} r2={r2_val:.3f}")

        print(f"  Layer {layer:02d} done in {time.time() - t_layer:.1f}s")
        any_processed = True

    if not any_processed:
        print("\nNo layers processed (activations not found). "
              "Imports and arg parsing confirmed OK.")
        return

    # ── Write outputs ──────────────────────────────────────────────────────────
    proj_out = merge_projections(proj_existing, fragment_ids, new_projections)
    save_json(results, results_path)
    save_json(proj_out, proj_path)

    print(f"\nResults    → {results_path}  ({len(results)} keys)")
    print(f"Projections→ {proj_path}  ({len(proj_out['embeddings'])} embedding keys)")
    print(f"Wall time  : {(time.time() - t_start) / 60:.1f} min")

    this_run_keys = {
        k: v for k, v in results.items()
        if f'{args.method}__{args.cleaning}__{args.pooling}' in k
    }
    print_summary(this_run_keys)


if __name__ == '__main__':
    main()
