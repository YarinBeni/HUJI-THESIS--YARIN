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
    cols = f"{'Config key':<58} {'bk':>4} {'metric':>10} {'metric2':>10}"
    sep  = '-' * 88
    print(f'\n{"="*88}\nSUMMARY\n{cols}\n{sep}')
    for key in sorted(results):
        rec = results[key]
        if rec.get('target') == 'ruler' or 'best_k_by_macro_f1' in rec:
            bk     = rec['best_k_by_macro_f1']
            m1     = rec['metrics_per_k'][str(bk)]['macro_f1_mean']
            m2     = rec['metrics_per_k'][str(bk)]['accuracy_mean']
            print(f"{key:<58} {bk:>4} {'f1='+f'{m1:.3f}':>10} {'acc='+f'{m2:.3f}':>10}")
        else:
            bk_sp  = rec['best_k_by_spearman']
            bk_r2  = rec['best_k_by_r2']
            sp_val = rec['metrics_per_k'][str(bk_sp)]['spearman_mean']
            r2_val = rec['metrics_per_k'][str(bk_r2)]['r2_mean']
            print(f"{key:<58} {bk_sp:>4} {'sp='+f'{sp_val:.3f}':>10} {'r2='+f'{r2_val:.3f}':>10}")
    print('=' * 88)


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
    p.add_argument('--target', default='year', choices=['year', 'ruler'],
                   help='Prediction target: year (regression) or ruler (PLS-DA classification)')
    p.add_argument('--overwrite', action='store_true',
                   help='Clear existing keys for this method/cleaning/pooling before writing')
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

    y_ruler = orcc_df['ruler'].values[labeled_orcc_idx].astype(str)

    # ── Load existing outputs for merge ────────────────────────────────────────
    results_path  = out_dir / f'pls_results_{args.method}.json'
    proj_path     = out_dir / f'pls_projections_{args.method}.json'
    results       = load_json(results_path)
    proj_existing = load_json(proj_path)

    if args.overwrite:
        prefix = f'{args.method}__{args.cleaning}__{args.pooling}__'
        if args.target == 'year':
            cleared = [k for k in list(results) if k.startswith(prefix) and '__year-' in k]
        else:  # ruler
            cleared = [k for k in list(results) if k.startswith(prefix) and k.endswith('__ruler')]
        for k in cleared:
            del results[k]
        if cleared:
            print(f'  [overwrite] Cleared {len(cleared)} existing keys ({args.target})')
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

        pk = f'{args.method}__{args.cleaning}__L{layer:02d}{pooling_infix}'

        if args.target == 'year':
            for yt in year_transforms:
                y = y_raw if yt == 'raw' else y_log

                metrics_per_k: dict = {}
                for k in n_components:
                    metrics_per_k[str(k)] = fit_pls_groupkfold(X_lab, y, groups, k)

                best_sp = max(n_components,
                              key=lambda k: metrics_per_k[str(k)]['spearman_mean'])
                best_r2 = max(n_components,
                              key=lambda k: metrics_per_k[str(k)]['r2_mean'])

                config_key = (f'{args.method}__{args.cleaning}__{args.pooling}'
                              f'__L{layer:02d}__year-{yt}')
                results[config_key] = {
                    'method':             args.method,
                    'cleaning':           args.cleaning,
                    'pooling':            args.pooling,
                    'layer':              layer,
                    'year_transform':     yt,
                    'n_labeled':          n_labeled,
                    'n_groups':           n_groups,
                    'metrics_per_k':      metrics_per_k,
                    'best_k_by_spearman': best_sp,
                    'best_k_by_r2':       best_r2,
                }

                model5 = fit_pls_full(X_lab, y, n_components=5)
                comps  = project(model5, X_norm)
                new_projections[f'{pk}__pls12-{yt}'] = comps[:, 0:2].tolist()
                new_projections[f'{pk}__pls23-{yt}'] = comps[:, 1:3].tolist()
                new_projections[f'{pk}__pls34-{yt}'] = comps[:, 2:4].tolist()

                sp_val = metrics_per_k[str(best_sp)]['spearman_mean']
                r2_val = metrics_per_k[str(best_r2)]['r2_mean']
                print(f"    year={yt}  best_k_sp={best_sp} sp={sp_val:.3f}  "
                      f"best_k_r2={best_r2} r2={r2_val:.3f}")

        else:  # target == 'ruler'
            from pls_utils import fit_plsda_stratified_kfold, fit_plsda_full

            metrics_per_k = {}
            for k in n_components:
                metrics_per_k[str(k)] = fit_plsda_stratified_kfold(X_lab, y_ruler, k)

            best_k = max(n_components,
                         key=lambda k: metrics_per_k[str(k)]['macro_f1_mean'])

            config_key = (f'{args.method}__{args.cleaning}__{args.pooling}'
                          f'__L{layer:02d}__ruler')
            results[config_key] = {
                'method':             args.method,
                'cleaning':           args.cleaning,
                'pooling':            args.pooling,
                'layer':              layer,
                'target':             'ruler',
                'n_labeled':          n_labeled,
                'metrics_per_k':      metrics_per_k,
                'best_k_by_macro_f1': best_k,
            }

            model_da = fit_plsda_full(X_lab, y_ruler, n_components=5)
            comps_da = project(model_da, X_norm)
            new_projections[f'{pk}__plsda12'] = comps_da[:, 0:2].tolist()

            best_acc = metrics_per_k[str(best_k)]['accuracy_mean']
            best_f1  = metrics_per_k[str(best_k)]['macro_f1_mean']
            print(f"    ruler  best_k={best_k} acc={best_acc:.3f} macro_f1={best_f1:.3f}")

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
