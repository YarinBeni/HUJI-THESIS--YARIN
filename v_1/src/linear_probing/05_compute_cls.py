"""
Step 5 (classification) — Predict ruler or year from hidden-state activations.

Two tasks:
  ruler — 38-class, StratifiedKFold(5)
  year  — 45-class (year as discrete label), StratifiedKFold(5)

Both use LogisticRegression on L2-normalised activations.
Rare classes (fewer than --min-count fragments) are dropped before CV
so that StratifiedKFold can always put ≥1 example per fold per class.

CLI
---
  --method    {qwen,random}
  --cleaning  {tier0,maximal}
  --pooling   {mean,last}
  --tasks     ruler,year  (default: both)
  --layers    all | 0,5,10,...
  --min-count minimum fragments per class to include (default: 5)
  --C         LogisticRegression regularisation (default: 1.0)
  --output-dir  (default: results/orcc_round1/cls)

Output (merged with existing file):
  cls_results_{method}.json
"""

import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

_THIS_DIR    = Path(__file__).resolve().parent
_RESULTS_DIR = _THIS_DIR / 'results'

SEAL_PARQUET = _THIS_DIR / '../../data/evaluation/corpora/seal_corpus.parquet'
ORCC_PARQUET = _THIS_DIR / '../../data/evaluation/corpora/orcc_corpus.parquet'

N_LAYERS = 29  # L00–L28


# ---------------------------------------------------------------------------
# Path helpers (identical to 05_compute_pls.py)
# ---------------------------------------------------------------------------

def get_seal_dir(base, method, cleaning, pooling):
    root = base / 'seal_round4' / 'activations'
    return root / (f'{method}_{cleaning}' if pooling == 'mean'
                   else f'{method}_{cleaning}_last')


def get_orcc_dir(base, method, cleaning, pooling):
    return base / 'orcc_round1' / 'activations' / f'{method}_{cleaning}_{pooling}'


def load_npz(path):
    return np.load(path)['activations'].astype(np.float32)


def load_json(path):
    return json.load(open(path)) if path.exists() else {}


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method',    required=True, choices=['qwen', 'random'])
    p.add_argument('--cleaning',  required=True, choices=['tier0', 'maximal'])
    p.add_argument('--pooling',   required=True, choices=['mean', 'last'])
    p.add_argument('--tasks',     default='ruler,year')
    p.add_argument('--layers',    default='all')
    p.add_argument('--min-count', type=int, default=5,
                   help='Min fragments per class (ruler or year) to include')
    p.add_argument('--C',         type=float, default=1.0)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--activations-base', default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from pls_utils import l2_normalize
    from cls_utils import fit_cls_cv

    args    = parse_args()
    base    = Path(args.activations_base) if args.activations_base else _RESULTS_DIR
    out_dir = Path(args.output_dir) if args.output_dir else _RESULTS_DIR / 'orcc_round1' / 'cls'
    tasks   = [t.strip() for t in args.tasks.split(',')]
    layers  = list(range(N_LAYERS)) if args.layers == 'all' \
              else [int(l.strip()) for l in args.layers.split(',')]

    seal_dir = get_seal_dir(base, args.method, args.cleaning, args.pooling)
    orcc_dir = get_orcc_dir(base, args.method, args.cleaning, args.pooling)

    print(f"=== CLS Probe: {args.method} {args.cleaning} {args.pooling} ===")
    print(f"  Tasks: {tasks}  Layers: {layers}  min_count={args.min_count}")

    orcc_df = pd.read_parquet(ORCC_PARQUET)
    seal_df = pd.read_parquet(SEAL_PARQUET)
    n_seal, n_orcc = len(seal_df), len(orcc_df)

    # Labeled ORCC rows (non-null year)
    labeled_mask    = ~orcc_df['year'].isna()
    labeled_orcc_idx = np.where(labeled_mask)[0]
    labeled_all_idx  = n_seal + labeled_orcc_idx

    orcc_sub = orcc_df.iloc[labeled_orcc_idx]

    # --- Build label arrays per task (with min-count filter) ---
    task_labels = {}
    for task in tasks:
        col = 'ruler' if task == 'ruler' else 'year'
        raw = orcc_sub[col].astype(str).values
        counts = pd.Series(raw).value_counts()
        keep_classes = counts[counts >= args.min_count].index
        mask = np.isin(raw, keep_classes)
        task_labels[task] = {
            'y':           raw[mask],
            'local_idx':   np.where(mask)[0],           # within labeled set
            'global_idx':  labeled_all_idx[mask],        # within full SEAL+ORCC array
            'n_classes':   int(keep_classes.shape[0]),
            'n_dropped':   int((~mask).sum()),
        }
        print(f"  Task '{task}': {mask.sum()} fragments, "
              f"{task_labels[task]['n_classes']} classes "
              f"(dropped {task_labels[task]['n_dropped']} with <{args.min_count} fragments)")

    # Load existing results for merge
    results_path = out_dir / f'cls_results_{args.method}.json'
    results      = load_json(results_path)

    t_start = time.time()
    any_processed = False

    for layer in layers:
        seal_npz = seal_dir / f'layer_{layer:02d}.npz'
        orcc_npz = orcc_dir / f'layer_{layer:02d}.npz'

        missing = [str(p) for p in (seal_npz, orcc_npz) if not p.exists()]
        if missing:
            print(f"  WARNING: Layer {layer:02d} — skipping (not found): {missing}")
            continue

        X_seal = load_npz(seal_npz)
        X_orcc = load_npz(orcc_npz)
        X_all  = np.concatenate([X_seal, X_orcc], axis=0)
        X_norm = l2_normalize(X_all)

        print(f"\n  Layer {layer:02d}")

        for task, info in task_labels.items():
            X_task = X_norm[info['global_idx']]
            y_task = info['y']

            m = fit_cls_cv(X_task, y_task, cv_strategy='stratified', n_splits=5, C=args.C)

            config_key = (f'{args.method}__{args.cleaning}__{args.pooling}'
                          f'__L{layer:02d}__{task}')
            results[config_key] = {
                'method':     args.method,
                'cleaning':   args.cleaning,
                'pooling':    args.pooling,
                'layer':      layer,
                'task':       task,
                'n_dropped':  info['n_dropped'],
                **m,
            }
            print(f"    {task}: acc={m['accuracy_mean']:.3f} "
                  f"macro_f1={m['macro_f1_mean']:.3f} "
                  f"(chance acc={m['chance_accuracy']:.3f} "
                  f"f1={m['chance_macro_f1']:.3f})")

        any_processed = True

    if not any_processed:
        print("No layers processed.")
        return

    save_json(results, results_path)
    print(f"\nResults → {results_path}  ({len(results)} keys)")
    print(f"Wall time: {(time.time() - t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
