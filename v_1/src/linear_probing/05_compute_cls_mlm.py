"""
05_compute_cls_mlm.py — Classification probe for the Akkadian MLM baseline.

Mirrors 05_compute_cls.py but for MLM:
  - tier0 cleaning only, mean pooling only
  - 17 layers: L00–L16, hidden_dim=384

Output: results/orcc_round1/cls/cls_results_mlm.json
"""

import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

_THIS_DIR    = Path(__file__).resolve().parent
_RESULTS_DIR = _THIS_DIR / 'results'

SEAL_PARQUET  = Path('v_1/data/evaluation/corpora/seal_corpus.parquet')
ORCC_PARQUET  = Path('v_1/data/evaluation/corpora/orcc_corpus.parquet')
SEAL_ACTS_DIR = _RESULTS_DIR / 'seal_round4' / 'activations' / 'mlm_tier0'
ORCC_ACTS_DIR = _RESULTS_DIR / 'orcc_round1' / 'activations' / 'mlm_tier0'
OUT_DIR       = _RESULTS_DIR / 'orcc_round1' / 'cls'

ALL_LAYERS = list(range(17))
MIN_COUNT  = 5   # min fragments per class


def main():
    sys.path.insert(0, str(_THIS_DIR))
    from pls_utils import l2_normalize
    from cls_utils import fit_cls_cv

    seal_df = pd.read_parquet(SEAL_PARQUET)
    orcc_df = pd.read_parquet(ORCC_PARQUET)
    n_seal  = len(seal_df)

    labeled_mask     = ~orcc_df['year'].isna()
    labeled_orcc_idx = np.where(labeled_mask)[0]
    labeled_all_idx  = n_seal + labeled_orcc_idx
    orcc_sub         = orcc_df.iloc[labeled_orcc_idx]

    task_labels = {}
    for task, col in [('ruler', 'ruler'), ('year', 'year')]:
        raw    = orcc_sub[col].astype(str).values
        counts = pd.Series(raw).value_counts()
        keep   = counts[counts >= MIN_COUNT].index
        mask   = np.isin(raw, keep)
        task_labels[task] = {
            'y':          raw[mask],
            'global_idx': labeled_all_idx[mask],
            'n_dropped':  int((~mask).sum()),
            'n_classes':  int(keep.shape[0]),
        }
        print(f"Task '{task}': {mask.sum()} fragments, "
              f"{task_labels[task]['n_classes']} classes "
              f"(dropped {task_labels[task]['n_dropped']} with <{MIN_COUNT})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / 'cls_results_mlm.json'
    results: dict = json.load(open(results_path)) if results_path.exists() else {}

    t_start = time.time()
    for layer in ALL_LAYERS:
        seal_npz = SEAL_ACTS_DIR / f'layer_{layer:02d}.npz'
        orcc_npz = ORCC_ACTS_DIR / f'layer_{layer:02d}.npz'
        if not seal_npz.exists() or not orcc_npz.exists():
            print(f"  Layer {layer:02d} — skipping (file not found)")
            continue

        X_all  = np.concatenate([
            np.load(seal_npz)['activations'].astype(np.float32),
            np.load(orcc_npz)['activations'].astype(np.float32),
        ], axis=0)
        X_norm = l2_normalize(X_all)

        print(f"\n  Layer {layer:02d}")
        for task, info in task_labels.items():
            X_task = X_norm[info['global_idx']]
            m = fit_cls_cv(X_task, info['y'], cv_strategy='stratified', n_splits=5)
            config_key = f'mlm__tier0__mean__L{layer:02d}__{task}'
            results[config_key] = {
                'method': 'mlm', 'cleaning': 'tier0', 'pooling': 'mean',
                'layer': layer, 'task': task, 'n_dropped': info['n_dropped'], **m,
            }
            print(f"    {task}: acc={m['accuracy_mean']:.3f}  "
                  f"macro_f1={m['macro_f1_mean']:.3f}  "
                  f"chance_acc={m['chance_accuracy']:.3f}")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} configs → {results_path}")
    print(f"Wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
