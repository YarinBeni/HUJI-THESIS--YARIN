"""probe_thalesian.py — Round 2 Phase 3: CLS + PLS probes for Thalesian encoders.

Drives the Round-1 CLS and PLS probes on Thalesian (Akkadian-finetuned UMT5)
encoder hidden states. Produces results JSONs in the same dirs and the same
config_key format as Round 1, so the existing aggregators (06_aggregate_cls.py /
06_aggregate_pls.py) pick them up after the method name is registered there.

Key differences from 05_compute_cls.py / 05_compute_pls.py:
  * No SEAL concatenation. L2-normalize is row-wise, so per-row results are
    identical with or without SEAL rows in the matrix.
  * Method name is free-form (not restricted to {qwen,random}).
  * Number of encoder layers is read from the activation dir's metadata.json
    instead of being hard-coded — AKK_300m has 9 layers (8 + emb),
    cuneiformBase-400m has 13 (12 + emb).

CLI
---
  --method      free-form, e.g. thalesian_akk300m
  --cleaning    {tier0, maximal}
  --pooling     {mean, last}
  --target      {cls, pls, both}   (default: both)
  --activations-base   path to results/ (default: auto)
  --output-cls-dir  default: results/orcc__probe_cls
  --output-pls-dir  default: results/orcc__probe_pls
  --layers      "all" (default) or comma-separated layer indices

Outputs
-------
  {output_cls_dir}/cls_results_{method}.json
  {output_pls_dir}/pls_results_{method}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR    = Path(__file__).resolve().parent           # round2_phase3/
_PROBES_DIR  = _THIS_DIR.parent                          # linear_probing/
_RESULTS_DIR = _PROBES_DIR / 'results'
_REPO_ROOT   = _PROBES_DIR.parents[2]                    # lititure-review/

ORCC_PARQUET = _REPO_ROOT / 'v_1/data/evaluation/corpora/orcc_corpus.parquet'

MIN_COUNT       = 5         # min frags per class to include (mirrors Round 1)
PLS_N_COMPONENTS = [1, 2, 3, 5]
YEAR_TRANSFORMS = ['raw', 'log']
N_SPLITS        = 5


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def orcc_acts_dir(base: Path, method: str, cleaning: str, pooling: str) -> Path:
    """Mirrors Round 1's ORCC layout: {method}_{cleaning}_{pooling}/."""
    return base / 'orcc__embed' / 'activations' / f'{method}_{cleaning}_{pooling}'


def load_metadata(acts_dir: Path) -> dict:
    meta_path = acts_dir / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {acts_dir}")
    with open(meta_path) as f:
        return json.load(f)


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method',   required=True,
                   help='Method tag (free-form), e.g. thalesian_akk300m')
    p.add_argument('--cleaning', required=True, choices=['tier0', 'maximal'])
    p.add_argument('--pooling',  required=True, choices=['mean', 'last'])
    p.add_argument('--target',   default='both', choices=['cls', 'pls', 'both'],
                   help='Which probe to run (default: both)')
    p.add_argument('--activations-base', type=Path, default=None,
                   help='Activations root (default: results/ next to this script)')
    p.add_argument('--output-cls-dir', type=Path, default=None,
                   help='CLS results dir (default: results/orcc__probe_cls)')
    p.add_argument('--output-pls-dir', type=Path, default=None,
                   help='PLS results dir (default: results/orcc__probe_pls)')
    p.add_argument('--layers', default='all',
                   help='"all" (default) or comma-separated layer indices')
    p.add_argument('--min-count', type=int, default=MIN_COUNT,
                   help=f'Min fragments per class to include (default: {MIN_COUNT})')
    p.add_argument('--C', type=float, default=1.0,
                   help='Logistic-regression C for CLS probe (default: 1.0)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Label preparation (mirrors 05_compute_cls.py / 05_compute_pls.py)
# ---------------------------------------------------------------------------

def build_task_labels(orcc_df: pd.DataFrame, min_count: int) -> tuple[np.ndarray, dict]:
    """Return labeled-row positions in orcc_df, plus per-task label arrays.

    The "labeled" subset = rows with non-null year. Round-1 scripts filter the
    same way (only year-labeled fragments enter either probe).
    """
    labeled_mask = ~orcc_df['year'].isna()
    labeled_idx  = np.where(labeled_mask)[0]
    orcc_sub     = orcc_df.iloc[labeled_idx]

    y_raw   = orcc_sub['year'].values.astype(float)
    y_log   = np.log(y_raw)
    y_ruler = orcc_sub['ruler'].astype(str).values

    # CLS per-task min-count filter (Round 1 does this independently per task)
    cls_task_info: dict[str, dict] = {}
    for task, raw in [('ruler', y_ruler), ('year', orcc_sub['year'].astype(str).values)]:
        counts = pd.Series(raw).value_counts()
        keep   = counts[counts >= min_count].index
        keep_mask = np.isin(raw, keep)
        cls_task_info[task] = {
            'y':         raw[keep_mask],
            'local_idx': np.where(keep_mask)[0],     # within labeled subset
            'n_classes': int(keep.shape[0]),
            'n_dropped': int((~keep_mask).sum()),
        }

    # PLS bundles
    pls_info = {
        'y_raw':   y_raw,
        'y_log':   y_log,
        'y_ruler': y_ruler,
        'groups':  y_ruler,                          # GroupKFold by ruler
        'n_labeled': int(len(labeled_idx)),
        'n_groups':  int(np.unique(y_ruler).size),
    }
    return labeled_idx, {'cls': cls_task_info, 'pls': pls_info}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(_PROBES_DIR))  # so pls_utils / cls_utils are importable

    from pls_utils import l2_normalize, fit_pls_groupkfold, fit_plsda_stratified_kfold
    from cls_utils import fit_cls_cv

    base    = args.activations_base if args.activations_base else _RESULTS_DIR
    cls_dir = args.output_cls_dir if args.output_cls_dir else _RESULTS_DIR / 'orcc__probe_cls'
    pls_dir = args.output_pls_dir if args.output_pls_dir else _RESULTS_DIR / 'orcc__probe_pls'

    acts_dir = orcc_acts_dir(base, args.method, args.cleaning, args.pooling)
    if not acts_dir.is_dir():
        print(f"ERROR: activations dir not found: {acts_dir}", file=sys.stderr)
        sys.exit(1)

    meta = load_metadata(acts_dir)
    n_layers = int(meta['n_layers'])
    hidden_dim = int(meta['hidden_dim'])

    if args.layers == 'all':
        layers = list(range(n_layers))
    else:
        layers = [int(s.strip()) for s in args.layers.split(',')]
        layers = [l for l in layers if 0 <= l < n_layers]

    print(f"=== Thalesian probe: {args.method} {args.cleaning} {args.pooling} ===")
    print(f"  acts_dir = {acts_dir}")
    print(f"  n_layers = {n_layers}  hidden_dim = {hidden_dim}")
    print(f"  layers   = {layers}")
    print(f"  target   = {args.target}")

    orcc_df = pd.read_parquet(ORCC_PARQUET)
    labeled_idx, label_bundle = build_task_labels(orcc_df, args.min_count)
    cls_info = label_bundle['cls']
    pls_info = label_bundle['pls']

    print(f"  n_labeled = {pls_info['n_labeled']}  n_groups = {pls_info['n_groups']}")
    for task, info in cls_info.items():
        print(f"  CLS task '{task}': {len(info['y'])} fragments, {info['n_classes']} classes "
              f"(dropped {info['n_dropped']} <{args.min_count})")

    # Load existing result files (merge, do not clobber)
    cls_results_path = cls_dir / f'cls_results_{args.method}.json'
    pls_results_path = pls_dir / f'pls_results_{args.method}.json'
    cls_results = load_json(cls_results_path)
    pls_results = load_json(pls_results_path)

    run_cls = args.target in ('cls', 'both')
    run_pls = args.target in ('pls', 'both')

    t_start = time.time()
    for layer in layers:
        npz = acts_dir / f'layer_{layer:02d}.npz'
        if not npz.exists():
            print(f"  WARNING: layer_{layer:02d}.npz not found in {acts_dir}; skip")
            continue

        X_full = np.load(npz)['activations'].astype(np.float32)
        assert X_full.shape[0] == len(orcc_df), (
            f"layer_{layer:02d}.npz row count {X_full.shape[0]} != ORCC rows {len(orcc_df)}"
        )

        # Row-wise L2 norm (per-row → SEAL concat would be a no-op for ORCC rows)
        X_norm   = l2_normalize(X_full)
        X_labeled = X_norm[labeled_idx]                  # (n_labeled, H)

        print(f"\n  Layer {layer:02d}")

        # -------- CLS --------
        if run_cls:
            for task, info in cls_info.items():
                X_task = X_labeled[info['local_idx']]
                m = fit_cls_cv(X_task, info['y'],
                               cv_strategy='stratified', n_splits=N_SPLITS, C=args.C)
                cfg_key = f'{args.method}__{args.cleaning}__{args.pooling}__L{layer:02d}__{task}'
                cls_results[cfg_key] = {
                    'method':    args.method,
                    'cleaning':  args.cleaning,
                    'pooling':   args.pooling,
                    'layer':     layer,
                    'task':      task,
                    'n_dropped': info['n_dropped'],
                    'n_classes': info['n_classes'],
                    **m,
                }
                print(f"    CLS {task}: acc={m['accuracy_mean']:.3f}  "
                      f"macro_f1={m['macro_f1_mean']:.3f}  "
                      f"(chance acc={m['chance_accuracy']:.3f} f1={m['chance_macro_f1']:.3f})")

        # -------- PLS --------
        if run_pls:
            # Year regression (raw + log) — defensive try/except around each k:
            # PLS NIPALS can divide by zero / produce non-finite SVD inputs when
            # X is rank-deficient (e.g. L0 with last-token pooling on padded
            # sequences). Mirror reprobe_pv.py:323-335 — skip with NaN metrics.
            nan_year_metrics = {
                'spearman_mean': float('nan'), 'spearman_std': float('nan'),
                'mae_mean': float('nan'), 'mae_std': float('nan'),
                'r2_mean': float('nan'), 'r2_std': float('nan'),
                'skipped': True,
            }
            for yt in YEAR_TRANSFORMS:
                y = pls_info['y_raw'] if yt == 'raw' else pls_info['y_log']
                metrics_per_k = {}
                for k in PLS_N_COMPONENTS:
                    try:
                        metrics_per_k[str(k)] = fit_pls_groupkfold(X_labeled, y, pls_info['groups'], k)
                    except Exception as e:
                        print(f"    [pls-skip] k={k} year-{yt}: {type(e).__name__}: {e}", flush=True)
                        metrics_per_k[str(k)] = {**nan_year_metrics,
                                                  'error': f"{type(e).__name__}: {e}"}
                # Pick best k among non-NaN entries, fall back to first if all NaN.
                valid_sp = [k for k in PLS_N_COMPONENTS
                            if not (isinstance(metrics_per_k[str(k)].get('spearman_mean'), float)
                                    and np.isnan(metrics_per_k[str(k)]['spearman_mean']))]
                valid_r2 = [k for k in PLS_N_COMPONENTS
                            if not (isinstance(metrics_per_k[str(k)].get('r2_mean'), float)
                                    and np.isnan(metrics_per_k[str(k)]['r2_mean']))]
                best_sp = (max(valid_sp, key=lambda k: metrics_per_k[str(k)]['spearman_mean'])
                           if valid_sp else PLS_N_COMPONENTS[0])
                best_r2 = (max(valid_r2, key=lambda k: metrics_per_k[str(k)]['r2_mean'])
                           if valid_r2 else PLS_N_COMPONENTS[0])
                cfg_key = (f'{args.method}__{args.cleaning}__{args.pooling}'
                           f'__L{layer:02d}__year-{yt}')
                pls_results[cfg_key] = {
                    'method':             args.method,
                    'cleaning':           args.cleaning,
                    'pooling':            args.pooling,
                    'layer':              layer,
                    'year_transform':     yt,
                    'n_labeled':          pls_info['n_labeled'],
                    'n_groups':           pls_info['n_groups'],
                    'metrics_per_k':      metrics_per_k,
                    'best_k_by_spearman': best_sp,
                    'best_k_by_r2':       best_r2,
                }
                sp_val = metrics_per_k[str(best_sp)]['spearman_mean']
                r2_val = metrics_per_k[str(best_r2)]['r2_mean']
                print(f"    PLS year={yt}  best_k_sp={best_sp} sp={sp_val:.3f}  "
                      f"best_k_r2={best_r2} r2={r2_val:.3f}")

            # Ruler PLS-DA (same defensive try/except as year regression)
            nan_ruler_metrics = {
                'accuracy_mean': float('nan'), 'accuracy_std': float('nan'),
                'macro_f1_mean': float('nan'), 'macro_f1_std': float('nan'),
                'skipped': True,
            }
            metrics_per_k = {}
            for k in PLS_N_COMPONENTS:
                try:
                    metrics_per_k[str(k)] = fit_plsda_stratified_kfold(X_labeled, pls_info['y_ruler'], k)
                except Exception as e:
                    print(f"    [plsda-skip] k={k} ruler: {type(e).__name__}: {e}", flush=True)
                    metrics_per_k[str(k)] = {**nan_ruler_metrics,
                                              'error': f"{type(e).__name__}: {e}"}
            valid_k = [k for k in PLS_N_COMPONENTS
                       if not (isinstance(metrics_per_k[str(k)].get('macro_f1_mean'), float)
                               and np.isnan(metrics_per_k[str(k)]['macro_f1_mean']))]
            best_k = (max(valid_k, key=lambda k: metrics_per_k[str(k)]['macro_f1_mean'])
                      if valid_k else PLS_N_COMPONENTS[0])
            cfg_key = (f'{args.method}__{args.cleaning}__{args.pooling}'
                       f'__L{layer:02d}__ruler')
            pls_results[cfg_key] = {
                'method':             args.method,
                'cleaning':           args.cleaning,
                'pooling':            args.pooling,
                'layer':              layer,
                'target':             'ruler',
                'n_labeled':          pls_info['n_labeled'],
                'metrics_per_k':      metrics_per_k,
                'best_k_by_macro_f1': best_k,
            }
            best_acc = metrics_per_k[str(best_k)]['accuracy_mean']
            best_f1  = metrics_per_k[str(best_k)]['macro_f1_mean']
            print(f"    PLS-DA ruler  best_k={best_k} acc={best_acc:.3f} macro_f1={best_f1:.3f}")

    if run_cls:
        save_json(cls_results, cls_results_path)
        print(f"\nCLS results → {cls_results_path}  ({len(cls_results)} keys)")
    if run_pls:
        save_json(pls_results, pls_results_path)
        print(f"PLS results → {pls_results_path}  ({len(pls_results)} keys)")

    print(f"Wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == '__main__':
    main()
