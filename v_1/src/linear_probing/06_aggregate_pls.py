"""
Step 6 — Aggregate PLS results across all methods and layers.

Reads pls_results_{qwen,random,mlm,tfidf}.json from the PLS output directory.
Produces:
  pls_best_layers.json  — best layer per (method, cleaning, pooling, year_transform)
  pls_layer_curves.json — all layer × k metrics per (method, cleaning, pooling, year_transform)

Prints a markdown summary table to stdout.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / 'results'

PLS_DIR = RESULTS_DIR / 'orcc_round1' / 'pls'
METHODS = ['qwen', 'random', 'mlm', 'tfidf']


def load_all_results() -> dict:
    merged = {}
    for method in METHODS:
        path = PLS_DIR / f'pls_results_{method}.json'
        if not path.exists():
            print(f"  [skip] {path.name} not found", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)
        merged.update(data)
        print(f"  Loaded {len(data)} configs from {path.name}", file=sys.stderr)
    return merged


def group_key(method: str, cleaning: str, pooling: str, year_transform: str) -> str:
    return f"{method}__{cleaning}__{pooling}__year-{year_transform}"


def _best_k_for_group(curve_rows: list[dict]) -> int:
    """Return k that achieves highest mean Spearman across layers."""
    ks = sorted({r['k'] for r in curve_rows})
    k_mean = {}
    for k in ks:
        vals = [r['spearman_mean'] for r in curve_rows if r['k'] == k and r['spearman_mean'] is not None]
        k_mean[k] = sum(vals) / len(vals) if vals else -float('inf')
    return max(k_mean, key=k_mean.get)


def main() -> None:
    all_results = load_all_results()
    if not all_results:
        print("ERROR: No results found. Check that pls_results_*.json files exist in:", file=sys.stderr)
        print(f"  {PLS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Group configs by (method, cleaning, pooling, year_transform)
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for config_key, entry in all_results.items():
        method = entry['method']
        cleaning = entry['cleaning']
        pooling = entry['pooling']
        year_transform = entry['year_transform']
        layer_raw = entry['layer']
        # Tolerate int or 'L15'-style string
        layer = layer_raw if isinstance(layer_raw, int) else int(str(layer_raw).lstrip('L'))
        gk = group_key(method, cleaning, pooling, year_transform)
        groups[gk].append((layer, entry))

    layer_curves: dict[str, list[dict]] = {}
    best_layers: dict[str, dict] = {}

    for gk, layer_entries in groups.items():
        layer_entries.sort(key=lambda x: x[0])

        curve_rows: list[dict] = []
        for layer, entry in layer_entries:
            for k_str, km in entry.get('metrics_per_k', {}).items():
                k = int(k_str)

                # Recompute from fold-level data, skipping degenerate (NaN) folds.
                # A fold is degenerate when y_test is constant (one ruler, one year) →
                # Spearman = NaN. We identify valid folds via spearman_folds and apply
                # the same mask to all other metrics for consistency.
                sp_folds = km.get('spearman_folds', [])
                valid_mask = [
                    i for i, v in enumerate(sp_folds)
                    if v is not None and not math.isnan(v)
                ]
                n_valid = len(valid_mask)

                def _nanmean_folds(key: str):
                    """Mean of stored fold values over valid (non-NaN-Spearman) indices."""
                    folds = km.get(key, [])
                    vals = [folds[i] for i in valid_mask if i < len(folds)]
                    finite = [v for v in vals if v is not None and not math.isnan(v)]
                    return float(np.mean(finite)) if finite else None

                def _nanstd_folds(key: str):
                    folds = km.get(key, [])
                    vals = [folds[i] for i in valid_mask if i < len(folds)]
                    finite = [v for v in vals if v is not None and not math.isnan(v)]
                    return float(np.std(finite)) if len(finite) > 1 else None

                sp_mean  = _nanmean_folds('spearman_folds')
                sp_std   = _nanstd_folds('spearman_folds')

                # Shuffled baseline: fold-level not stored; use scalar from JSON
                # (also NaN for degenerate folds — mark None so it doesn't block selection)
                raw_sh_sp = km.get('shuffled_spearman_mean')
                sh_sp = (None if raw_sh_sp is None or
                         (isinstance(raw_sh_sp, float) and math.isnan(raw_sh_sp))
                         else float(raw_sh_sp))

                curve_rows.append({
                    'layer':                  layer,
                    'k':                      k,
                    'n_valid_folds':          n_valid,
                    'n_total_folds':          len(sp_folds) if sp_folds else None,
                    'r2_mean':                _nanmean_folds('r2_folds'),
                    'r2_std':                 _nanstd_folds('r2_folds'),
                    'spearman_mean':          sp_mean,
                    'spearman_std':           sp_std,
                    'mae_mean':               _nanmean_folds('mae_folds'),
                    'mae_std':                _nanstd_folds('mae_folds'),
                    'mase_mean':              _nanmean_folds('mase_folds'),
                    'mase_std':               _nanstd_folds('mase_folds'),
                    'mdape_mean':             _nanmean_folds('mdape_folds'),
                    'mdape_std':              _nanstd_folds('mdape_folds'),
                    'shuffled_r2_mean':       km.get('shuffled_r2_mean'),
                    'shuffled_spearman_mean': sh_sp,
                })

        layer_curves[gk] = curve_rows

        valid_rows = [r for r in curve_rows if r['spearman_mean'] is not None]
        if not valid_rows:
            continue

        best_row = max(valid_rows, key=lambda r: r['spearman_mean'])
        sp = best_row['spearman_mean']
        sh = best_row.get('shuffled_spearman_mean')

        best_layers[gk] = {
            'best_layer':             best_row['layer'],
            'best_k':                 best_row['k'],
            'n_valid_folds':          best_row.get('n_valid_folds'),
            'n_total_folds':          best_row.get('n_total_folds'),
            'spearman_mean':          sp,
            'spearman_std':           best_row.get('spearman_std'),
            'r2_mean':                best_row.get('r2_mean'),
            'mae_mean':               best_row.get('mae_mean'),
            'mase_mean':              best_row.get('mase_mean'),
            'mdape_mean':             best_row.get('mdape_mean'),
            'shuffled_spearman_mean': sh,
            'delta_vs_shuffled':      (sp - sh) if (sp is not None and sh is not None) else None,
        }

    PLS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PLS_DIR / 'pls_best_layers.json', 'w') as f:
        json.dump(best_layers, f, indent=2)
    print(f"Saved {len(best_layers)} entries → pls_best_layers.json")

    with open(PLS_DIR / 'pls_layer_curves.json', 'w') as f:
        json.dump(layer_curves, f, indent=2)
    print(f"Saved {len(layer_curves)} curves  → pls_layer_curves.json")

    _print_table(best_layers)


def _fmt(val, spec='.4f') -> str:
    if val is None:
        return 'N/A'
    try:
        return format(float(val), spec)
    except (TypeError, ValueError):
        return str(val)


def _print_table(best_layers: dict) -> None:
    headers = ['Group', 'Layer', 'k', 'n_valid', 'Spearman', 'R²', 'MAE', 'Shuffled', 'Δ']
    rows = []
    for gk in sorted(best_layers):
        b = best_layers[gk]
        n_v = b.get('n_valid_folds')
        n_t = b.get('n_total_folds')
        nv_str = f"{n_v}/{n_t}" if n_v is not None and n_t is not None else 'N/A'
        rows.append([
            gk,
            str(b['best_layer']),
            str(b['best_k']),
            nv_str,
            _fmt(b.get('spearman_mean')),
            _fmt(b.get('r2_mean')),
            _fmt(b.get('mae_mean'), '.2f'),
            _fmt(b.get('shuffled_spearman_mean')),
            _fmt(b.get('delta_vs_shuffled')),
        ])

    if not rows:
        print('\n(No results to display)')
        return

    col_widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]

    def fmt_row(cells: list[str]) -> str:
        return '| ' + ' | '.join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + ' |'

    sep = '| ' + ' | '.join('-' * w for w in col_widths) + ' |'

    print('\n## PLS Best Layers\n')
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print()


if __name__ == '__main__':
    main()
