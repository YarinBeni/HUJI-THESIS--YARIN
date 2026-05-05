"""
Step 6 — Aggregate PLS results across all methods and layers.

Reads pls_results_{qwen,random,mlm,tfidf}.json from the PLS output directory.
Produces:
  pls_best_layers.json  — best layer per (method, cleaning, pooling, year_transform)
  pls_layer_curves.json — all layer × k metrics per (method, cleaning, pooling, year_transform)

Prints a markdown summary table to stdout.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from utils import RESULTS_DIR

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
                curve_rows.append({
                    'layer': layer,
                    'k': k,
                    'r2_mean': km.get('r2_mean'),
                    'r2_std': km.get('r2_std'),
                    'spearman_mean': km.get('spearman_mean'),
                    'spearman_std': km.get('spearman_std'),
                    'mae_mean': km.get('mae_mean'),
                    'mae_std': km.get('mae_std'),
                    'mase_mean': km.get('mase_mean'),
                    'mase_std': km.get('mase_std'),
                    'mdape_mean': km.get('mdape_mean'),
                    'mdape_std': km.get('mdape_std'),
                    'shuffled_r2_mean': km.get('shuffled_r2_mean'),
                    'shuffled_spearman_mean': km.get('shuffled_spearman_mean'),
                })

        layer_curves[gk] = curve_rows

        valid_rows = [r for r in curve_rows if r['spearman_mean'] is not None]
        if not valid_rows:
            continue

        best_row = max(valid_rows, key=lambda r: r['spearman_mean'])
        sp = best_row['spearman_mean']
        sh = best_row.get('shuffled_spearman_mean')

        best_layers[gk] = {
            'best_layer': best_row['layer'],
            'best_k': best_row['k'],
            'spearman_mean': sp,
            'spearman_std': best_row.get('spearman_std'),
            'r2_mean': best_row.get('r2_mean'),
            'mae_mean': best_row.get('mae_mean'),
            'mase_mean': best_row.get('mase_mean'),
            'mdape_mean': best_row.get('mdape_mean'),
            'shuffled_spearman_mean': sh,
            'delta_vs_shuffled': (sp - sh) if (sp is not None and sh is not None) else None,
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
    headers = ['Group', 'Layer', 'k', 'Spearman', 'R²', 'MAE', 'Shuffled', 'Δ']
    rows = []
    for gk in sorted(best_layers):
        b = best_layers[gk]
        rows.append([
            gk,
            str(b['best_layer']),
            str(b['best_k']),
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
