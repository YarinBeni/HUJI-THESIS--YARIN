"""
Step 6 (CLS) — Aggregate linear classification results across all methods and layers.

Reads cls_results_{qwen,random,mlm,tfidf}.json from results/orcc__probe_cls/.
Produces:
  cls_best_layers.json   — best layer per (method, cleaning, pooling, task)
  cls_layer_curves.json  — all layers × metrics per group

Group key: '{method}__{cleaning}__{pooling}__{task}'
  e.g. 'qwen__tier0__mean__ruler', 'mlm__tier0__mean__year'

Best-layer selection: macro_f1_mean (primary criterion).

Prints a markdown summary table to stdout.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / 'results'

CLS_DIR = RESULTS_DIR / 'orcc__probe_cls'
METHODS = ['qwen', 'random', 'mlm', 'tfidf']


def load_all_results() -> dict:
    merged = {}
    for method in METHODS:
        path = CLS_DIR / f'cls_results_{method}.json'
        if not path.exists():
            print(f"  [skip] {path.name} not found", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)
        merged.update(data)
        print(f"  Loaded {len(data)} configs from {path.name}", file=sys.stderr)
    return merged


def group_key(method: str, cleaning: str, pooling: str, task: str) -> str:
    return f"{method}__{cleaning}__{pooling}__{task}"


def parse_config_key(config_key: str) -> tuple[str, str, str, int, str] | None:
    """
    Parse '{method}__{cleaning}__{pooling}__L{NN}__{task}' into components.
    Returns (method, cleaning, pooling, layer, task) or None on parse failure.
    """
    parts = config_key.split('__')
    if len(parts) != 5:
        return None
    method, cleaning, pooling, layer_str, task = parts
    try:
        layer = int(layer_str.lstrip('L'))
    except ValueError:
        return None
    return method, cleaning, pooling, layer, task


def main() -> None:
    all_results = load_all_results()
    if not all_results:
        print("ERROR: No results found. Check that cls_results_*.json files exist in:", file=sys.stderr)
        print(f"  {CLS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Group configs by (method, cleaning, pooling, task)
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    for config_key, entry in all_results.items():
        parsed = parse_config_key(config_key)
        if parsed is None:
            # Fall back to reading fields from entry dict
            method   = entry.get('method')
            cleaning = entry.get('cleaning')
            pooling  = entry.get('pooling')
            layer    = entry.get('layer')
            task     = entry.get('task')
            if None in (method, cleaning, pooling, layer, task):
                print(f"  [warn] Cannot parse config key: {config_key!r}", file=sys.stderr)
                continue
        else:
            method, cleaning, pooling, layer, task = parsed
            # Prefer entry-level fields if available (more reliable)
            method   = entry.get('method',   method)
            cleaning = entry.get('cleaning', cleaning)
            pooling  = entry.get('pooling',  pooling)
            layer    = entry.get('layer',    layer)
            task     = entry.get('task',     task)

        # Normalize layer to int
        if isinstance(layer, str):
            layer = int(str(layer).lstrip('L'))

        gk = group_key(method, cleaning, pooling, task)
        groups[gk].append((layer, entry))

    layer_curves: dict[str, list[dict]] = {}
    best_layers: dict[str, dict] = {}

    for gk, layer_entries in groups.items():
        layer_entries.sort(key=lambda x: x[0])

        curve_rows: list[dict] = []
        for layer, entry in layer_entries:
            acc_mean  = entry.get('accuracy_mean')
            acc_std   = entry.get('accuracy_std')
            mf1_mean  = entry.get('macro_f1_mean')
            mf1_std   = entry.get('macro_f1_std')
            wf1_mean  = entry.get('weighted_f1_mean')
            chance_acc = entry.get('chance_accuracy')
            chance_f1  = entry.get('chance_macro_f1')

            curve_rows.append({
                'layer':             layer,
                'accuracy_mean':     float(acc_mean)  if acc_mean  is not None else None,
                'accuracy_std':      float(acc_std)   if acc_std   is not None else None,
                'macro_f1_mean':     float(mf1_mean)  if mf1_mean  is not None else None,
                'macro_f1_std':      float(mf1_std)   if mf1_std   is not None else None,
                'weighted_f1_mean':  float(wf1_mean)  if wf1_mean  is not None else None,
                'chance_accuracy':   float(chance_acc) if chance_acc is not None else None,
                'chance_macro_f1':   float(chance_f1)  if chance_f1  is not None else None,
            })

        layer_curves[gk] = curve_rows

        valid_rows = [r for r in curve_rows if r.get('macro_f1_mean') is not None]
        if not valid_rows:
            continue

        best_row = max(valid_rows, key=lambda r: r['macro_f1_mean'])

        # Pull n_classes and n_dropped from entry at the best layer
        best_layer_entry = next(
            (e for l, e in layer_entries if l == best_row['layer']), {}
        )

        best_layers[gk] = {
            'best_layer':            best_row['layer'],
            'best_layer_accuracy':   best_row.get('accuracy_mean'),
            'best_layer_macro_f1':   best_row.get('macro_f1_mean'),
            'best_layer_weighted_f1': best_row.get('weighted_f1_mean'),
            'chance_accuracy':       best_row.get('chance_accuracy'),
            'chance_macro_f1':       best_row.get('chance_macro_f1'),
            'n_classes':             best_layer_entry.get('n_classes'),
            'n_dropped':             best_layer_entry.get('n_dropped'),
        }

    CLS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CLS_DIR / 'cls_best_layers.json', 'w') as f:
        json.dump(best_layers, f, indent=2)
    print(f"Saved {len(best_layers)} entries → cls_best_layers.json")

    with open(CLS_DIR / 'cls_layer_curves.json', 'w') as f:
        json.dump(layer_curves, f, indent=2)
    print(f"Saved {len(layer_curves)} curves  → cls_layer_curves.json")

    _print_table(best_layers)


def _fmt(val, spec='.4f') -> str:
    if val is None:
        return 'N/A'
    try:
        return format(float(val), spec)
    except (TypeError, ValueError):
        return str(val)


def _print_table(best_layers: dict) -> None:
    headers = ['Group', 'Best Layer', 'Accuracy', 'Macro-F1', 'Chance-Acc', 'Chance-F1']
    rows = []
    for gk in sorted(best_layers):
        b = best_layers[gk]
        rows.append([
            gk,
            str(b['best_layer']),
            _fmt(b.get('best_layer_accuracy')),
            _fmt(b.get('best_layer_macro_f1')),
            _fmt(b.get('chance_accuracy')),
            _fmt(b.get('chance_macro_f1')),
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

    print('\n## CLS Best Layers\n')
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print()


if __name__ == '__main__':
    main()
