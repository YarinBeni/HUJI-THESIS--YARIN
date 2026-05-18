"""
Step 7 (CLS) — Plot linear classification layer curves.

Reads cls_layer_curves.json from results/orcc__probe_cls/.
Output: results/orcc__probe_cls/figures/

For each group: one PNG, 1×2 layout (figsize=(12, 5)):
  Left:  Accuracy vs. Layer  — single line, dashed chance_accuracy
  Right: Macro-F1 vs. Layer  — single line, dashed chance_macro_f1

Combined best-of figure per task (ruler, year):
  All methods on one plot, best macro_F1 per layer, color by method.
  Filename: 'best_of_{task}.png'
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
CLS_DIR = _HERE / 'results' / 'orcc__probe_cls'
FIGURES_DIR = CLS_DIR / 'figures'

METHOD_COLORS = {
    'qwen':   '#1976D2',
    'random': '#7B1FA2',
    'mlm':    '#E53935',
    'tfidf':  '#388E3C',
}


def load_curves() -> dict:
    path = CLS_DIR / 'cls_layer_curves.json'
    if not path.exists():
        print(f"ERROR: {path} not found. Run 06_aggregate_cls.py first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def parse_group_key(gk: str) -> tuple[str, str, str, str]:
    """'{method}__{cleaning}__{pooling}__{task}' → (method, cleaning, pooling, task)."""
    parts = gk.split('__')
    return parts[0], parts[1], parts[2], parts[3]


def _safe_vals(rows: list[dict], key: str) -> tuple[list, list]:
    """Return (layers, values) filtering out None entries."""
    pairs = [(r['layer'], r[key]) for r in rows if r.get(key) is not None]
    if not pairs:
        return [], []
    ls, vs = zip(*pairs)
    return list(ls), list(vs)


def plot_group(gk: str, rows: list[dict]) -> None:
    """1×2 PNG: Accuracy | Macro-F1 vs layer for a single group."""
    method, cleaning, pooling, task = parse_group_key(gk)
    out_path = FIGURES_DIR / f'{method}_{cleaning}_{pooling}_{task}.png'

    sorted_rows = sorted(rows, key=lambda r: r['layer'])

    ls_acc, acc_vals   = _safe_vals(sorted_rows, 'accuracy_mean')
    ls_f1,  f1_vals    = _safe_vals(sorted_rows, 'macro_f1_mean')

    # Baselines (constant across layers; take first non-None)
    chance_acc = next((r['chance_accuracy'] for r in sorted_rows if r.get('chance_accuracy') is not None), None)
    chance_f1  = next((r['chance_macro_f1']  for r in sorted_rows if r.get('chance_macro_f1')  is not None), None)

    color = METHOD_COLORS.get(method, '#333333')

    fig, (ax_acc, ax_f1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{method} | {cleaning} | pooling={pooling} | task={task}', fontsize=12)

    # Accuracy subplot
    if ls_acc:
        ax_acc.plot(ls_acc, acc_vals, marker='o', markersize=3, color=color, label='accuracy')
    if chance_acc is not None:
        ax_acc.axhline(chance_acc, linestyle='--', color='gray', alpha=0.7,
                       label=f'chance ({chance_acc:.3f})')
    ax_acc.set_xlabel('Layer')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Accuracy vs Layer')
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    # Macro-F1 subplot
    if ls_f1:
        ax_f1.plot(ls_f1, f1_vals, marker='o', markersize=3, color=color, label='macro-F1')
    if chance_f1 is not None:
        ax_f1.axhline(chance_f1, linestyle='--', color='gray', alpha=0.7,
                      label=f'chance ({chance_f1:.3f})')
    ax_f1.set_xlabel('Layer')
    ax_f1.set_ylabel('Macro-F1')
    ax_f1.set_title('Macro-F1 vs Layer')
    ax_f1.legend(fontsize=8)
    ax_f1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_best_of(all_curves: dict) -> None:
    """
    Combined best-of figure per task: best macro-F1 per layer per method.
    Produces 'best_of_{task}.png' for each unique task.
    """
    parsed = {gk: parse_group_key(gk) for gk in all_curves}
    tasks = sorted({v[3] for v in parsed.values()})
    methods = sorted({v[0] for v in parsed.values()})

    for task in tasks:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle(f'Best Macro-F1 per Layer — task={task}', fontsize=13)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Macro-F1')
        ax.grid(True, alpha=0.3)

        for method in methods:
            # Collect all groups for this method+task (may span cleanings/poolings)
            method_gks = [gk for gk, (m, c, p, t) in parsed.items()
                          if m == method and t == task]
            if not method_gks:
                continue

            # Best macro_f1_mean across all groups (cleaning × pooling) per layer
            best_per_layer: dict[int, float] = {}
            for gk in method_gks:
                for r in all_curves[gk]:
                    mf1 = r.get('macro_f1_mean')
                    if mf1 is None:
                        continue
                    ly = r['layer']
                    if ly not in best_per_layer or mf1 > best_per_layer[ly]:
                        best_per_layer[ly] = mf1

            if not best_per_layer:
                continue

            ls = sorted(best_per_layer)
            f1_vals = [best_per_layer[l] for l in ls]
            ax.plot(ls, f1_vals, marker='o', markersize=3,
                    label=method, color=METHOD_COLORS.get(method))

        ax.legend(fontsize=8)
        plt.tight_layout()
        out_path = FIGURES_DIR / f'best_of_{task}.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved combined: {out_path.name}")


def main() -> None:
    all_curves = load_curves()
    print(f"Loaded {len(all_curves)} groups from cls_layer_curves.json")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for gk, rows in sorted(all_curves.items()):
        plot_group(gk, rows)

    plot_best_of(all_curves)
    print("Done.")


if __name__ == '__main__':
    main()
