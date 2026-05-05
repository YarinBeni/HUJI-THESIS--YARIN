"""
Step 7 — Plot PLS layer curves (Spearman and R² vs layer).

Reads pls_layer_curves.json from the PLS output directory.
For each (method, cleaning, pooling, year_transform): one PNG with two subplots,
one line per k, dashed horizontal shuffled baseline at best k.

Also produces combined 'best-of' figures (one per year_transform):
best Spearman per layer per method, color by method, panel by cleaning × pooling.

Output: results/orcc_round1/pls/figures/
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from utils import RESULTS_DIR

PLS_DIR = RESULTS_DIR / 'orcc_round1' / 'pls'
FIGURES_DIR = PLS_DIR / 'figures'

K_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 5: '#d62728'}
METHOD_COLORS = {
    'qwen':   '#1976D2',
    'random': '#7B1FA2',
    'mlm':    '#E53935',
    'tfidf':  '#388E3C',
}


def load_curves() -> dict:
    path = PLS_DIR / 'pls_layer_curves.json'
    if not path.exists():
        print(f"ERROR: {path} not found. Run 06_aggregate_pls.py first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def parse_group_key(gk: str) -> tuple:
    """'{method}__{cleaning}__{pooling}__year-{transform}' → (method, cleaning, pooling, year_transform)."""
    parts = gk.split('__')
    return parts[0], parts[1], parts[2], parts[3].replace('year-', '')


def _best_k_by_spearman(rows: list[dict]) -> int:
    """Return k with highest mean Spearman across all layers."""
    ks = sorted({r['k'] for r in rows})
    k_mean: dict[int, float] = {}
    for k in ks:
        vals = [r['spearman_mean'] for r in rows if r['k'] == k and r['spearman_mean'] is not None]
        k_mean[k] = float(np.mean(vals)) if vals else -np.inf
    return max(k_mean, key=k_mean.get)


def plot_group(gk: str, rows: list[dict]) -> None:
    """Two-subplot PNG: Spearman and R² vs layer, one line per k."""
    method, cleaning, pooling, year_transform = parse_group_key(gk)
    out_path = FIGURES_DIR / f'{method}_{cleaning}_{pooling}_{year_transform}.png'

    ks = sorted({r['k'] for r in rows})

    # Shuffled baseline: mean shuffled_spearman_mean across layers for best k
    best_k = _best_k_by_spearman(rows)
    shuffled_vals = [
        r['shuffled_spearman_mean'] for r in rows
        if r['k'] == best_k and r.get('shuffled_spearman_mean') is not None
    ]
    shuffled_baseline = float(np.mean(shuffled_vals)) if shuffled_vals else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{method} | {cleaning} | pooling={pooling} | y={year_transform}', fontsize=12)

    for k in ks:
        k_rows = sorted([r for r in rows if r['k'] == k], key=lambda r: r['layer'])
        ls = [r['layer'] for r in k_rows]
        sp = [r['spearman_mean'] for r in k_rows]
        r2 = [r['r2_mean'] for r in k_rows]
        color = K_COLORS.get(k)
        ax1.plot(ls, sp, marker='o', markersize=3, label=f'k={k}', color=color)
        ax2.plot(ls, r2, marker='o', markersize=3, label=f'k={k}', color=color)

    if shuffled_baseline is not None:
        ax1.axhline(shuffled_baseline, linestyle='--', color='gray', alpha=0.7,
                    label=f'shuffled (k={best_k})')

    for ax, ylabel, title in [
        (ax1, 'Spearman ρ', 'Spearman vs Layer'),
        (ax2, 'R²',         'R² vs Layer'),
    ]:
        ax.set_xlabel('Layer')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_best_of(all_curves: dict) -> None:
    """One figure per year_transform: best Spearman per layer, color by method, panel by cleaning×pooling."""
    parsed = {gk: parse_group_key(gk) for gk in all_curves}
    year_transforms = sorted({v[3] for v in parsed.values()})
    cleanings = sorted({v[1] for v in parsed.values()})
    poolings = sorted({v[2] for v in parsed.values()})
    methods = sorted({v[0] for v in parsed.values()})

    for yt in year_transforms:
        n_rows = max(len(cleanings), 1)
        n_cols = max(len(poolings), 1)
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(6 * n_cols, 4 * n_rows),
                                  squeeze=False)
        fig.suptitle(f'Best Spearman per Layer — year={yt}', fontsize=13)

        for ri, cleaning in enumerate(cleanings):
            for ci, pooling in enumerate(poolings):
                ax = axes[ri][ci]
                ax.set_title(f'{cleaning} | {pooling}', fontsize=9)
                ax.set_xlabel('Layer')
                ax.set_ylabel('Spearman ρ')
                ax.grid(True, alpha=0.3)

                for method in methods:
                    gk = f'{method}__{cleaning}__{pooling}__year-{yt}'
                    if gk not in all_curves:
                        continue
                    rows = all_curves[gk]
                    best_per_layer: dict[int, float] = {}
                    for r in rows:
                        sp = r['spearman_mean']
                        if sp is None:
                            continue
                        ly = r['layer']
                        if ly not in best_per_layer or sp > best_per_layer[ly]:
                            best_per_layer[ly] = sp
                    if not best_per_layer:
                        continue
                    ls = sorted(best_per_layer)
                    sp_vals = [best_per_layer[l] for l in ls]
                    ax.plot(ls, sp_vals, marker='o', markersize=3,
                            label=method, color=METHOD_COLORS.get(method))

                ax.legend(fontsize=7)

        plt.tight_layout()
        out_path = FIGURES_DIR / f'best_of_year-{yt}.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved combined: {out_path.name}")


def main() -> None:
    all_curves = load_curves()
    print(f"Loaded {len(all_curves)} groups from pls_layer_curves.json")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for gk, rows in sorted(all_curves.items()):
        plot_group(gk, rows)

    plot_best_of(all_curves)
    print("Done.")


if __name__ == '__main__':
    main()
