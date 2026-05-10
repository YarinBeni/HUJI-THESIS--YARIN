"""
Step 7 — Plot PLS layer curves (multi-metric vs layer).

Reads pls_layer_curves.json from the PLS output directory.

For year-regression configs: one PNG per group with a 2×3 layout:
  Row 0: Spearman ρ | R² (clipped ≥ -10) | MAE
  Row 1: MASE       | MDAPE               | (hidden)

For ruler (classification) configs: one PNG per group with a 1×2 layout:
  Left: Accuracy vs. Layer  | Right: Macro-F1 vs. Layer

Also produces combined 'best-of' figures (one per year_transform):
  best Spearman + best MAE per layer per method (regression only).

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
PLS_DIR = _HERE / 'results' / 'orcc_round1' / 'pls'
FIGURES_DIR = PLS_DIR / 'figures'

K_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 5: '#d62728'}
METHOD_COLORS = {
    'qwen':   '#1976D2',
    'random': '#7B1FA2',
    'mlm':    '#E53935',
    'tfidf':  '#388E3C',
}

R2_CLIP = -10.0  # clip extreme negative R² for readability


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


def _is_ruler_group(rows: list[dict]) -> bool:
    """Detect ruler (classification) configs by presence of 'accuracy_mean' key."""
    return any('accuracy_mean' in r for r in rows)


def _best_k_by_spearman(rows: list[dict]) -> int:
    """Return k with highest mean Spearman across all layers."""
    ks = sorted({r['k'] for r in rows})
    k_mean: dict[int, float] = {}
    for k in ks:
        vals = [r['spearman_mean'] for r in rows if r['k'] == k and r.get('spearman_mean') is not None]
        k_mean[k] = float(np.mean(vals)) if vals else -np.inf
    return max(k_mean, key=k_mean.get)


def _safe_vals(layer_rows: list[dict], key: str) -> tuple[list, list]:
    """Return (layers, values) filtering out None values."""
    pairs = [(r['layer'], r[key]) for r in layer_rows if r.get(key) is not None]
    if not pairs:
        return [], []
    ls, vs = zip(*pairs)
    return list(ls), list(vs)


def plot_group_ruler(gk: str, rows: list[dict]) -> None:
    """1×2 PNG for ruler (classification) configs: Accuracy | Macro-F1 vs layer."""
    method, cleaning, pooling, year_transform = parse_group_key(gk)
    out_path = FIGURES_DIR / f'{method}_{cleaning}_{pooling}_{year_transform}.png'

    fig, (ax_acc, ax_f1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{method} | {cleaning} | pooling={pooling} | ruler classification', fontsize=12)

    # Sort rows by layer
    sorted_rows = sorted(rows, key=lambda r: r['layer'])

    # Single line (no k sweep for ruler)
    ls_acc, acc_vals = _safe_vals(sorted_rows, 'accuracy_mean')
    ls_f1, f1_vals = _safe_vals(sorted_rows, 'macro_f1_mean')

    # Chance baselines
    chance_acc = next((r['chance_accuracy'] for r in sorted_rows if r.get('chance_accuracy') is not None), None)
    chance_f1 = next((r['chance_macro_f1'] for r in sorted_rows if r.get('chance_macro_f1') is not None), None)
    shuffled_acc = next((r['shuffled_accuracy_mean'] for r in sorted_rows if r.get('shuffled_accuracy_mean') is not None), None)
    shuffled_f1 = next((r['shuffled_macro_f1_mean'] for r in sorted_rows if r.get('shuffled_macro_f1_mean') is not None), None)

    color = METHOD_COLORS.get(method, '#333333')

    if ls_acc:
        ax_acc.plot(ls_acc, acc_vals, marker='o', markersize=3, color=color, label='accuracy')
    if chance_acc is not None:
        ax_acc.axhline(chance_acc, linestyle=':', color='black', alpha=0.6, label=f'chance ({chance_acc:.3f})')
    if shuffled_acc is not None:
        ax_acc.axhline(shuffled_acc, linestyle='--', color='gray', alpha=0.7, label=f'shuffled ({shuffled_acc:.3f})')
    ax_acc.set_xlabel('Layer')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Accuracy vs Layer')
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    if ls_f1:
        ax_f1.plot(ls_f1, f1_vals, marker='o', markersize=3, color=color, label='macro-F1')
    if chance_f1 is not None:
        ax_f1.axhline(chance_f1, linestyle=':', color='black', alpha=0.6, label=f'chance ({chance_f1:.3f})')
    if shuffled_f1 is not None:
        ax_f1.axhline(shuffled_f1, linestyle='--', color='gray', alpha=0.7, label=f'shuffled ({shuffled_f1:.3f})')
    ax_f1.set_xlabel('Layer')
    ax_f1.set_ylabel('Macro-F1')
    ax_f1.set_title('Macro-F1 vs Layer')
    ax_f1.legend(fontsize=8)
    ax_f1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_group_regression(gk: str, rows: list[dict]) -> None:
    """2×3 PNG for year-regression configs: 6 subplots (bottom-right hidden)."""
    method, cleaning, pooling, year_transform = parse_group_key(gk)
    out_path = FIGURES_DIR / f'{method}_{cleaning}_{pooling}_{year_transform}.png'

    ks = sorted({r['k'] for r in rows})

    # Shuffled baseline for spearman and r2
    best_k = _best_k_by_spearman(rows)
    best_k_rows = [r for r in rows if r['k'] == best_k]

    def _shuffled_mean(key: str) -> float | None:
        vals = [r[key] for r in best_k_rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    shuffled_sp = _shuffled_mean('shuffled_spearman_mean')
    shuffled_r2 = _shuffled_mean('shuffled_r2_mean')

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f'{method} | {cleaning} | pooling={pooling} | y={year_transform}', fontsize=12)

    # Subplot definitions: (row, col, data_key, ylabel, title, clip_val, shuffled_val)
    subplot_specs = [
        (0, 0, 'spearman_mean', 'Spearman ρ',             'Spearman vs Layer',   None,    shuffled_sp),
        (0, 1, 'r2_mean',       f'R² (clipped ≥ {R2_CLIP})', 'R² vs Layer',      R2_CLIP, shuffled_r2),
        (0, 2, 'mae_mean',      'MAE',                     'MAE vs Layer',        None,    None),
        (1, 0, 'mase_mean',     'MASE',                    'MASE vs Layer',       None,    None),
        (1, 1, 'mdape_mean',    'MDAPE',                   'MDAPE vs Layer',      None,    None),
    ]

    for ri, ci, data_key, ylabel, title, clip_val, shuffled_val in subplot_specs:
        ax = axes[ri][ci]
        for k in ks:
            k_rows = sorted([r for r in rows if r['k'] == k], key=lambda r: r['layer'])
            pairs = [(r['layer'], r.get(data_key)) for r in k_rows]
            # Filter None, apply clip
            if clip_val is not None:
                pairs = [(l, max(v, clip_val) if v is not None else None) for l, v in pairs]
            valid_pairs = [(l, v) for l, v in pairs if v is not None]
            if valid_pairs:
                ls, vs = zip(*valid_pairs)
                ax.plot(list(ls), list(vs), marker='o', markersize=3,
                        label=f'k={k}', color=K_COLORS.get(k))
        if shuffled_val is not None:
            ax.axhline(shuffled_val, linestyle='--', color='gray', alpha=0.7,
                       label=f'shuffled (k={best_k})')
        ax.set_xlabel('Layer')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide bottom-right axes (position [1][2])
    axes[1][2].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_group(gk: str, rows: list[dict]) -> None:
    """Dispatch to ruler or regression plot based on data keys."""
    if _is_ruler_group(rows):
        plot_group_ruler(gk, rows)
    else:
        plot_group_regression(gk, rows)


def plot_best_of(all_curves: dict) -> None:
    """
    Combined best-of figures (regression year configs only).
    One figure per year_transform for Spearman, one for MAE.
    Color by method, panel by cleaning × pooling.
    """
    parsed = {gk: parse_group_key(gk) for gk in all_curves}

    # Exclude ruler configs from best-of figures
    regression_gks = {gk for gk, rows in all_curves.items() if not _is_ruler_group(rows)}

    year_transforms = sorted({parsed[gk][3] for gk in regression_gks})
    cleanings = sorted({parsed[gk][1] for gk in regression_gks})
    poolings = sorted({parsed[gk][2] for gk in regression_gks})
    methods = sorted({parsed[gk][0] for gk in regression_gks})

    for yt in year_transforms:
        n_rows = max(len(cleanings), 1)
        n_cols = max(len(poolings), 1)

        # --- Best-of Spearman figure ---
        fig_sp, axes_sp = plt.subplots(n_rows, n_cols,
                                        figsize=(6 * n_cols, 4 * n_rows),
                                        squeeze=False)
        fig_sp.suptitle(f'Best Spearman per Layer — year={yt}', fontsize=13)

        # --- Best-of MAE figure ---
        fig_mae, axes_mae = plt.subplots(n_rows, n_cols,
                                          figsize=(6 * n_cols, 4 * n_rows),
                                          squeeze=False)
        fig_mae.suptitle(f'Best MAE per Layer — year={yt}', fontsize=13)

        for ri, cleaning in enumerate(cleanings):
            for ci, pooling in enumerate(poolings):
                ax_sp = axes_sp[ri][ci]
                ax_mae = axes_mae[ri][ci]
                for ax in (ax_sp, ax_mae):
                    ax.set_title(f'{cleaning} | {pooling}', fontsize=9)
                    ax.set_xlabel('Layer')
                    ax.grid(True, alpha=0.3)
                ax_sp.set_ylabel('Spearman ρ')
                ax_mae.set_ylabel('MAE')

                for method in methods:
                    gk = f'{method}__{cleaning}__{pooling}__year-{yt}'
                    if gk not in all_curves:
                        continue
                    rows = all_curves[gk]
                    color = METHOD_COLORS.get(method)

                    # Best Spearman per layer (over all k)
                    best_sp_per_layer: dict[int, float] = {}
                    for r in rows:
                        sp = r.get('spearman_mean')
                        if sp is None:
                            continue
                        ly = r['layer']
                        if ly not in best_sp_per_layer or sp > best_sp_per_layer[ly]:
                            best_sp_per_layer[ly] = sp

                    if best_sp_per_layer:
                        ls = sorted(best_sp_per_layer)
                        ax_sp.plot(ls, [best_sp_per_layer[l] for l in ls],
                                   marker='o', markersize=3, label=method, color=color)

                    # Best MAE per layer (over all k — lowest MAE wins)
                    best_mae_per_layer: dict[int, float] = {}
                    for r in rows:
                        mae = r.get('mae_mean')
                        if mae is None:
                            continue
                        ly = r['layer']
                        if ly not in best_mae_per_layer or mae < best_mae_per_layer[ly]:
                            best_mae_per_layer[ly] = mae

                    if best_mae_per_layer:
                        ls = sorted(best_mae_per_layer)
                        ax_mae.plot(ls, [best_mae_per_layer[l] for l in ls],
                                    marker='o', markersize=3, label=method, color=color)

                ax_sp.legend(fontsize=7)
                ax_mae.legend(fontsize=7)

        fig_sp.tight_layout()
        out_sp = FIGURES_DIR / f'best_of_year-{yt}.png'
        fig_sp.savefig(out_sp, dpi=120, bbox_inches='tight')
        plt.close(fig_sp)
        print(f"  Saved combined: {out_sp.name}")

        fig_mae.tight_layout()
        out_mae = FIGURES_DIR / f'best_of_mae_year-{yt}.png'
        fig_mae.savefig(out_mae, dpi=120, bbox_inches='tight')
        plt.close(fig_mae)
        print(f"  Saved combined: {out_mae.name}")


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
