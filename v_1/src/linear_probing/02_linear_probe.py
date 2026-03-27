"""
Step 2 — Linear Probe at Every Layer.
Train logistic regression probes at every layer for both cleaning conditions.
Run hyperparameter search, random-label baseline, and final test-set evaluation.
Produce all plots.
"""

import argparse
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from utils import (
    load_letters, get_splits, load_layer_activations, load_metadata,
    activations_dir, RESULTS_DIR, PERIODS, PERIOD_COLORS,
    TFIDF_BASELINES, C_GRID, SEED,
)


def run(args):
    t0 = time.time()

    model_name = args.model
    n_permutations = args.n_permutations
    pooling = args.pooling

    # Determine directory suffix for this pooling method
    suffix = '' if pooling == 'mean' else f'_{pooling}'
    cleaning_tags = [f'tier0{suffix}', f'maximal{suffix}']
    tag_labels = ['tier0', 'maximal']  # short labels for display/keys

    print(f"Pooling method: {pooling}")
    print(f"Reading from directories: {cleaning_tags}")

    # ── Load data and splits ────────────────────────────────────────────────
    df = load_letters()
    le = LabelEncoder()
    le.fit(PERIODS)  # fixed order: OB=0, NA=1, LB=2
    y_all = le.transform(df['period'].values)

    train_idx, val_idx, test_idx = get_splits(df)
    train_val_idx = np.concatenate([train_idx, val_idx])

    print(f"Data: {len(df)} texts")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Train+Val: {len(train_val_idx)}")

    y_tv = y_all[train_val_idx]
    y_test = y_all[test_idx]

    # 5-fold CV on train+val
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ── Load metadata to determine number of layers ─────────────────────────
    meta = load_metadata(model_name, cleaning_tags[0])
    n_layers = meta['n_layers']
    hidden_dim = meta['hidden_dim']
    print(f"Model: {model_name}, {n_layers} layers, hidden_dim={hidden_dim}")

    # ── 2a. Layer-accuracy curve ────────────────────────────────────────────
    results = {}
    for cleaning_tag, cleaning_label in zip(cleaning_tags, tag_labels):
        print(f"\n{'='*70}")
        print(f"PROBING — {cleaning_label} cleaning ({pooling} pooling)")
        print(f"{'='*70}")
        results[cleaning_label] = {}

        for layer in range(n_layers):
            X = load_layer_activations(model_name, cleaning_tag, layer)
            X_tv = X[train_val_idx]

            best_C, best_acc, best_f1, best_acc_std = None, 0, 0, 0
            for C in C_GRID:
                clf = LogisticRegression(
                    C=C, penalty='l2', max_iter=1000,
                    random_state=SEED, solver='lbfgs',
                )
                acc_scores = cross_val_score(clf, X_tv, y_tv, cv=skf, scoring='accuracy')
                f1_scores = cross_val_score(clf, X_tv, y_tv, cv=skf, scoring='f1_macro')
                if acc_scores.mean() > best_acc:
                    best_C = C
                    best_acc = acc_scores.mean()
                    best_acc_std = acc_scores.std()
                    best_f1 = f1_scores.mean()

            results[cleaning][layer] = {
                'accuracy': float(best_acc),
                'accuracy_std': float(best_acc_std),
                'f1_macro': float(best_f1),
                'best_C': float(best_C),
            }
            print(f"  Layer {layer:2d}: acc={best_acc:.4f} +/- {best_acc_std:.4f}, "
                  f"F1={best_f1:.4f}, C={best_C}")

    # ── Find best layers ────────────────────────────────────────────────────
    best_layer_tier0 = max(
        range(n_layers), key=lambda l: results['tier0'][l]['accuracy']
    )
    best_layer_maximal = max(
        range(n_layers), key=lambda l: results['maximal'][l]['accuracy']
    )
    best_C_tier0 = results['tier0'][best_layer_tier0]['best_C']
    best_C_maximal = results['maximal'][best_layer_maximal]['best_C']

    print(f"\nBest layer (tier0):   {best_layer_tier0} "
          f"(acc={results['tier0'][best_layer_tier0]['accuracy']:.4f})")
    print(f"Best layer (maximal): {best_layer_maximal} "
          f"(acc={results['maximal'][best_layer_maximal]['accuracy']:.4f})")

    # ── 2b. Random-label baseline (at best tier0 layer) ─────────────────────
    print(f"\n{'='*70}")
    print(f"RANDOM-LABEL BASELINE ({n_permutations} permutations at layer {best_layer_tier0})")
    print(f"{'='*70}")

    X_best = load_layer_activations(model_name, cleaning_tags[0], best_layer_tier0)
    X_best_tv = X_best[train_val_idx]

    null_accs = []
    for i in range(n_permutations):
        y_shuffled = np.random.RandomState(SEED + i).permutation(y_tv)
        clf = LogisticRegression(
            C=best_C_tier0, max_iter=1000, random_state=SEED,
            solver='lbfgs',
        )
        acc = cross_val_score(clf, X_best_tv, y_shuffled, cv=skf, scoring='accuracy').mean()
        null_accs.append(acc)
        if (i + 1) % 200 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}")

    null_accs = np.array(null_accs)
    real_acc_tier0 = results['tier0'][best_layer_tier0]['accuracy']
    p_value = float((null_accs >= real_acc_tier0).mean())
    print(f"  Null distribution: mean={null_accs.mean():.4f}, "
          f"std={null_accs.std():.4f}, max={null_accs.max():.4f}")
    print(f"  Real accuracy: {real_acc_tier0:.4f}")
    print(f"  p-value: {p_value}")

    # ── 2c. Final test-set evaluation ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL TEST-SET EVALUATION")
    print(f"{'='*70}")

    test_results = {}
    for cleaning_tag, cleaning_label, best_layer, best_C in [
        (cleaning_tags[0], 'tier0', best_layer_tier0, best_C_tier0),
        (cleaning_tags[1], 'maximal', best_layer_maximal, best_C_maximal),
    ]:
        X = load_layer_activations(model_name, cleaning_tag, best_layer)
        X_tv = X[train_val_idx]
        X_te = X[test_idx]

        clf = LogisticRegression(
            C=best_C, max_iter=1000, random_state=SEED,
            solver='lbfgs',
        )
        clf.fit(X_tv, y_tv)
        y_pred = clf.predict(X_te)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        per_class = classification_report(
            y_test, y_pred, target_names=PERIODS, output_dict=True
        )

        test_results[cleaning_label] = {
            'best_layer': int(best_layer),
            'best_C': float(best_C),
            'test_accuracy': float(test_acc),
            'test_f1_macro': float(test_f1),
            'confusion_matrix': cm.tolist(),
            'per_class': {k: v for k, v in per_class.items() if k in PERIODS},
            'cv_accuracy': float(results[cleaning_label][best_layer]['accuracy']),
        }
        print(f"\n  [{cleaning_label}] Layer {best_layer}, C={best_C}")
        print(f"    Test accuracy:  {test_acc:.4f}")
        print(f"    Test F1 macro:  {test_f1:.4f}")
        print(f"    CV accuracy:    {results[cleaning_label][best_layer]['accuracy']:.4f}")
        print(f"    CV-Test gap:    {results[cleaning_label][best_layer]['accuracy'] - test_acc:+.4f}")
        print(f"    Confusion matrix:\n{cm}")

    # ── Save all results to JSON ────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        'model': model_name,
        'pooling': pooling,
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'n_texts': len(df),
        'split_sizes': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx),
        },
        'layer_results': {
            cleaning: {
                str(layer): vals for layer, vals in layer_dict.items()
            }
            for cleaning, layer_dict in results.items()
        },
        'best_layers': {
            'tier0': int(best_layer_tier0),
            'maximal': int(best_layer_maximal),
        },
        'permutation_test': {
            'n_permutations': n_permutations,
            'best_layer': int(best_layer_tier0),
            'real_accuracy': float(real_acc_tier0),
            'null_mean': float(null_accs.mean()),
            'null_std': float(null_accs.std()),
            'null_max': float(null_accs.max()),
            'p_value': p_value,
            'null_distribution': null_accs.tolist(),
        },
        'test_results': test_results,
        'tfidf_baselines': TFIDF_BASELINES,
    }

    pooling_tag = f'_{pooling}' if pooling != 'mean' else ''
    results_path = RESULTS_DIR / f'probe_results_{model_name}{pooling_tag}.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # ── 2d. Plots ───────────────────────────────────────────────────────────
    plots_dir = RESULTS_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Layer-accuracy curve ---
    _plot_layer_curve(results, n_layers, plots_dir, pooling)

    # --- Plot 2: Random-label null distribution ---
    _plot_null_distribution(null_accs, real_acc_tier0, p_value, plots_dir, pooling)

    # --- Plot 3: t-SNE at 3 layers (early / best / late) ---
    _plot_tsne_layers(
        model_name, cleaning_tags, n_layers, best_layer_tier0, best_layer_maximal,
        y_all, plots_dir, pooling,
    )

    # --- Plot 4: Confusion matrix ---
    _plot_confusion_matrix(test_results, plots_dir, pooling)

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed / 60:.1f} min")


# =============================================================================
# Plotting functions
# =============================================================================

def _plot_layer_curve(results, n_layers, plots_dir, pooling='mean'):
    """Layer-accuracy curve for tier0 and maximal, with TF-IDF baselines."""
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = list(range(n_layers))

    for cleaning, color, marker in [('tier0', '#1976D2', 'o'), ('maximal', '#388E3C', 's')]:
        accs = [results[cleaning][l]['accuracy'] for l in layers]
        stds = [results[cleaning][l]['accuracy_std'] for l in layers]
        accs = np.array(accs)
        stds = np.array(stds)
        ax.plot(layers, accs, f'{marker}-', color=color, linewidth=2, markersize=5,
                label=f'{cleaning}', zorder=3)
        ax.fill_between(layers, accs - stds, accs + stds, color=color, alpha=0.15)

    # TF-IDF baselines
    baseline_lines = [
        ('Unigram cleaned (69.1%)', TFIDF_BASELINES['unigram_cleaned'], '--', 'gray'),
        ('Unigram raw (84.8%)', TFIDF_BASELINES['unigram_raw'], '--', 'orange'),
        ('Bigram cleaned (91.2%)', TFIDF_BASELINES['bigram_cleaned'], ':', 'red'),
        ('2-5gram cleaned (96.7%)', TFIDF_BASELINES['2_5gram_cleaned'], ':', 'darkred'),
    ]
    for label, val, ls, color in baseline_lines:
        ax.axhline(y=val, color=color, linestyle=ls, alpha=0.6, label=label)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('5-fold CV Accuracy', fontsize=12)
    ax.set_title(f'Linear Probe Accuracy by Layer ({pooling} pooling)', fontsize=14)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.set_ylim(0.25, 1.02)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    ptag = f'_{pooling}' if pooling != 'mean' else ''
    path = plots_dir / f'layer_accuracy_curve{ptag}.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {path}")


def _plot_null_distribution(null_accs, real_acc, p_value, plots_dir, pooling='mean'):
    """Histogram of null (random-label) accuracies + real accuracy line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_accs, bins=50, color='#90CAF9', edgecolor='white', alpha=0.8,
            label=f'Null distribution (n={len(null_accs)})')
    ax.axvline(x=real_acc, color='red', linewidth=2.5, linestyle='-',
               label=f'Real accuracy ({real_acc:.3f})')
    ax.axvline(x=null_accs.mean(), color='gray', linewidth=1.5, linestyle='--',
               label=f'Null mean ({null_accs.mean():.3f})')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Random-Label Baseline (p = {p_value})', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    ptag = f'_{pooling}' if pooling != 'mean' else ''
    path = plots_dir / f'confound_random_label{ptag}.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {path}")


def _plot_tsne_layers(model_name, cleaning_tags, n_layers, best_layer_tier0, best_layer_maximal,
                      y_all, plots_dir, pooling='mean'):
    """t-SNE at early / best / late layers, 3x2 grid (top=tier0, bottom=maximal)."""
    early_layer = 2
    late_layer = n_layers - 2  # second to last

    layer_picks = {
        cleaning_tags[0]: [early_layer, best_layer_tier0, late_layer],
        cleaning_tags[1]: [early_layer, best_layer_maximal, late_layer],
    }
    tag_labels = ['tier0', 'maximal']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, (cleaning_tag, cleaning_label) in enumerate(zip(cleaning_tags, tag_labels)):
        for col, layer in enumerate(layer_picks[cleaning_tag]):
            ax = axes[row, col]
            X = load_layer_activations(model_name, cleaning_tag, layer)

            # Use a subset for speed if too many
            tsne = TSNE(n_components=2, perplexity=40, random_state=SEED, max_iter=1000)
            X_2d = tsne.fit_transform(X)

            for pi, period in enumerate(PERIODS):
                mask = y_all == pi
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=PERIOD_COLORS[period],
                    label=f'{period} (n={mask.sum()})',
                    alpha=0.35, s=7, linewidths=0, rasterized=True,
                )

            label_map = {
                layer_picks[cleaning_tag][0]: 'Early',
                layer_picks[cleaning_tag][1]: 'Best',
                layer_picks[cleaning_tag][2]: 'Late',
            }
            title_prefix = label_map.get(layer, '')
            ax.set_title(f'{cleaning_label} — Layer {layer} ({title_prefix})', fontsize=11)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            if col == 0:
                ax.set_ylabel(f'{cleaning_label}', fontsize=12, fontweight='bold')
            if row == 0 and col == 2:
                ax.legend(markerscale=4, fontsize=8, loc='best', framealpha=0.8)

    fig.suptitle('t-SNE of Activations at Early / Best / Late Layers', fontsize=14, y=1.01)
    plt.tight_layout()
    ptag = f'_{pooling}' if pooling != 'mean' else ''
    path = plots_dir / f'tsne_by_layer{ptag}.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {path}")


def _plot_confusion_matrix(test_results, plots_dir, pooling='mean'):
    """Confusion matrices for tier0 and maximal on test set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (cleaning, res) in zip(axes, test_results.items()):
        cm = np.array(res['confusion_matrix'])
        disp = ConfusionMatrixDisplay(cm, display_labels=PERIODS)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{cleaning} — Layer {res["best_layer"]}\n'
                     f'Acc={res["test_accuracy"]:.3f}, F1={res["test_f1_macro"]:.3f}',
                     fontsize=11)

    fig.suptitle('Test-Set Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()
    ptag = f'_{pooling}' if pooling != 'mean' else ''
    path = plots_dir / f'confusion_matrix_best_layer{ptag}.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Step 2: Linear probe at every layer')
    parser.add_argument('--model', type=str, required=True,
                        help='Model short name (matching activations directory)')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'last_token'],
                        help='Pooling method (default: mean)')
    parser.add_argument('--n-permutations', type=int, default=1000,
                        help='Number of permutations for random-label test (default: 1000)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
