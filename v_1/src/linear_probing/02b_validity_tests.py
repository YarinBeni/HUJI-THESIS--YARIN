"""
Step 2b — Validity Tests for Linear Probing Results.
Runs four experiments to validate that the high probe accuracy reflects
genuine representational structure, not artifacts of probe capacity or
architecture.

Experiments:
  A. Learning curve      — Does the probe need many examples, or few suffice?
  B. PCA dimensionality  — Is the signal in a compact subspace?
  C. Linear vs MLP       — Is the encoding truly linear?
  D. Random baseline     — Does pretraining matter vs. random weights?

All experiments reuse existing activations (no GPU needed).
"""

import argparse
import json
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from utils import (
    load_letters, get_splits, load_layer_activations,
    RESULTS_DIR, PERIODS, PERIOD_COLORS,
    TFIDF_BASELINES, SEED,
)


# =============================================================================
# Experiment A — Learning Curve
# =============================================================================

def run_learning_curve(X_tv, y_tv, best_C, fractions, n_repeats=10):
    """Train probes on varying fractions of data to measure sample efficiency.

    Returns dict with 'accuracies' and 'stds' lists (one per fraction).
    """
    n_total = len(y_tv)
    results_acc = []
    results_std = []

    for frac in fractions:
        n_sub = max(int(n_total * frac), 10)  # at least 10 samples
        repeat_accs = []

        for rep in range(n_repeats):
            rng = np.random.RandomState(SEED + rep)
            sub_idx = rng.choice(n_total, size=n_sub, replace=False)
            X_sub = X_tv[sub_idx]
            y_sub = y_tv[sub_idx]

            # Use 3-fold when small, 5-fold otherwise
            n_folds = 3 if n_sub < 100 else 5
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=SEED)

            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=best_C, penalty='l2', max_iter=1000,
                                   random_state=SEED, solver='lbfgs'),
            )
            scores = cross_val_score(pipe, X_sub, y_sub, cv=skf,
                                     scoring='accuracy', n_jobs=-1)
            repeat_accs.append(scores.mean())

        results_acc.append(float(np.mean(repeat_accs)))
        results_std.append(float(np.std(repeat_accs)))

    return {'accuracies': results_acc, 'stds': results_std}


# =============================================================================
# Experiment B — PCA Dimensionality Reduction
# =============================================================================

def run_pca_dimensionality(X_tv, y_tv, best_C, n_components_list):
    """Probe after PCA to k dimensions. Returns accuracies per k."""
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    # Cap at CV training-fold size (n-1)/n of samples, since PCA fits on train fold only
    max_k = min(X_tv.shape[0] * (n_folds - 1) // n_folds, X_tv.shape[1])
    results_acc = []
    results_std = []

    for k in n_components_list:
        k_actual = min(k, max_k)

        pipe = make_pipeline(
            PCA(n_components=k_actual, random_state=SEED),
            StandardScaler(),
            LogisticRegression(C=best_C, penalty='l2', max_iter=1000,
                               random_state=SEED, solver='lbfgs'),
        )
        scores = cross_val_score(pipe, X_tv, y_tv, cv=skf,
                                 scoring='accuracy', n_jobs=-1)
        results_acc.append(float(scores.mean()))
        results_std.append(float(scores.std()))

    return {'accuracies': results_acc, 'stds': results_std}


# =============================================================================
# Experiment C — Linear vs MLP Probe
# =============================================================================

MLP_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1]


def probe_single_layer_mlp(X_tv, y_tv, skf, n_jobs=-1):
    """Train MLP probe at one layer with GridSearchCV over alpha."""
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256,),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
        ),
    )
    grid = GridSearchCV(
        pipe,
        param_grid={'mlpclassifier__alpha': MLP_ALPHA_GRID},
        cv=skf,
        scoring='accuracy',
        refit=False,
        n_jobs=n_jobs,
    )
    grid.fit(X_tv, y_tv)
    mean_accs = grid.cv_results_['mean_test_score']
    std_accs = grid.cv_results_['std_test_score']
    best_idx = np.argmax(mean_accs)
    return {
        'accuracy': float(mean_accs[best_idx]),
        'accuracy_std': float(std_accs[best_idx]),
        'best_alpha': float(MLP_ALPHA_GRID[best_idx]),
    }


def _mlp_one_layer(model_name, cleaning_tag, layer, train_val_idx, y_tv, skf):
    """Run MLP probe for a single layer (used for parallel dispatch)."""
    X = load_layer_activations(model_name, cleaning_tag, layer)
    X_tv = X[train_val_idx]
    # Limit internal GridSearchCV jobs so parallel layers don't oversubscribe
    return layer, probe_single_layer_mlp(X_tv, y_tv, skf, n_jobs=20)


# =============================================================================
# Plotting
# =============================================================================

def plot_learning_curve(lc_results, plots_dir, pooling):
    """Learning curve: accuracy vs. training fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cleaning in zip(axes, ['tier0', 'maximal']):
        data = lc_results[cleaning]
        fracs = lc_results['fractions']
        accs = np.array(data['accuracies'])
        stds = np.array(data['stds'])

        ax.plot(fracs, accs, 'o-', color='#1976D2', linewidth=2, markersize=6)
        ax.fill_between(fracs, accs - stds, accs + stds, alpha=0.2,
                        color='#1976D2')
        ax.axhline(y=data['full_accuracy'], color='red', linestyle='--',
                   alpha=0.6, label=f'Full data ({data["full_accuracy"]:.3f})')

        ax.set_xlabel('Training Fraction', fontsize=12)
        ax.set_ylabel('5-fold CV Accuracy', fontsize=12)
        ax.set_title(f'{cleaning} — Layer {data["layer"]}', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0.25, 1.02)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Learning Curve ({pooling} pooling)', fontsize=14, y=1.02)
    plt.tight_layout()
    ptag = f'_{pooling}' if pooling != 'mean' else ''
    path = plots_dir / f'learning_curve{ptag}.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {path}")


def plot_pca_dimensionality(pca_results, plots_dir, pooling):
    """Accuracy vs. number of PCA components."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cleaning in zip(axes, ['tier0', 'maximal']):
        data = pca_results[cleaning]
        ks = pca_results['n_components']
        accs = np.array(data['accuracies'])
        stds = np.array(data['stds'])

        ax.plot(ks, accs, 'o-', color='#388E3C', linewidth=2, markersize=6)
        ax.fill_between(ks, accs - stds, accs + stds, alpha=0.2,
                        color='#388E3C')
        ax.axhline(y=data['full_accuracy'], color='red', linestyle='--',
                   alpha=0.6, label=f'Full dims ({data["full_accuracy"]:.3f})')

        ax.set_xlabel('Number of PCA Components', fontsize=12)
        ax.set_ylabel('5-fold CV Accuracy', fontsize=12)
        ax.set_title(f'{cleaning} — Layer {data["layer"]}', fontsize=11)
        ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.set_ylim(0.25, 1.02)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'PCA Dimensionality Reduction ({pooling} pooling)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    ptag = f'_{pooling}' if pooling != 'mean' else ''
    path = plots_dir / f'pca_accuracy_vs_dims{ptag}.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {path}")


def plot_mlp_vs_linear(mlp_results, n_layers, plots_dir, pooling):
    """Layer-accuracy curves: linear vs MLP."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    layers = list(range(n_layers))

    for ax, cleaning in zip(axes, ['tier0', 'maximal']):
        lin_accs = [mlp_results[cleaning]['linear'][str(l)]['accuracy']
                    for l in layers]
        mlp_accs = [mlp_results[cleaning]['mlp'][str(l)]['accuracy']
                    for l in layers]

        ax.plot(layers, lin_accs, 'o-', color='#1976D2', linewidth=2,
                markersize=4, label='Linear (LogReg)')
        ax.plot(layers, mlp_accs, 's--', color='#E53935', linewidth=2,
                markersize=4, label='MLP (256, ReLU)')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('5-fold CV Accuracy', fontsize=12)
        ax.set_title(f'{cleaning}', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0.25, 1.02)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Linear vs MLP Probe ({pooling} pooling)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    ptag = f'_{pooling}' if pooling != 'mean' else ''
    path = plots_dir / f'mlp_vs_linear{ptag}.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {path}")


def plot_random_baseline(trained_results, random_results, plots_dir, pooling):
    """Trained vs random-weights layer curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_layers = trained_results['n_layers']
    layers = list(range(n_layers))

    for ax, cleaning in zip(axes, ['tier0', 'maximal']):
        t_accs = [trained_results['layer_results'][cleaning][str(l)]['accuracy']
                  for l in layers]
        r_accs = [random_results['layer_results'][cleaning][str(l)]['accuracy']
                  for l in layers]

        ax.plot(layers, t_accs, 'o-', color='#1976D2', linewidth=2,
                markersize=4, label='Pretrained Qwen')
        ax.plot(layers, r_accs, 's--', color='#9E9E9E', linewidth=2,
                markersize=4, label='Random Weights')

        # TF-IDF baselines
        suffix = 'cleaned' if cleaning == 'maximal' else 'raw'
        baselines = [
            (f'Unigram {suffix}', TFIDF_BASELINES[f'unigram_{suffix}'], ':', 'gray'),
            (f'2-5gram {suffix}', TFIDF_BASELINES[f'2_5gram_{suffix}'], ':', 'darkred'),
        ]
        for label, val, ls, color in baselines:
            ax.axhline(y=val, color=color, linestyle=ls, alpha=0.6, label=label)

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('5-fold CV Accuracy', fontsize=12)
        ax.set_title(f'{cleaning}', fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_ylim(0.25, 1.02)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Pretrained vs Random Weights ({pooling} pooling)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    ptag = f'_{pooling}' if pooling != 'mean' else ''
    path = plots_dir / f'random_baseline_comparison{ptag}.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {path}")


# =============================================================================
# Main
# =============================================================================

def run(args):
    t0 = time.time()

    model_name = args.model
    pooling = args.pooling
    pooling_suffix = f'_{pooling}' if pooling != 'mean' else ''
    cleaning_tags = [f'tier0{pooling_suffix}', f'maximal{pooling_suffix}']
    tag_labels = ['tier0', 'maximal']

    print(f"Model: {model_name}, Pooling: {pooling}")

    # ── Load data and splits ────────────────────────────────────────────────
    df = load_letters()
    le = LabelEncoder()
    le.fit(PERIODS)
    y_all = le.transform(df['period'].values)

    train_idx, val_idx, test_idx = get_splits(df)
    train_val_idx = np.concatenate([train_idx, val_idx])
    y_tv = y_all[train_val_idx]

    print(f"Data: {len(df)} texts, Train+Val: {len(train_val_idx)}")

    # ── Load existing probe results to get best layers and C values ─────────
    probe_results_path = RESULTS_DIR / 'letters__probe_cls__period' / f'probe_results_{model_name}{pooling_suffix}.json'
    with open(probe_results_path) as f:
        probe_results = json.load(f)

    best_layers = probe_results['best_layers']  # {'tier0': 4, 'maximal': 3}
    n_layers = probe_results['n_layers']

    # Get best C for each cleaning at its best layer
    best_Cs = {}
    for cleaning in tag_labels:
        bl = str(best_layers[cleaning])
        best_Cs[cleaning] = probe_results['layer_results'][cleaning][bl]['best_C']

    print(f"Best layers: {best_layers}")
    print(f"Best C values: {best_Cs}")

    # ── Setup output ────────────────────────────────────────────────────────
    plots_dir = RESULTS_DIR / 'letters__probe_cls__period' / 'figures'
    plots_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # =====================================================================
    # Experiment A — Learning Curve
    # =====================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT A — LEARNING CURVE")
    print(f"{'='*70}")

    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    lc_results = {'fractions': fractions}

    for cleaning_tag, cleaning_label in zip(cleaning_tags, tag_labels):
        layer = best_layers[cleaning_label]
        best_C = best_Cs[cleaning_label]
        print(f"\n  [{cleaning_label}] Layer {layer}, C={best_C}")

        X = load_layer_activations(model_name, cleaning_tag, layer)
        X_tv = X[train_val_idx]

        lc = run_learning_curve(X_tv, y_tv, best_C, fractions, n_repeats=10)
        # Store the full-data accuracy from existing results for reference
        bl_str = str(layer)
        full_acc = probe_results['layer_results'][cleaning_label][bl_str]['accuracy']
        lc['layer'] = layer
        lc['best_C'] = best_C
        lc['full_accuracy'] = full_acc
        lc_results[cleaning_label] = lc

        for frac, acc, std in zip(fractions, lc['accuracies'], lc['stds']):
            print(f"    {frac:5.0%} ({int(len(y_tv)*frac):4d} texts): "
                  f"acc={acc:.4f} +/- {std:.4f}")

    plot_learning_curve(lc_results, plots_dir, pooling)

    # =====================================================================
    # Experiment B — PCA Dimensionality Reduction
    # =====================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT B — PCA DIMENSIONALITY REDUCTION")
    print(f"{'='*70}")

    hidden_dim = probe_results['hidden_dim']
    n_components_list = [2, 5, 10, 25, 50, 100, 250, 500, 1000, hidden_dim]
    pca_results = {'n_components': n_components_list}

    for cleaning_tag, cleaning_label in zip(cleaning_tags, tag_labels):
        layer = best_layers[cleaning_label]
        best_C = best_Cs[cleaning_label]
        print(f"\n  [{cleaning_label}] Layer {layer}, C={best_C}")

        X = load_layer_activations(model_name, cleaning_tag, layer)
        X_tv = X[train_val_idx]

        pca_r = run_pca_dimensionality(X_tv, y_tv, best_C, n_components_list)
        bl_str = str(layer)
        pca_r['layer'] = layer
        pca_r['full_accuracy'] = probe_results['layer_results'][cleaning_label][bl_str]['accuracy']
        pca_results[cleaning_label] = pca_r

        for k, acc, std in zip(n_components_list, pca_r['accuracies'],
                               pca_r['stds']):
            print(f"    k={k:5d}: acc={acc:.4f} +/- {std:.4f}")

    plot_pca_dimensionality(pca_results, plots_dir, pooling)

    # =====================================================================
    # Experiment C — Linear vs MLP at All Layers
    # =====================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT C — LINEAR VS MLP PROBE")
    print(f"{'='*70}")

    mlp_results = {}

    for cleaning_tag, cleaning_label in zip(cleaning_tags, tag_labels):
        print(f"\n  [{cleaning_label}]")
        mlp_results[cleaning_label] = {'linear': {}, 'mlp': {}}

        # Reuse existing linear results (identical GridSearchCV already ran)
        for layer in range(n_layers):
            existing = probe_results['layer_results'][cleaning_label][str(layer)]
            mlp_results[cleaning_label]['linear'][str(layer)] = {
                'accuracy': existing['accuracy'],
                'accuracy_std': existing['accuracy_std'],
                'best_C': existing['best_C'],
            }

        # Parallelize MLP across layers (3 concurrent, ~20 internal jobs each)
        print(f"  Running MLP probes across {n_layers} layers (parallel)...")
        layer_mlps = Parallel(n_jobs=3, prefer='processes')(
            delayed(_mlp_one_layer)(
                model_name, cleaning_tag, layer, train_val_idx, y_tv, skf
            )
            for layer in range(n_layers)
        )

        for layer, mlp_res in layer_mlps:
            mlp_results[cleaning_label]['mlp'][str(layer)] = mlp_res
            lin_acc = mlp_results[cleaning_label]['linear'][str(layer)]['accuracy']
            delta = mlp_res['accuracy'] - lin_acc
            print(f"    Layer {layer:2d}: "
                  f"linear={lin_acc:.4f}, "
                  f"MLP={mlp_res['accuracy']:.4f}, "
                  f"delta={delta:+.4f}")

    plot_mlp_vs_linear(mlp_results, n_layers, plots_dir, pooling)

    # =====================================================================
    # Experiment D — Random Baseline Comparison
    # =====================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT D — RANDOM BASELINE COMPARISON")
    print(f"{'='*70}")

    random_model_name = f'{model_name}-random'
    random_results_path = RESULTS_DIR / 'letters__probe_cls__period' / f'probe_results_{random_model_name}{pooling_suffix}.json'

    random_comparison = {
        'trained_results_path': str(probe_results_path),
        'random_results_path': str(random_results_path),
        'comparison_loaded': False,
    }

    if random_results_path.exists():
        with open(random_results_path) as f:
            random_results = json.load(f)
        random_comparison['comparison_loaded'] = True

        # Print comparison at best layers
        for cleaning_label in tag_labels:
            bl = str(best_layers[cleaning_label])
            t_acc = probe_results['layer_results'][cleaning_label][bl]['accuracy']
            r_acc = random_results['layer_results'][cleaning_label][bl]['accuracy']
            print(f"  [{cleaning_label}] Layer {bl}: "
                  f"trained={t_acc:.4f}, random={r_acc:.4f}, "
                  f"delta={t_acc - r_acc:+.4f}")

        plot_random_baseline(probe_results, random_results, plots_dir, pooling)
    else:
        print(f"  WARNING: Random baseline results not found at {random_results_path}")
        print(f"  Run 02_linear_probe.py --model {random_model_name} first.")

    # ── Save all results ────────────────────────────────────────────────────
    output = {
        'model': model_name,
        'pooling': pooling,
        'best_layers': best_layers,
        'learning_curve': lc_results,
        'pca_dimensionality': pca_results,
        'mlp_comparison': mlp_results,
        'random_baseline': random_comparison,
    }

    out_path = RESULTS_DIR / 'letters__probe_cls__period' / f'validity_results_{model_name}{pooling_suffix}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}")

    elapsed = time.time() - t0
    print(f"Total wall time: {elapsed / 60:.1f} min")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 2b: Validity tests for linear probing results')
    parser.add_argument('--model', type=str, required=True,
                        help='Model short name (matching activations directory)')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'last_token'],
                        help='Pooling method (default: mean)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
