"""
Step 3 — Interpret Results and Classify Outcome.
Read probe results, classify as Outcome A/B/C per PLAN.md criteria,
extract time direction vector if Outcome A, produce summary.
"""

import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from utils import (
    load_letters, get_splits, load_layer_activations, load_metadata,
    RESULTS_DIR, PERIODS, PERIOD_COLORS, TFIDF_BASELINES, SEED,
)


def run(args):
    model_name = args.model

    # ── Load probe results ──────────────────────────────────────────────────
    results_path = RESULTS_DIR / f'probe_results_{model_name}.json'
    with open(results_path) as f:
        probe = json.load(f)

    print(f"Loaded results from {results_path}")
    print(f"Model: {probe['model']}, {probe['n_layers']} layers\n")

    # ── Extract key metrics ─────────────────────────────────────────────────
    test_tier0 = probe['test_results']['tier0']
    test_maximal = probe['test_results']['maximal']
    perm = probe['permutation_test']

    best_layer_tier0 = test_tier0['best_layer']
    best_layer_maximal = test_maximal['best_layer']

    test_acc_tier0 = test_tier0['test_accuracy']
    test_acc_maximal = test_maximal['test_accuracy']

    p_value = perm['p_value']
    n_layers = probe['n_layers']

    # Layer-accuracy curves
    tier0_accs = [probe['layer_results']['tier0'][str(l)]['accuracy']
                  for l in range(n_layers)]
    maximal_accs = [probe['layer_results']['maximal'][str(l)]['accuracy']
                    for l in range(n_layers)]

    print("KEY METRICS:")
    print(f"  Best layer (tier0):     {best_layer_tier0}")
    print(f"  Test acc (tier0):       {test_acc_tier0:.4f}")
    print(f"  Best layer (maximal):   {best_layer_maximal}")
    print(f"  Test acc (maximal):     {test_acc_maximal:.4f}")
    print(f"  Permutation p-value:    {p_value}")
    print()

    # ── Classify layer position ─────────────────────────────────────────────
    def layer_region(layer, n_layers):
        """Classify layer as early/mid/late."""
        if layer <= 3:
            return 'early'
        elif layer >= n_layers - 4:
            return 'late'
        else:
            return 'mid'

    region_tier0 = layer_region(best_layer_tier0, n_layers)
    region_maximal = layer_region(best_layer_maximal, n_layers)

    # Check if curve is flat (no clear peak)
    tier0_range = max(tier0_accs) - min(tier0_accs)
    is_flat = tier0_range < 0.05

    # ── Classify outcome ────────────────────────────────────────────────────
    cleaned_unigram_floor = TFIDF_BASELINES['unigram_cleaned']  # 69.1%
    cleaned_bigram = TFIDF_BASELINES['bigram_cleaned']          # 91.2%

    # Determine outcome
    if test_acc_maximal < cleaned_unigram_floor:
        outcome = 'C'
        outcome_label = 'FAIL — Below cleaned unigram floor'
        next_steps = [
            'Probe encodes less temporal info than single-character frequencies after cleaning.',
            'Fine-tuning is mandatory for this model on Akkadian.',
            'Write up as a negative result (Gurnee & Tegmark does not extend to OOD languages).',
        ]
    elif (test_acc_maximal >= cleaned_bigram
          and region_maximal in ('mid', 'late')
          and p_value < 0.01
          and not is_flat):
        outcome = 'A'
        outcome_label = 'STRONG — Beats cleaned baselines, peaks at mid/late layers'
        next_steps = [
            'Evidence of temporal representation beyond surface statistics.',
            'Extract time direction vector from probe weights.',
            'Proceed to Track C (SAE decomposition).',
            'Replicate on SEAL literary texts when available.',
        ]
    else:
        outcome = 'B'
        outcome_label = 'PARTIAL — Works but likely surface-level encoding'
        next_steps = [
            'Probe works but may rely on surface features.',
            f'Best layer region: tier0={region_tier0}, maximal={region_maximal}.',
            'Compare raw vs cleaned layer curves to identify which features mattered.',
            'Publishable as: "LLMs encode surface-level temporal signal for OOD languages".',
            'Proceed to fine-tuning for deeper representations.',
        ]

    print(f"OUTCOME: {outcome} — {outcome_label}")
    print(f"  test_acc_maximal ({test_acc_maximal:.3f}) vs cleaned_unigram ({cleaned_unigram_floor})")
    print(f"  test_acc_maximal ({test_acc_maximal:.3f}) vs cleaned_bigram ({cleaned_bigram})")
    print(f"  best layer region (maximal): {region_maximal}")
    print(f"  p-value: {p_value}")
    print(f"  curve range: {tier0_range:.3f} ({'flat' if is_flat else 'peaked'})")
    print()

    # ── Layer shift analysis ────────────────────────────────────────────────
    layer_shift = abs(best_layer_tier0 - best_layer_maximal)
    print(f"LAYER SHIFT: tier0={best_layer_tier0} -> maximal={best_layer_maximal} "
          f"(delta={layer_shift})")
    if layer_shift > 3:
        print("  INTERESTING: Large layer shift suggests different features at different depths.")
    print()

    # ── Outcome A: extract time direction ───────────────────────────────────
    time_direction_path = None
    if outcome == 'A':
        print("Extracting time direction vector...")

        df = load_letters()
        le = LabelEncoder()
        le.fit(PERIODS)
        y_all = le.transform(df['period'].values)

        train_idx, val_idx, test_idx = get_splits(df)
        train_val_idx = np.concatenate([train_idx, val_idx])
        y_tv = y_all[train_val_idx]

        # Retrain probe at best layer on all train+val
        X = load_layer_activations(model_name, 'maximal', best_layer_maximal)
        X_tv = X[train_val_idx]

        clf = LogisticRegression(
            C=test_maximal['best_C'], max_iter=1000, random_state=SEED,
            multi_class='multinomial', solver='lbfgs',
        )
        clf.fit(X_tv, y_tv)

        # For a 3-class problem, the "time direction" is the first principal
        # component of the weight matrix (captures most variance in class separation)
        from sklearn.decomposition import PCA
        W = clf.coef_  # (3, hidden_dim)
        pca_w = PCA(n_components=1, random_state=SEED)
        pca_w.fit(W)
        direction = pca_w.components_[0]  # (hidden_dim,)
        direction = direction / np.linalg.norm(direction)  # unit vector

        time_direction_path = RESULTS_DIR / f'time_direction_{model_name}.npy'
        np.save(time_direction_path, direction)
        print(f"  Saved time direction to {time_direction_path}")
        print(f"  Direction norm: {np.linalg.norm(direction):.4f}")

        # Project all activations onto time direction
        scores = X @ direction  # (4957,)

        # Plot: projected scores by period
        plots_dir = RESULTS_DIR / 'plots'
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: score histograms by period
        ax = axes[0]
        for pi, period in enumerate(PERIODS):
            mask = y_all == pi
            ax.hist(scores[mask], bins=50, alpha=0.5, color=PERIOD_COLORS[period],
                    label=f'{period} (n={mask.sum()})', density=True)
        ax.set_xlabel('Projection onto time direction', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Temporal Score Distribution by Period', fontsize=12)
        ax.legend(fontsize=10)

        # Panel 2: t-SNE of raw activations colored by period
        ax = axes[1]
        tsne = TSNE(n_components=2, perplexity=40, random_state=SEED, max_iter=1000)
        X_2d = tsne.fit_transform(X)
        for pi, period in enumerate(PERIODS):
            mask = y_all == pi
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=PERIOD_COLORS[period], label=f'{period}',
                       alpha=0.35, s=7, linewidths=0, rasterized=True)
        ax.set_title(f't-SNE of Layer {best_layer_maximal} (maximal)', fontsize=12)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.legend(markerscale=4, fontsize=9)

        fig.suptitle('Time Direction Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        path = plots_dir / 'time_direction_analysis.png'
        plt.savefig(path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Saved plot to {path}")

    # ── Cleaning ablation comparison plot ────────────────────────────────────
    plots_dir = RESULTS_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    layers = list(range(n_layers))
    ax.plot(layers, tier0_accs, 'o-', color='#1976D2', linewidth=2, markersize=4,
            label='tier0 (raw)')
    ax.plot(layers, maximal_accs, 's-', color='#388E3C', linewidth=2, markersize=4,
            label='maximal (all 11 filters)')
    ax.axvline(x=best_layer_tier0, color='#1976D2', linestyle=':', alpha=0.5)
    ax.axvline(x=best_layer_maximal, color='#388E3C', linestyle=':', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('5-fold CV Accuracy', fontsize=12)
    ax.set_title('Cleaning Ablation: tier0 vs Maximal', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = plots_dir / 'cleaning_ablation_comparison.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved {path}")

    # ── Save summary JSON ───────────────────────────────────────────────────
    summary = {
        'model': model_name,
        'outcome': outcome,
        'outcome_label': outcome_label,
        'next_steps': next_steps,
        'metrics': {
            'best_layer_tier0': int(best_layer_tier0),
            'best_layer_maximal': int(best_layer_maximal),
            'layer_region_tier0': region_tier0,
            'layer_region_maximal': region_maximal,
            'layer_shift': int(layer_shift),
            'test_accuracy_tier0': float(test_acc_tier0),
            'test_accuracy_maximal': float(test_acc_maximal),
            'test_f1_tier0': float(test_tier0['test_f1_macro']),
            'test_f1_maximal': float(test_maximal['test_f1_macro']),
            'cv_accuracy_tier0': float(test_tier0['cv_accuracy']),
            'cv_accuracy_maximal': float(test_maximal['cv_accuracy']),
            'permutation_p_value': float(p_value),
            'curve_range_tier0': float(tier0_range),
            'curve_is_flat': is_flat,
        },
        'baselines': TFIDF_BASELINES,
        'per_class_tier0': test_tier0.get('per_class', {}),
        'per_class_maximal': test_maximal.get('per_class', {}),
        'confusion_matrix_tier0': test_tier0['confusion_matrix'],
        'confusion_matrix_maximal': test_maximal['confusion_matrix'],
        'time_direction_path': str(time_direction_path) if time_direction_path else None,
    }

    summary_path = RESULTS_DIR / f'summary_{model_name}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Model:              {model_name}")
    print(f"  Outcome:            {outcome} — {outcome_label}")
    print(f"  Test acc (tier0):   {test_acc_tier0:.4f}")
    print(f"  Test acc (maximal): {test_acc_maximal:.4f}")
    print(f"  Best layer (tier0): {best_layer_tier0} ({region_tier0})")
    print(f"  Best layer (max):   {best_layer_maximal} ({region_maximal})")
    print(f"  p-value:            {p_value}")
    print(f"\n  Next steps:")
    for step in next_steps:
        print(f"    - {step}")


def parse_args():
    parser = argparse.ArgumentParser(description='Step 3: Analyze results and classify outcome')
    parser.add_argument('--model', type=str, required=True,
                        help='Model short name (matching probe results file)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
