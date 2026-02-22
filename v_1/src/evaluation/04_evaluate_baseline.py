#!/usr/bin/env python3
"""
Step 4: Evaluation Metrics

Computes evaluation metrics for LLM baseline predictions:
- Per-task accuracy (Period, Domain, Place)
- F1-score (macro, weighted)
- Confusion matrices
- Per-group performance breakdown
- Cross-model comparison

Usage:
    python 04_evaluate_baseline.py
    python 04_evaluate_baseline.py --models gpt-oss-20b qwen-2.5-72b
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PREDICTIONS_PARQUET,
    RESULTS_REPORT_MD,
    METRICS_JSON,
    VALID_PERIODS,
)


def normalize_period(value: str) -> str:
    """Normalize period values to match ground truth labels."""
    if pd.isna(value):
        return 'Unknown'

    value = str(value).strip()

    # Direct matches
    if value in VALID_PERIODS:
        return value

    # Common variations
    value_lower = value.lower()
    if 'old babylonian' in value_lower or 'ob' in value_lower:
        return 'Old Babylonian'
    if 'neo-assyrian' in value_lower or 'neo assyrian' in value_lower or 'na' in value_lower:
        return 'Neo-Assyrian'
    if 'late babylonian' in value_lower or 'lb' in value_lower:
        return 'Late Babylonian'

    return 'Unknown'


def normalize_domain(value: str) -> str:
    """Normalize domain values."""
    if pd.isna(value):
        return 'Unknown'

    value = str(value).strip().lower()

    if 'admin' in value:
        return 'Administrative Letter'
    if 'political' in value:
        return 'Political Letter'
    if 'private' in value:
        return 'Private Letter'
    if 'diplomatic' in value:
        return 'Diplomatic Letter'

    return 'Unknown'


def calculate_metrics(y_true: list, y_pred: list, labels: list = None) -> dict:
    """
    Calculate classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of valid labels (for consistent ordering)

    Returns:
        Dictionary of metrics
    """
    # Filter out unknown/error predictions for core metrics
    valid_mask = [
        t != 'Unknown' and p not in ['Unknown', 'Parse Error', 'API Error']
        for t, p in zip(y_true, y_pred)
    ]
    y_true_valid = [t for t, m in zip(y_true, valid_mask) if m]
    y_pred_valid = [p for p, m in zip(y_pred, valid_mask) if m]

    if len(y_true_valid) == 0:
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'n_valid': 0,
            'n_total': len(y_true),
        }

    return {
        'accuracy': accuracy_score(y_true_valid, y_pred_valid),
        'f1_macro': f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
        'n_valid': len(y_true_valid),
        'n_total': len(y_true),
    }


def get_confusion_matrix_str(y_true: list, y_pred: list, labels: list) -> str:
    """Generate confusion matrix as formatted string."""
    # Filter to valid labels only
    valid_pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t in labels and p in labels
    ]
    if not valid_pairs:
        return "No valid predictions for confusion matrix"

    y_true_f = [t for t, p in valid_pairs]
    y_pred_f = [p for t, p in valid_pairs]

    cm = confusion_matrix(y_true_f, y_pred_f, labels=labels)

    # Format as table
    max_label_len = max(len(l) for l in labels)
    header = " " * (max_label_len + 2) + "  ".join(f"{l[:8]:>8}" for l in labels)
    lines = [header]
    for i, label in enumerate(labels):
        row = f"{label:<{max_label_len}}  " + "  ".join(f"{cm[i,j]:>8}" for j in range(len(labels)))
        lines.append(row)

    return "\n".join(lines)


def find_models(df: pd.DataFrame) -> list:
    """Find model names from column suffixes."""
    models = set()
    for col in df.columns:
        if col.startswith('pred_period_'):
            model = col.replace('pred_period_', '').replace('_', '-')
            models.add(model)
    return sorted(models)


def evaluate_model(df: pd.DataFrame, model: str) -> dict:
    """
    Evaluate a single model's predictions.

    Args:
        df: DataFrame with ground truth and predictions
        model: Model name

    Returns:
        Dictionary of metrics
    """
    suffix = model.replace('-', '_')

    # Get columns
    pred_period_col = f'pred_period_{suffix}'
    pred_domain_col = f'pred_domain_{suffix}'
    pred_place_col = f'pred_place_{suffix}'
    pred_conf_col = f'pred_confidence_{suffix}'

    # Check if model predictions exist
    if pred_period_col not in df.columns:
        return None

    # Filter to rows with predictions
    has_pred = df[pred_period_col].notna()
    df_pred = df[has_pred].copy()

    if len(df_pred) == 0:
        return None

    # Normalize predictions
    df_pred['pred_period_norm'] = df_pred[pred_period_col].apply(normalize_period)
    df_pred['pred_domain_norm'] = df_pred[pred_domain_col].apply(normalize_domain) if pred_domain_col in df_pred.columns else 'Unknown'

    # Period metrics
    period_metrics = calculate_metrics(
        df_pred['true_period'].tolist(),
        df_pred['pred_period_norm'].tolist(),
        labels=VALID_PERIODS
    )

    # Domain metrics (if domain predictions available)
    domain_metrics = {}
    if pred_domain_col in df_pred.columns:
        domain_metrics = calculate_metrics(
            df_pred['true_domain'].fillna('Unknown').tolist(),
            df_pred['pred_domain_norm'].tolist(),
        )

    # Per-group breakdown
    group_metrics = {}
    for group in df_pred['true_temporal_group'].unique():
        group_df = df_pred[df_pred['true_temporal_group'] == group]
        group_metrics[str(group)] = calculate_metrics(
            group_df['true_period'].tolist(),
            group_df['pred_period_norm'].tolist(),
            labels=VALID_PERIODS
        )

    # Per-period breakdown
    period_breakdown = {}
    for period in VALID_PERIODS:
        period_df = df_pred[df_pred['true_period'] == period]
        if len(period_df) > 0:
            correct = (period_df['pred_period_norm'] == period).sum()
            period_breakdown[period] = {
                'n_total': len(period_df),
                'n_correct': int(correct),
                'accuracy': correct / len(period_df),
            }

    # Confidence distribution
    confidence_dist = {}
    if pred_conf_col in df_pred.columns:
        confidence_dist = df_pred[pred_conf_col].value_counts().to_dict()

    # Token usage
    input_tok_col = f'input_tokens_{suffix}'
    output_tok_col = f'output_tokens_{suffix}'
    token_usage = {
        'total_input': int(df_pred[input_tok_col].sum()) if input_tok_col in df_pred.columns else 0,
        'total_output': int(df_pred[output_tok_col].sum()) if output_tok_col in df_pred.columns else 0,
    }
    token_usage['total'] = token_usage['total_input'] + token_usage['total_output']

    return {
        'model': model,
        'n_predictions': len(df_pred),
        'period': period_metrics,
        'domain': domain_metrics,
        'by_group': group_metrics,
        'by_period': period_breakdown,
        'confidence_distribution': confidence_dist,
        'token_usage': token_usage,
    }


def generate_report(results: dict, df: pd.DataFrame) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# LLM Baseline Evaluation Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**Total texts**: {len(df):,}")
    lines.append(f"\n**Models evaluated**: {len(results['models'])}")

    # Summary table
    lines.append("\n## Summary: Period Prediction Accuracy\n")
    lines.append("| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Valid Predictions |")
    lines.append("|-------|----------|------------|---------------|-------------------|")

    for model, metrics in results['models'].items():
        if metrics is None:
            continue
        pm = metrics['period']
        lines.append(
            f"| {model} | {pm['accuracy']:.1%} | {pm['f1_macro']:.3f} | "
            f"{pm['f1_weighted']:.3f} | {pm['n_valid']:,} / {pm['n_total']:,} |"
        )

    # Per-model details
    for model, metrics in results['models'].items():
        if metrics is None:
            continue

        lines.append(f"\n---\n\n## Model: {model}\n")

        # Period metrics
        pm = metrics['period']
        lines.append("### Period Classification\n")
        lines.append(f"- **Accuracy**: {pm['accuracy']:.1%}")
        lines.append(f"- **F1 (Macro)**: {pm['f1_macro']:.3f}")
        lines.append(f"- **F1 (Weighted)**: {pm['f1_weighted']:.3f}")
        lines.append(f"- **Precision (Macro)**: {pm['precision_macro']:.3f}")
        lines.append(f"- **Recall (Macro)**: {pm['recall_macro']:.3f}")

        # Per-period breakdown
        lines.append("\n#### Per-Period Breakdown\n")
        lines.append("| Period | Total | Correct | Accuracy |")
        lines.append("|--------|-------|---------|----------|")
        for period, pdata in metrics.get('by_period', {}).items():
            lines.append(
                f"| {period} | {pdata['n_total']:,} | {pdata['n_correct']:,} | {pdata['accuracy']:.1%} |"
            )

        # Per-group breakdown
        if metrics.get('by_group'):
            lines.append("\n#### Per-Group Breakdown\n")
            lines.append("| Group | Accuracy | F1 (Macro) | N |")
            lines.append("|-------|----------|------------|---|")
            for group, gdata in sorted(metrics['by_group'].items()):
                lines.append(
                    f"| {group} | {gdata['accuracy']:.1%} | {gdata['f1_macro']:.3f} | {gdata['n_valid']:,} |"
                )

        # Domain metrics
        if metrics.get('domain') and metrics['domain'].get('n_valid', 0) > 0:
            dm = metrics['domain']
            lines.append("\n### Domain Classification\n")
            lines.append(f"- **Accuracy**: {dm['accuracy']:.1%}")
            lines.append(f"- **F1 (Macro)**: {dm['f1_macro']:.3f}")

        # Token usage
        tu = metrics.get('token_usage', {})
        if tu.get('total', 0) > 0:
            lines.append("\n### Token Usage\n")
            lines.append(f"- **Total input tokens**: {tu['total_input']:,}")
            lines.append(f"- **Total output tokens**: {tu['total_output']:,}")
            lines.append(f"- **Total tokens**: {tu['total']:,}")

        # Confidence distribution
        conf = metrics.get('confidence_distribution', {})
        if conf:
            lines.append("\n### Confidence Distribution\n")
            for level, count in sorted(conf.items(), key=lambda x: -x[1]):
                lines.append(f"- {level}: {count:,}")

    # Confusion matrices
    lines.append("\n---\n\n## Confusion Matrices\n")

    for model, metrics in results['models'].items():
        if metrics is None:
            continue

        suffix = model.replace('-', '_')
        pred_col = f'pred_period_{suffix}'

        if pred_col in df.columns:
            lines.append(f"\n### {model} - Period\n")
            df_valid = df[df[pred_col].notna()].copy()
            df_valid['pred_norm'] = df_valid[pred_col].apply(normalize_period)

            cm_str = get_confusion_matrix_str(
                df_valid['true_period'].tolist(),
                df_valid['pred_norm'].tolist(),
                labels=VALID_PERIODS
            )
            lines.append(f"```\n{cm_str}\n```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM baseline predictions"
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=None,
        help="Specific models to evaluate (default: all available)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: Baseline Evaluation")
    print("=" * 60)

    # Load predictions
    if not PREDICTIONS_PARQUET.exists():
        print(f"\nError: {PREDICTIONS_PARQUET} not found")
        print("Run 03_aggregate_results.py first.")
        sys.exit(1)

    print(f"\nLoading predictions from {PREDICTIONS_PARQUET}...")
    df = pd.read_parquet(PREDICTIONS_PARQUET)
    print(f"  Loaded {len(df):,} texts")

    # Find models
    available_models = find_models(df)
    print(f"  Available models: {available_models}")

    if args.models:
        models_to_eval = [m for m in args.models if m.replace('-', '_') in [am.replace('-', '_') for am in available_models]]
    else:
        models_to_eval = available_models

    if not models_to_eval:
        print("\nNo models to evaluate!")
        sys.exit(1)

    print(f"\nEvaluating {len(models_to_eval)} models: {models_to_eval}")

    # Evaluate each model
    results = {'models': {}}

    for model in models_to_eval:
        print(f"\n  Evaluating {model}...")
        model_results = evaluate_model(df, model)
        results['models'][model] = model_results

        if model_results:
            print(f"    Period accuracy: {model_results['period']['accuracy']:.1%}")
            print(f"    Valid predictions: {model_results['period']['n_valid']:,}")

    # Add metadata
    results['metadata'] = {
        'total_texts': len(df),
        'evaluation_date': datetime.now().isoformat(),
        'models_evaluated': list(models_to_eval),
    }

    # Save metrics JSON
    print(f"\nSaving metrics to {METRICS_JSON}...")
    with open(METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate and save report
    print(f"Generating report at {RESULTS_REPORT_MD}...")
    report = generate_report(results, df)
    with open(RESULTS_REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(report)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nPeriod Prediction Accuracy:")
    print("-" * 40)
    for model, metrics in results['models'].items():
        if metrics:
            print(f"  {model}: {metrics['period']['accuracy']:.1%} "
                  f"(F1={metrics['period']['f1_macro']:.3f})")

    print(f"\nReports saved to:")
    print(f"  - {RESULTS_REPORT_MD}")
    print(f"  - {METRICS_JSON}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
