#!/usr/bin/env python3
"""
Step 3: Results Aggregation

Reads all model prediction caches and combines into a single table
with ground truth labels for evaluation.

Usage:
    python 03_aggregate_results.py
    python 03_aggregate_results.py --models gpt-oss-20b qwen-2.5-72b  # Specific models
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CACHE_DIR,
    TEXTS_PARQUET,
    PREDICTIONS_PARQUET,
    ALL_MODELS,
)


def load_cache(cache_path: Path) -> pd.DataFrame:
    """
    Load predictions from a model cache file.

    Args:
        cache_path: Path to JSONL cache file

    Returns:
        DataFrame with columns: fragment_id, model, prediction fields
    """
    records = []
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                pred = data.get('prediction', {})
                usage = data.get('usage', {})

                record = {
                    'fragment_id': data['fragment_id'],
                    'model': data.get('model', cache_path.stem),
                    'pred_period': pred.get('period', 'Unknown'),
                    'pred_century': pred.get('century_estimate', 'Unknown'),
                    'pred_place': pred.get('place_discovery', 'Unknown'),
                    'pred_catalog_id': pred.get('catalog_id', 'Unknown'),
                    'pred_reasoning': pred.get('reasoning', ''),
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                }
                records.append(record)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: Failed to parse line in {cache_path}: {e}")
                continue

    return pd.DataFrame(records)


def get_available_caches() -> list:
    """Get list of available cache files."""
    if not CACHE_DIR.exists():
        return []
    return list(CACHE_DIR.glob("*.jsonl"))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LLM prediction results"
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=None,
        help="Specific models to aggregate (default: all available)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 3: Results Aggregation")
    print("=" * 60)

    # Find cache files
    cache_files = get_available_caches()
    if not cache_files:
        print("\nNo cache files found in", CACHE_DIR)
        print("Run 02_llm_baseline.py first to generate predictions.")
        sys.exit(1)

    print(f"\nFound {len(cache_files)} cache files:")
    for cf in cache_files:
        print(f"  - {cf.name}")

    # Filter to specified models if provided
    if args.models:
        safe_names = {m.replace('/', '_').replace('.', '_'): m for m in args.models}
        cache_files = [cf for cf in cache_files if cf.stem in safe_names]
        print(f"\nFiltered to {len(cache_files)} specified models")

    if not cache_files:
        print("\nNo matching cache files found.")
        sys.exit(1)

    # Load ground truth
    print(f"\nLoading ground truth from {TEXTS_PARQUET}...")
    ground_truth = pd.read_parquet(TEXTS_PARQUET)
    print(f"  Loaded {len(ground_truth):,} texts")

    # Select ground truth columns
    gt_cols = ['fragment_id', 'temporal_group', 'period',
               'domain_standard', 'domain_finegrained', 'place_discovery', 'corpus_source']
    ground_truth = ground_truth[[c for c in gt_cols if c in ground_truth.columns]]

    # Rename for clarity
    ground_truth = ground_truth.rename(columns={
        'period': 'true_period',
        'temporal_group': 'true_temporal_group',
        'domain_standard': 'true_domain',
        'domain_finegrained': 'true_domain_fine',
        'place_discovery': 'true_place',
    })

    # Load and merge predictions
    print("\nLoading predictions...")
    all_predictions = []

    for cache_path in cache_files:
        print(f"  Loading {cache_path.name}...")
        pred_df = load_cache(cache_path)
        print(f"    {len(pred_df):,} predictions")
        all_predictions.append(pred_df)

    # Combine all predictions
    predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"\nTotal predictions: {len(predictions):,}")

    # Pivot to wide format (one row per fragment, columns for each model)
    print("\nPivoting to wide format...")

    # Get unique models
    models = predictions['model'].unique()
    print(f"  Models: {list(models)}")

    # Start with ground truth
    result = ground_truth.copy()

    # Add predictions for each model
    for model in models:
        model_preds = predictions[predictions['model'] == model].copy()
        model_suffix = f"_{model.replace('-', '_')}"

        # Rename columns
        rename_cols = {
            'pred_period': f'pred_period{model_suffix}',
            'pred_century': f'pred_century{model_suffix}',
            'pred_place': f'pred_place{model_suffix}',
            'pred_catalog_id': f'pred_catalog_id{model_suffix}',
            'pred_reasoning': f'pred_reasoning{model_suffix}',
            'input_tokens': f'input_tokens{model_suffix}',
            'output_tokens': f'output_tokens{model_suffix}',
        }
        model_preds = model_preds.rename(columns=rename_cols)

        # Select columns to merge
        merge_cols = ['fragment_id'] + list(rename_cols.values())
        model_preds = model_preds[[c for c in merge_cols if c in model_preds.columns]]

        # Remove duplicates (keep last)
        model_preds = model_preds.drop_duplicates(subset=['fragment_id'], keep='last')

        # Merge
        result = result.merge(model_preds, on='fragment_id', how='left')
        print(f"    Merged {model}: {len(model_preds):,} predictions")

    # Save results
    print(f"\nSaving to {PREDICTIONS_PARQUET}...")
    result.to_parquet(PREDICTIONS_PARQUET, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total texts:    {len(result):,}")
    print(f"  Models:         {len(models)}")
    print(f"  Output file:    {PREDICTIONS_PARQUET}")

    # Coverage by model
    print("\nCoverage by model:")
    for model in models:
        col = f"pred_period_{model.replace('-', '_')}"
        if col in result.columns:
            coverage = result[col].notna().sum()
            print(f"  {model}: {coverage:,} / {len(result):,} ({100*coverage/len(result):.1f}%)")

    print("\n" + "=" * 60)
    print("Done! Ready for evaluation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
