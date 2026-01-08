#!/usr/bin/env python3
"""
Create unified dataset from all processed sources.

Combines:
- eBL corpus (v_1/data/processed/ebl/ebl_corpus.parquet)
- Archibab corpus (v_1/data/processed/archibab/archibab_corpus.parquet)
- ORACC corpus (v_1/data/processed/oracc/oracc_corpus.parquet)

Output:
- v_1/data/processed/unified/unified_corpus.parquet (all data)
- v_1/data/processed/unified/train.parquet (80%)
- v_1/data/processed/unified/val.parquet (10%)
- v_1/data/processed/unified/test.parquet (10%)

The split is done at the fragment/text level to ensure no data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def load_source(parquet_path: Path, source_name: str) -> pd.DataFrame:
    """Load a processed parquet file."""
    if not parquet_path.exists():
        print(f"Warning: {source_name} not found at {parquet_path}")
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    print(f"  {source_name}: {len(df):,} words from {df['fragment_id'].nunique()} texts")
    return df


def create_train_val_test_split(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """
    Split dataset by fragment_id to avoid data leakage.

    Args:
        df: DataFrame with 'fragment_id' column
        train_ratio: Fraction for training set (default 0.8)
        val_ratio: Fraction for validation set (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(seed)

    # Get unique fragment IDs
    fragment_ids = df['fragment_id'].unique()
    np.random.shuffle(fragment_ids)

    n_fragments = len(fragment_ids)
    n_train = int(n_fragments * train_ratio)
    n_val = int(n_fragments * val_ratio)

    train_ids = set(fragment_ids[:n_train])
    val_ids = set(fragment_ids[n_train:n_train + n_val])
    test_ids = set(fragment_ids[n_train + n_val:])

    train_df = df[df['fragment_id'].isin(train_ids)].copy()
    val_df = df[df['fragment_id'].isin(val_ids)].copy()
    test_df = df[df['fragment_id'].isin(test_ids)].copy()

    return train_df, val_df, test_df


def create_unified_dataset(input_dir: str, output_dir: str):
    """
    Combine all source parquets and create train/val/test splits.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading processed corpora...")

    # Load all sources
    ebl_df = load_source(input_path / 'ebl' / 'ebl_corpus.parquet', 'eBL')
    archibab_df = load_source(input_path / 'archibab' / 'archibab_corpus.parquet', 'Archibab')
    oracc_df = load_source(input_path / 'oracc' / 'oracc_corpus.parquet', 'ORACC')

    # Combine all sources
    print("\nCombining sources...")
    all_dfs = [df for df in [ebl_df, archibab_df, oracc_df] if not df.empty]

    if not all_dfs:
        print("No data to combine!")
        return

    unified_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure consistent data types across sources
    unified_df['line_num'] = pd.to_numeric(unified_df['line_num'], errors='coerce').fillna(0).astype(int)
    unified_df['word_idx'] = pd.to_numeric(unified_df['word_idx'], errors='coerce').fillna(0).astype(int)
    unified_df['fragment_id'] = unified_df['fragment_id'].astype(str)

    print(f"Total unified corpus: {len(unified_df):,} words")
    print(f"Total unique texts: {unified_df['fragment_id'].nunique()}")

    # Show source distribution
    print("\nSource distribution:")
    for source, count in unified_df['source'].value_counts().items():
        print(f"  {source}: {count:,} words ({count/len(unified_df)*100:.1f}%)")

    # Calculate sign statistics
    all_signs = unified_df['value_signs'].str.split().explode()
    sign_counts = Counter(all_signs)
    print(f"\nSign vocabulary size: {len(sign_counts):,}")
    print("Top 10 most common signs:")
    for sign, count in sign_counts.most_common(10):
        print(f"  {sign}: {count:,}")

    # Save unified corpus
    unified_path = output_path / 'unified_corpus.parquet'
    print(f"\nSaving unified corpus to {unified_path}...")
    unified_df.to_parquet(unified_path, index=False)

    # Create train/val/test splits
    print("\nCreating train/val/test splits (80/10/10 by text)...")
    train_df, val_df, test_df = create_train_val_test_split(unified_df)

    # Save splits
    train_path = output_path / 'train.parquet'
    val_path = output_path / 'val.parquet'
    test_path = output_path / 'test.parquet'

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\n✓ Dataset creation complete!")
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_df):,} words from {train_df['fragment_id'].nunique()} texts")
    print(f"  Val:   {len(val_df):,} words from {val_df['fragment_id'].nunique()} texts")
    print(f"  Test:  {len(test_df):,} words from {test_df['fragment_id'].nunique()} texts")

    print(f"\nOutput files:")
    print(f"  {unified_path}")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

    # Create a summary statistics file
    stats = {
        'total_words': len(unified_df),
        'total_texts': unified_df['fragment_id'].nunique(),
        'total_signs': len(all_signs),
        'unique_signs': len(sign_counts),
        'source_counts': unified_df['source'].value_counts().to_dict(),
        'certainty_counts': unified_df['certainty'].value_counts().to_dict(),
        'train_words': len(train_df),
        'train_texts': train_df['fragment_id'].nunique(),
        'val_words': len(val_df),
        'val_texts': val_df['fragment_id'].nunique(),
        'test_words': len(test_df),
        'test_texts': test_df['fragment_id'].nunique(),
    }

    stats_path = output_path / 'dataset_stats.txt'
    with open(stats_path, 'w') as f:
        f.write("Akkadian LLM Dataset Statistics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total words: {stats['total_words']:,}\n")
        f.write(f"Total texts: {stats['total_texts']:,}\n")
        f.write(f"Total signs: {stats['total_signs']:,}\n")
        f.write(f"Unique signs: {stats['unique_signs']:,}\n\n")
        f.write("Source Distribution:\n")
        for source, count in stats['source_counts'].items():
            f.write(f"  {source}: {count:,}\n")
        f.write("\nCertainty Distribution:\n")
        for cert, count in stats['certainty_counts'].items():
            f.write(f"  {cert}: {count:,}\n")
        f.write("\nTrain/Val/Test Split:\n")
        f.write(f"  Train: {stats['train_words']:,} words ({stats['train_texts']} texts)\n")
        f.write(f"  Val: {stats['val_words']:,} words ({stats['val_texts']} texts)\n")
        f.write(f"  Test: {stats['test_words']:,} words ({stats['test_texts']} texts)\n")

    print(f"\nStatistics saved to {stats_path}")


if __name__ == '__main__':
    import argparse

    # Get script directory to build relative paths
    script_dir = Path(__file__).parent.parent  # Go up to v_1/

    parser = argparse.ArgumentParser(description='Create unified dataset from processed sources')
    parser.add_argument(
        '--input_dir',
        default=str(script_dir / 'data/processed'),
        help='Directory containing processed source parquets'
    )
    parser.add_argument(
        '--output_dir',
        default=str(script_dir / 'data/processed/unified'),
        help='Output directory for unified dataset'
    )

    args = parser.parse_args()
    create_unified_dataset(args.input_dir, args.output_dir)
