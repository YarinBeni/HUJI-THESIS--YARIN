#!/usr/bin/env python3
"""
Step 1: Text Reconstruction for LLM Baseline Evaluation

Reconstructs fragment-level texts from word-level rows.
Outputs structured parquet and JSONL files for batch API processing.

Usage:
    python 01_prepare_texts.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SOURCE_PARQUET,
    TEXTS_PARQUET,
    TEXTS_JSONL,
    TOKEN_STATS_JSON,
    EVAL_DIR,
    CACHE_DIR,
)


def reconstruct_text(group: pd.DataFrame) -> str:
    """
    Reconstruct full text from word-level rows.

    Steps:
    1. Sort rows by (fragment_line_num, index_in_line)
    2. Join clean_value with spaces within each line
    3. Join lines with newlines

    Args:
        group: DataFrame containing all words for one fragment_id

    Returns:
        Reconstructed text string
    """
    # Sort by line number and position within line
    sorted_group = group.sort_values(['fragment_line_num', 'index_in_line'])

    # Group by line and join words
    lines = []
    for line_num, line_group in sorted_group.groupby('fragment_line_num', sort=True):
        words = line_group['clean_value'].fillna('').tolist()
        line_text = ' '.join(w for w in words if w)
        if line_text:
            lines.append(line_text)

    return '\n'.join(lines)


def extract_metadata(group: pd.DataFrame) -> dict:
    """
    Extract ground truth metadata from a fragment group.
    Takes the first non-null value for each field.

    Args:
        group: DataFrame containing all words for one fragment_id

    Returns:
        Dictionary of metadata fields
    """
    def first_valid(series):
        valid = series.dropna()
        return valid.iloc[0] if len(valid) > 0 else None

    return {
        'temporal_group': first_valid(group['temporal_group']),
        'period': first_valid(group['period']),
        'period_approx': first_valid(group.get('period_approx', pd.Series())),
        'domain_standard': first_valid(group['domain_standard']),
        'domain_finegrained': first_valid(group.get('domain_finegrained', pd.Series())),
        'place_discovery': first_valid(group['place_discovery']),
        'corpus_source': first_valid(group['corpus_source']),
    }


def estimate_tokens(char_count: int) -> int:
    """
    Estimate token count from character count.
    Rule of thumb: ~4 characters per token for transliteration text.
    """
    return max(1, char_count // 4)


def main():
    print("=" * 60)
    print("Step 1: Text Reconstruction for LLM Evaluation")
    print("=" * 60)

    # Ensure output directories exist
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load source data
    print(f"\nLoading source data from {SOURCE_PARQUET}...")
    df = pd.read_parquet(SOURCE_PARQUET)
    print(f"  Loaded {len(df):,} word-level rows")

    # Get unique fragment IDs
    fragment_ids = df['fragment_id'].unique()
    print(f"  Found {len(fragment_ids):,} unique fragments")

    # Process each fragment
    print("\nReconstructing texts...")
    records = []

    for fid in tqdm(fragment_ids, desc="Processing fragments"):
        group = df[df['fragment_id'] == fid]

        # Reconstruct text
        full_text = reconstruct_text(group)

        # Extract metadata
        metadata = extract_metadata(group)

        # Calculate statistics
        word_count = len(group)
        line_count = group['fragment_line_num'].nunique()
        char_count = len(full_text)

        records.append({
            'fragment_id': fid,
            'full_text': full_text,
            'word_count': word_count,
            'line_count': line_count,
            'char_count': char_count,
            **metadata,
        })

    # Create output DataFrame
    result_df = pd.DataFrame(records)

    # Save parquet
    print(f"\nSaving parquet to {TEXTS_PARQUET}...")
    result_df.to_parquet(TEXTS_PARQUET, index=False)
    print(f"  Saved {len(result_df):,} texts")

    # Save JSONL for batch processing
    print(f"\nSaving JSONL to {TEXTS_JSONL}...")
    with open(TEXTS_JSONL, 'w', encoding='utf-8') as f:
        for _, row in result_df.iterrows():
            record = {
                'id': row['fragment_id'],
                'text': row['full_text'],
                'word_count': row['word_count'],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Saved {len(result_df):,} lines")

    # Calculate token statistics
    print("\nCalculating token statistics...")
    total_chars = result_df['char_count'].sum()
    estimated_tokens = result_df['char_count'].apply(estimate_tokens)

    # Estimate prompt template tokens (roughly 500 tokens for the template)
    prompt_template_tokens = 500

    token_stats = {
        'total_texts': len(result_df),
        'total_chars': int(total_chars),
        'estimated_tokens': int(estimated_tokens.sum()),
        'avg_tokens_per_text': float(estimated_tokens.mean()),
        'p50_tokens_per_text': float(estimated_tokens.quantile(0.5)),
        'p95_tokens_per_text': float(estimated_tokens.quantile(0.95)),
        'max_tokens_per_text': int(estimated_tokens.max()),
        'prompt_template_tokens': prompt_template_tokens,
        'estimated_total_input_tokens': int(
            estimated_tokens.sum() + (prompt_template_tokens * len(result_df))
        ),
    }

    print(f"\nSaving token stats to {TOKEN_STATS_JSON}...")
    with open(TOKEN_STATS_JSON, 'w', encoding='utf-8') as f:
        json.dump(token_stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total texts:              {token_stats['total_texts']:,}")
    print(f"  Total characters:         {token_stats['total_chars']:,}")
    print(f"  Estimated tokens (text):  {token_stats['estimated_tokens']:,}")
    print(f"  Avg tokens per text:      {token_stats['avg_tokens_per_text']:.1f}")
    print(f"  P95 tokens per text:      {token_stats['p95_tokens_per_text']:.1f}")
    print(f"  Max tokens per text:      {token_stats['max_tokens_per_text']:,}")
    print(f"  Prompt template tokens:   {token_stats['prompt_template_tokens']}")
    print(f"  Est. total input tokens:  {token_stats['estimated_total_input_tokens']:,}")

    # Distribution by temporal group
    print("\n" + "-" * 40)
    print("Distribution by temporal group:")
    group_counts = result_df['temporal_group'].value_counts()
    for group, count in group_counts.items():
        print(f"  {group}: {count:,}")

    # Distribution by period
    print("\n" + "-" * 40)
    print("Distribution by period:")
    period_counts = result_df['period'].value_counts()
    for period, count in period_counts.items():
        print(f"  {period}: {count:,}")

    # Distribution by corpus source
    print("\n" + "-" * 40)
    print("Distribution by corpus source:")
    source_counts = result_df['corpus_source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count:,}")

    print("\n" + "=" * 60)
    print("Done! Ready for LLM baseline evaluation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
