#!/usr/bin/env python3
"""
Prepare data for Akkadian MLM training.

This script:
1. Loads the unified parquet splits (train/val/test)
2. Builds fragment-level text representations
3. Creates sign vocabulary
4. Saves everything for training

Usage:
    python v_1/src/01_prepare_data.py
"""

import argparse
from pathlib import Path
import json

import pandas as pd

from data_utils import (
    build_fragment_texts,
    build_sign_vocabulary,
    create_eval_subset,
    SPECIAL_TOKEN_IDS,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Akkadian MLM training")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("v_1/data/processed/unified"),
        help="Directory containing train/val/test parquet files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("v_1/data/prepared"),
        help="Output directory for prepared data"
    )
    parser.add_argument(
        "--eval_subset_size",
        type=int,
        default=500,
        help="Number of fragments for fixed eval subset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AKKADIAN MLM DATA PREPARATION")
    print("=" * 60)

    # Step 1: Load all splits
    print("\n📂 Loading parquet splits...")
    train_df = pd.read_parquet(args.data_dir / "train.parquet")
    val_df = pd.read_parquet(args.data_dir / "val.parquet")
    test_df = pd.read_parquet(args.data_dir / "test.parquet")

    print(f"   Train: {len(train_df):,} words")
    print(f"   Val:   {len(val_df):,} words")
    print(f"   Test:  {len(test_df):,} words")

    # Step 2: Build vocabulary from training data only
    print("\n📚 Building sign vocabulary from training data...")
    sign_to_id, id_to_sign = build_sign_vocabulary(
        train_df,
        min_freq=1,
        save_path=str(args.output_dir / "vocab.json")
    )
    print(f"   Vocabulary size: {len(sign_to_id):,}")
    print(f"   Special tokens: {len(SPECIAL_TOKEN_IDS)}")
    print(f"   Regular signs: {len(sign_to_id) - len(SPECIAL_TOKEN_IDS):,}")

    # Step 3: Build fragment texts for each split
    print("\n📝 Building fragment-level texts...")

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"   Processing {split_name}...")
        fragment_df = build_fragment_texts(df, include_metadata=True)
        fragment_df.to_parquet(args.output_dir / f"{split_name}_fragments.parquet")
        print(f"      {len(fragment_df):,} fragments, avg {fragment_df['num_signs'].mean():.1f} signs/fragment")

    # Step 4: Create fixed eval subset for analysis
    print(f"\n🔬 Creating fixed eval subset ({args.eval_subset_size} fragments)...")
    val_fragments = pd.read_parquet(args.output_dir / "val_fragments.parquet")
    eval_subset = create_eval_subset(
        val_fragments,
        n_fragments=args.eval_subset_size,
        seed=args.seed,
        save_path=str(args.output_dir / "eval_subset_ids.json")
    )
    eval_subset.to_parquet(args.output_dir / "eval_subset.parquet")
    print(f"   Saved {len(eval_subset)} fragments for pre/post analysis")

    # Step 5: Save metadata
    print("\n💾 Saving metadata...")
    metadata = {
        "vocab_size": len(sign_to_id),
        "num_special_tokens": len(SPECIAL_TOKEN_IDS),
        "train_fragments": len(pd.read_parquet(args.output_dir / "train_fragments.parquet")),
        "val_fragments": len(pd.read_parquet(args.output_dir / "val_fragments.parquet")),
        "test_fragments": len(pd.read_parquet(args.output_dir / "test_fragments.parquet")),
        "eval_subset_size": len(eval_subset),
        "seed": args.seed,
        "data_dir": str(args.data_dir),
    }
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nFiles created:")
    for f in sorted(args.output_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
