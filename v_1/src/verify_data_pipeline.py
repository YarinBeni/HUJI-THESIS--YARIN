#!/usr/bin/env python3
"""
Data Pipeline Verification Script

This script provides full visibility into how the Akkadian data flows from:
  1. Source parquets (eBL, ORACC, Archibab) → one row per WORD
  2. Unified dataset → merged sources, still one row per WORD
  3. Fragment texts → one row per FRAGMENT (text joined from words)
  4. Training dataset → tokenized sequences ready for MLM

Run this script to verify that the data looks exactly as expected at each stage.

Usage:
    python v_1/src/verify_data_pipeline.py
"""

import sys
from pathlib import Path

# Add training module to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "training" / "baseline"))

import pandas as pd
import json


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def verify_source_parquets(data_dir: Path):
    """
    STEP 1: Verify the source-level parquets (one row per word).

    These files are created by:
    - v_1/src/preprocessing/02_process_ebl.py
    - v_1/src/preprocessing/03_process_archibab.py
    - v_1/src/preprocessing/04_process_oracc.py
    """
    print_header("STEP 1: SOURCE PARQUETS (One Row Per Word)")

    sources = {
        'eBL': data_dir / 'ebl' / 'ebl_corpus.parquet',
        'ORACC': data_dir / 'oracc' / 'oracc_corpus.parquet',
        'Archibab': data_dir / 'archibab' / 'archibab_corpus.parquet',
    }

    for source_name, path in sources.items():
        print_subheader(f"{source_name}: {path.name}")

        if not path.exists():
            print(f"  [NOT FOUND] {path}")
            continue

        df = pd.read_parquet(path)
        print(f"  Total rows (words): {len(df):,}")
        print(f"  Total fragments: {df['fragment_id'].nunique():,}")
        print(f"  Columns: {df.columns.tolist()}")

        # Show sample fragment
        sample_frag_id = df['fragment_id'].iloc[0]
        sample_frag = df[df['fragment_id'] == sample_frag_id].head(10)

        print(f"\n  SAMPLE FRAGMENT: '{sample_frag_id}'")
        print(f"  First 10 words (rows) of this fragment:")
        print()

        for idx, row in sample_frag.iterrows():
            print(f"    Row {idx}:")
            print(f"      line_num:    {row.get('line_num', 'N/A')}")
            print(f"      word_idx:    {row.get('word_idx', 'N/A')}")
            print(f"      value_raw:   {row.get('value_raw', 'N/A')}")
            print(f"      value_signs: {row.get('value_signs', 'N/A')}")
            print(f"      certainty:   {row.get('certainty', 'N/A')}")
            print()


def verify_unified_dataset(data_dir: Path):
    """
    STEP 2: Verify the unified dataset (merged sources, still one row per word).

    Created by: v_1/src/preprocessing/05_create_unified.py
    """
    print_header("STEP 2: UNIFIED DATASET (Merged Sources, One Row Per Word)")

    unified_path = data_dir / 'unified' / 'unified_corpus.parquet'

    if not unified_path.exists():
        print(f"[NOT FOUND] {unified_path}")
        return

    df = pd.read_parquet(unified_path)

    print(f"Path: {unified_path}")
    print(f"Total rows (words): {len(df):,}")
    print(f"Total fragments: {df['fragment_id'].nunique():,}")
    print(f"Columns: {df.columns.tolist()}")

    print_subheader("Source Distribution")
    for source, count in df['source'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {source}: {count:,} words ({pct:.1f}%)")

    # Show a sample from each source
    print_subheader("Sample Rows from Each Source")

    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        sample_frag_id = source_df['fragment_id'].iloc[0]
        sample = source_df[source_df['fragment_id'] == sample_frag_id].head(5)

        print(f"\n  [{source.upper()}] Fragment: '{sample_frag_id}'")
        print(f"  Showing first 5 words:")

        for _, row in sample.iterrows():
            print(f"    line={row['line_num']}, word_idx={row['word_idx']}")
            print(f"      value_raw:   '{row['value_raw']}'")
            print(f"      value_signs: '{row['value_signs']}'")
            print()


def verify_train_val_test_splits(data_dir: Path):
    """
    STEP 3: Verify train/val/test splits (still one row per word).

    Created by: v_1/src/preprocessing/05_create_unified.py
    """
    print_header("STEP 3: TRAIN/VAL/TEST SPLITS (One Row Per Word)")

    splits = ['train', 'val', 'test']

    for split in splits:
        path = data_dir / 'unified' / f'{split}.parquet'

        if not path.exists():
            print(f"  [{split.upper()}] NOT FOUND: {path}")
            continue

        df = pd.read_parquet(path)
        print(f"  [{split.upper()}] {len(df):,} words, {df['fragment_id'].nunique():,} fragments")


def verify_fragment_texts(prepared_dir: Path):
    """
    STEP 4: Verify fragment-level texts (one row per fragment).

    Created by: v_1/src/training/baseline/01_prepare_data.py
    Using: build_fragment_texts() from data_utils.py

    THIS IS THE KEY STEP:
    - Input: one row per word with 'value_signs' column
    - Process: group by fragment_id, sort by (line_num, word_idx), join value_signs with spaces
    - Output: one row per fragment with 'text' column containing all signs
    """
    print_header("STEP 4: FRAGMENT TEXTS (One Row Per Fragment)")

    print("""
    HOW THIS TRANSFORMATION WORKS:

    Input (unified parquet - one row per WORD):
    ┌──────────────┬──────────┬──────────┬─────────────┐
    │ fragment_id  │ line_num │ word_idx │ value_signs │
    ├──────────────┼──────────┼──────────┼─────────────┤
    │ BM.123456    │ 1        │ 0        │ a na        │
    │ BM.123456    │ 1        │ 1        │ be li       │
    │ BM.123456    │ 1        │ 2        │ šu          │
    │ BM.123456    │ 2        │ 0        │ ma          │
    └──────────────┴──────────┴──────────┴─────────────┘

    Transformation (data_utils.py:reconstruct_fragment_text):
    1. Group by fragment_id
    2. Sort by (line_num, word_idx)
    3. Join all value_signs with spaces

    Output (fragment parquet - one row per FRAGMENT):
    ┌──────────────┬────────────────────────────┬───────────┐
    │ fragment_id  │ text                       │ num_signs │
    ├──────────────┼────────────────────────────┼───────────┤
    │ BM.123456    │ "a na be li šu ma"         │ 6         │
    └──────────────┴────────────────────────────┴───────────┘

    The 'text' column contains ALL SIGNS from the fragment as a single
    space-separated string. This is what gets tokenized for the model.
    """)

    for split in ['train', 'val', 'test']:
        path = prepared_dir / f'{split}_fragments.parquet'

        if not path.exists():
            print(f"  [{split.upper()}_fragments] NOT FOUND: {path}")
            continue

        df = pd.read_parquet(path)
        print(f"\n  [{split.upper()}_fragments] {len(df):,} fragments")
        print(f"    Columns: {df.columns.tolist()}")
        print(f"    Avg signs per fragment: {df['num_signs'].mean():.1f}")

        # Show concrete examples
        print_subheader(f"Sample Fragments from {split}")

        sample = df.sample(n=min(3, len(df)), random_state=42)

        for _, row in sample.iterrows():
            text = row['text']
            signs = text.split()

            print(f"\n    Fragment ID: '{row['fragment_id']}'")
            print(f"    Source: {row.get('source', 'N/A')}")
            print(f"    Num signs: {row['num_signs']}")
            print(f"    Text (first 100 chars): '{text[:100]}...'")
            print(f"    First 10 signs: {signs[:10]}")
            print(f"    Sign count from split: {len(signs)}")


def verify_training_dataset(prepared_dir: Path):
    """
    STEP 5: Verify the PyTorch training dataset (tokenized sequences).

    Created by: AkkadianMLMDataset in data_utils.py
    Used by: v_1/run_training.py

    THIS SHOWS HOW THE MODEL SEES THE DATA:
    - Input: fragment text (e.g., "a na be li šu ma")
    - Process:
      1. Split on spaces → ['a', 'na', 'be', 'li', 'šu', 'ma']
      2. Convert to IDs using vocab → [45, 892, 234, 567, 123, 88]
      3. Add special tokens → [2, 45, 892, 234, 567, 123, 88, 3, 0, 0, ...]
         (where 2=[CLS], 3=[SEP], 0=[PAD])
      4. Apply 15% MLM masking → some tokens become [MASK] (id=4)
    """
    print_header("STEP 5: TRAINING DATASET (Tokenized for Model)")

    from data_utils import (
        load_vocabulary,
        tokenize_text,
        AkkadianMLMDataset,
        SPECIAL_TOKEN_IDS,
    )

    vocab_path = prepared_dir / 'vocab.json'
    train_fragments_path = prepared_dir / 'train_fragments.parquet'

    if not vocab_path.exists() or not train_fragments_path.exists():
        print(f"  Missing files. Run 01_prepare_data.py first.")
        return

    # Load vocabulary
    sign_to_id, id_to_sign = load_vocabulary(str(vocab_path))
    print(f"  Vocabulary size: {len(sign_to_id):,}")
    print(f"  Special tokens: {SPECIAL_TOKEN_IDS}")

    # Load fragments
    train_df = pd.read_parquet(train_fragments_path)

    print_subheader("Manual Tokenization Example")

    # Pick a sample fragment
    sample_row = train_df.iloc[0]
    text = sample_row['text']
    signs = text.split()

    print(f"\n  Fragment ID: '{sample_row['fragment_id']}'")
    print(f"  Raw text: '{text[:80]}...'")
    print(f"  First 10 signs after split: {signs[:10]}")

    # Tokenize
    input_ids, attention_mask = tokenize_text(text, sign_to_id, max_length=512)

    print(f"\n  After tokenization:")
    print(f"    input_ids length: {len(input_ids)}")
    print(f"    First 15 token IDs: {input_ids[:15]}")

    # Decode back to signs
    decoded = [id_to_sign.get(tid, '[???]') for tid in input_ids[:15]]
    print(f"    Decoded back: {decoded}")

    print(f"\n  Token mapping example:")
    for i, (tid, sign) in enumerate(zip(input_ids[:10], decoded[:10])):
        print(f"    Position {i}: ID={tid:5d} → '{sign}'")

    print_subheader("MLM Dataset Example (What Model Sees)")

    # Create a mini dataset
    mini_df = train_df.head(5)
    dataset = AkkadianMLMDataset(
        mini_df,
        sign_to_id,
        max_length=64,  # Short for demo
        mask_prob=0.15,
        seed=42
    )

    print(f"\n  Dataset size: {len(dataset)}")

    # Get one sample
    sample = dataset[0]

    print(f"\n  Sample batch item:")
    print(f"    input_ids shape: {sample['input_ids'].shape}")
    print(f"    attention_mask shape: {sample['attention_mask'].shape}")
    print(f"    labels shape: {sample['labels'].shape}")

    print(f"\n  First 20 positions:")
    print(f"    {'Pos':<4} {'InputID':<8} {'Label':<8} {'Input':<12} {'Label':<12}")
    print(f"    {'-'*48}")

    for i in range(20):
        input_id = sample['input_ids'][i].item()
        label = sample['labels'][i].item()
        input_sign = id_to_sign.get(input_id, '[???]')
        label_sign = id_to_sign.get(label, '(ignore)') if label != -100 else '(ignore)'

        marker = "← MASKED" if label != -100 else ""
        print(f"    {i:<4} {input_id:<8} {label:<8} {input_sign:<12} {label_sign:<12} {marker}")

    print(f"""
    INTERPRETATION:
    - 'InputID': What the model receives as input
    - 'Label': The correct answer (-100 means "don't compute loss here")
    - When Label != -100, this is a MASKED position
    - The model must predict the original sign from the masked input
    - [CLS]=2, [SEP]=3, [MASK]=4, [PAD]=0
    """)


def trace_single_fragment_full_pipeline(data_dir: Path, prepared_dir: Path):
    """
    BONUS: Trace a single fragment through the entire pipeline.
    """
    print_header("FULL PIPELINE TRACE: Single Fragment")

    from data_utils import load_vocabulary, tokenize_text

    # Load unified data
    unified_path = data_dir / 'unified' / 'train.parquet'
    if not unified_path.exists():
        print("  Unified data not found.")
        return

    df = pd.read_parquet(unified_path)

    # Pick a fragment with ~5-10 words for clarity
    frag_counts = df.groupby('fragment_id').size()
    good_frags = frag_counts[(frag_counts >= 5) & (frag_counts <= 10)]

    if len(good_frags) == 0:
        print("  No suitable fragments found.")
        return

    frag_id = good_frags.index[0]
    frag_df = df[df['fragment_id'] == frag_id].copy()

    print(f"\n  Selected Fragment: '{frag_id}'")
    print(f"  Number of words: {len(frag_df)}")

    print_subheader("Stage 1: Raw Word Rows")
    print(f"\n  From train.parquet (one row per word):")

    frag_df = frag_df.sort_values(['line_num', 'word_idx'])

    print(f"\n  {'Row':<4} {'Line':<5} {'Idx':<4} {'value_raw':<20} {'value_signs':<20}")
    print(f"  {'-'*60}")

    for i, (_, row) in enumerate(frag_df.iterrows()):
        print(f"  {i:<4} {row['line_num']:<5} {row['word_idx']:<4} {str(row['value_raw'])[:18]:<20} {str(row['value_signs'])[:18]:<20}")

    print_subheader("Stage 2: Reconstruct Fragment Text")

    # Simulate what build_fragment_texts does
    signs_list = frag_df['value_signs'].dropna().tolist()
    text = ' '.join(signs_list)
    signs = text.split()

    print(f"\n  Process: Join all value_signs with spaces")
    print(f"  value_signs list: {signs_list}")
    print(f"  Joined text: '{text}'")
    print(f"  Split into signs: {signs}")
    print(f"  Total signs: {len(signs)}")

    print_subheader("Stage 3: Tokenize for Model")

    vocab_path = prepared_dir / 'vocab.json'
    if vocab_path.exists():
        sign_to_id, id_to_sign = load_vocabulary(str(vocab_path))

        input_ids, attention_mask = tokenize_text(text, sign_to_id, max_length=64)

        print(f"\n  Vocabulary lookup:")
        print(f"  {'Sign':<12} {'Token ID':<10}")
        print(f"  {'-'*22}")
        print(f"  {'[CLS]':<12} {sign_to_id['[CLS]']:<10} (special)")

        for sign in signs[:10]:
            tid = sign_to_id.get(sign, sign_to_id['[UNK]'])
            print(f"  {sign:<12} {tid:<10}")

        print(f"  {'[SEP]':<12} {sign_to_id['[SEP]']:<10} (special)")
        print(f"  {'[PAD]':<12} {sign_to_id['[PAD]']:<10} (padding)")

        print(f"\n  Final input_ids (first 20): {input_ids[:20]}")
        print(f"  Attention mask (first 20): {attention_mask[:20]}")
    else:
        print("  Vocab not found. Run 01_prepare_data.py first.")


def main():
    """Run full pipeline verification."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║        AKKADIAN DATA PIPELINE VERIFICATION                       ║
    ║                                                                  ║
    ║  This script traces your data through every transformation:     ║
    ║    Source CSVs → Parquets → Unified → Fragments → Model Input   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Paths
    base_dir = Path(__file__).parent.parent  # v_1/
    data_dir = base_dir / 'data' / 'processed'
    prepared_dir = base_dir / 'data' / 'prepared'

    print(f"Base directory: {base_dir.absolute()}")
    print(f"Processed data: {data_dir.absolute()}")
    print(f"Prepared data: {prepared_dir.absolute()}")

    # Run all verification steps
    verify_source_parquets(data_dir)
    verify_unified_dataset(data_dir)
    verify_train_val_test_splits(data_dir)
    verify_fragment_texts(prepared_dir)
    verify_training_dataset(prepared_dir)
    trace_single_fragment_full_pipeline(data_dir, prepared_dir)

    print_header("SUMMARY")
    print("""
    Data Flow Summary:

    1. SOURCE PARQUETS (02_process_ebl.py, etc.)
       ├── One row per WORD
       ├── Each word has: fragment_id, line_num, word_idx, value_raw, value_signs
       └── value_signs = signs separated by spaces (e.g., "a na" for word "a-na")

    2. UNIFIED DATASET (05_create_unified.py)
       ├── Merges eBL + ORACC + Archibab
       ├── Still one row per WORD
       └── Adds source column

    3. TRAIN/VAL/TEST SPLIT (05_create_unified.py)
       ├── 80/10/10 split by fragment_id
       └── No data leakage between splits

    4. FRAGMENT TEXTS (01_prepare_data.py using data_utils.py)
       ├── Groups rows by fragment_id
       ├── Sorts by (line_num, word_idx)
       ├── Joins all value_signs with spaces
       └── One row per FRAGMENT with 'text' column

    5. TRAINING DATASET (AkkadianMLMDataset)
       ├── Splits text on spaces → list of signs
       ├── Converts signs to token IDs using vocab
       ├── Adds [CLS], [SEP], [PAD] tokens
       ├── Applies 15% MLM masking
       └── Returns input_ids, attention_mask, labels tensors

    KEY INSIGHT:
    The value_signs column already contains pre-tokenized signs.
    Example: word "a-na" → value_signs="a na" → signs=['a', 'na']

    Fragment reconstruction simply concatenates these:
    word1: "a na" + word2: "be li" → fragment text: "a na be li"
    """)


if __name__ == "__main__":
    main()
