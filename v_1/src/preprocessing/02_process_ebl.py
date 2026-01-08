#!/usr/bin/env python3
"""
Process eBL (Electronic Babylonian Library) CSV files into unified parquet format.

Input: v_1/data/raw/extracted/full_corpus_dir/*.csv (~28,000 files)
Output: v_1/data/processed/ebl/ebl_corpus.parquet

Tokenization follows EvaCun 2025 approach:
- Logo-syllabic tokenization with transliterated signs as minimal units
- Split on hyphens and dots to get individual signs
- Remove editorial marks (?, *, <>, [], etc.)
- Remove determinatives ({d}, {m}, etc.)
- Normalize subscripts (ša₂ → ša for syllables)
- Keep logograms whole
"""

import pandas as pd
import re
from pathlib import Path
from glob import glob
from tqdm import tqdm


def tokenize_to_signs(value: str) -> str:
    """
    Apply EvaCun 2025 logo-syllabic tokenization.

    Converts a word like "a-na" into space-separated signs "a na".

    Based on EvaCun 2025 Section 6.2.1:
    - Treats logograms and determinatives as indivisible symbols
    - Treats syllabograms and phonetic complements as divisible
    - Collapses homonymous syllabic signs (ša₂ → ša)
    - Keeps logograms separate (DU vs DU₃)

    Args:
        value: Raw transliteration value (e.g., "a-na", "be-lí-ia")

    Returns:
        Space-separated signs (e.g., "a na", "be lí ia")
    """
    if not value or not isinstance(value, str):
        return ""

    # 1. Remove editorial marks
    # ? = uncertain reading, * = corrected by editor
    # [] = restored/missing, <> = omitted by scribe
    # ⸢⸣ = damaged but readable, # = damaged
    clean = re.sub(r'[?*#\[\]<>⸢⸣]', '', value)

    # 2. Remove determinatives (semantic classifiers in curly braces)
    # e.g., {d}AMAR.UTU → AMAR.UTU (divine name marker removed)
    # e.g., {m}Hammurabi → Hammurabi (personal name marker removed)
    clean = re.sub(r'\{[^}]+\}', '', clean)

    # 3. Normalize subscripts for syllabic signs
    # ša₂ → ša, but keep logograms distinct
    # Subscript digits: ₀₁₂₃₄₅₆₇₈₉ₓ
    # For now, remove all subscripts (EvaCun collapses homonymous syllables)
    clean = re.sub(r'[₀₁₂₃₄₅₆₇₈₉ₓ]', '', clean)

    # 4. Split on hyphens, dots, and plus signs to get individual signs
    # a-na → [a, na]
    # DINGIR.MEŠ → [DINGIR, MEŠ]
    # ša+ša → [ša, ša]
    signs = re.split(r'[-\.+]', clean)

    # 5. Clean up: remove empty strings, strip whitespace
    signs = [s.strip() for s in signs if s.strip()]

    # 6. Join with spaces
    return ' '.join(signs)


def extract_fragment_id_from_filename(filename: str) -> str:
    """
    Extract fragment ID from filename.

    Example: EBL_1848,0720.121.csv → 1848,0720.121
    """
    stem = Path(filename).stem  # Remove .csv
    if stem.startswith('EBL_'):
        return stem[4:]  # Remove 'EBL_' prefix
    return stem


def determine_certainty(value: str) -> str:
    """
    Determine certainty level based on editorial marks.

    Based on Akk/preprocessing/main_preprocess.py
    """
    if not value or not isinstance(value, str):
        return "UNKNOWN"

    # Check for different editorial marks
    if 'x' in value.lower() or '...' in value:
        return "MISSING"
    if '[' in value or ']' in value:
        return "MISSING_BUT_COMPLETED"
    if '\u2E22' in value or '\u2E23' in value:  # ⸢ ⸣ (damaged but readable)
        return "BLURRED"
    if '?' in value:
        return "HAS_DOUBTS"
    if '*' in value:
        return "FIXED_BY_EDITOR"
    if '<' in value or '>' in value:
        return "FORGOTTEN_SIGN"

    return "SURE"


def should_include_word(value: str, language: str) -> bool:
    """
    Filter out fragmentary words and non-Akkadian languages.

    Based on EvaCun 2025 preprocessing.
    """
    if not value or not isinstance(value, str):
        return False

    # Filter language - for now, keep only AKKADIAN
    # TODO: Decide if we want to include SUMERIAN, EMESAL
    if language not in ["AKKADIAN", "akk"]:
        return False

    # Remove fragmentary words (EvaCun approach)
    if 'x' in value.lower() or 'X' in value:
        return False
    if '...' in value:
        return False
    if value.startswith('[') or value.endswith(']'):
        return False
    if value.startswith('⸢') or value.endswith('⸣'):
        return False

    return True


def process_ebl_file(filepath: str) -> pd.DataFrame:
    """
    Process a single eBL CSV file.

    Returns DataFrame with standardized schema.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

    # Extract fragment ID
    fragment_id = extract_fragment_id_from_filename(filepath)

    # Filter rows
    df_filtered = df[df.apply(lambda row: should_include_word(
        row.get('value', ''),
        row.get('word_language', '')
    ), axis=1)].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    # Apply tokenization to get sign-level representation
    df_filtered['value_signs'] = df_filtered['value'].apply(tokenize_to_signs)

    # Create standardized schema
    result = pd.DataFrame({
        'source': 'ebl',
        'fragment_id': fragment_id,
        'line_num': df_filtered['fragment_line_num'],
        'word_idx': df_filtered['index_in_line'],
        'language': df_filtered['word_language'].str.upper(),
        'value_raw': df_filtered['value'],
        'value_signs': df_filtered['value_signs'],  # EvaCun 2025 tokenization
        'value_clean': df_filtered['clean_value'],
        'lemma': df_filtered['lemma'].astype(str),
        'domain': df_filtered['domain'].astype(str),
        'place_discovery': df_filtered.get('place_discovery', '').astype(str),
        'place_composition': df_filtered.get('place_composition', '').astype(str),
        'certainty': df_filtered['value'].apply(determine_certainty)
    })

    return result


def process_all_ebl_files(input_dir: str, output_file: str, limit: int = None):
    """
    Process all eBL CSV files and combine into single parquet.

    Args:
        input_dir: Directory containing EBL_*.csv files
        output_file: Path to output parquet file
        limit: Optional limit on number of files to process (for testing)
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(glob(str(input_path / "*.csv")))

    if limit:
        csv_files = csv_files[:limit]

    print(f"Found {len(csv_files)} eBL CSV files")
    print(f"Processing to: {output_path.absolute()}\n")

    all_fragments = []
    processed = 0
    skipped = 0

    for filepath in tqdm(csv_files, desc="Processing eBL files"):
        fragment_df = process_ebl_file(filepath)

        if not fragment_df.empty:
            all_fragments.append(fragment_df)
            processed += 1
        else:
            skipped += 1

    if not all_fragments:
        print("No data to process!")
        return

    # Combine all fragments
    print("\nCombining all fragments...")
    combined_df = pd.concat(all_fragments, ignore_index=True)

    # Save as parquet
    print(f"Saving to {output_path}...")
    combined_df.to_parquet(output_path, index=False)

    print(f"\n✓ Processing complete!")
    print(f"  Processed: {processed} fragments")
    print(f"  Skipped: {skipped} fragments (no valid data)")
    print(f"  Total words: {len(combined_df):,}")
    print(f"  Unique fragments: {combined_df['fragment_id'].nunique()}")
    print(f"  Languages: {combined_df['language'].value_counts().to_dict()}")
    print(f"  Certainty: {combined_df['certainty'].value_counts().to_dict()}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process eBL CSV files to parquet')
    # Get script directory to build relative paths
    script_dir = Path(__file__).parent.parent  # Go up to v_1/

    parser.add_argument(
        '--input_dir',
        default=str(script_dir / 'data/raw/extracted/full_corpus_dir'),
        help='Directory containing eBL CSV files'
    )
    parser.add_argument(
        '--output_file',
        default=str(script_dir / 'data/processed/ebl/ebl_corpus.parquet'),
        help='Output parquet file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )

    args = parser.parse_args()
    process_all_ebl_files(args.input_dir, args.output_file, args.limit)
