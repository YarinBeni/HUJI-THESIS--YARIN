#!/usr/bin/env python3
"""
Process Archibab CSV file into unified parquet format.

Input: v_1/data/raw/archibab.csv (~1,500 texts)
Output: v_1/data/processed/archibab/archibab_corpus.parquet

Tokenization follows EvaCun 2025 approach (same as eBL processing).
"""

import pandas as pd
import re
from pathlib import Path
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
    clean = re.sub(r'[?*#\[\]<>⸢⸣]', '', value)

    # 2. Remove determinatives (semantic classifiers in curly braces)
    clean = re.sub(r'\{[^}]+\}', '', clean)

    # 3. Normalize subscripts for syllabic signs
    clean = re.sub(r'[₀₁₂₃₄₅₆₇₈₉ₓ]', '', clean)

    # 4. Split on hyphens, dots, and plus signs to get individual signs
    signs = re.split(r'[-\.+]', clean)

    # 5. Clean up: remove empty strings, strip whitespace
    signs = [s.strip() for s in signs if s.strip()]

    # 6. Join with spaces
    return ' '.join(signs)


def determine_certainty(value: str) -> str:
    """
    Determine certainty level based on editorial marks.
    """
    if not value or not isinstance(value, str):
        return "UNKNOWN"

    if 'x' in value.lower() or '...' in value:
        return "MISSING"
    if '[' in value or ']' in value:
        return "MISSING_BUT_COMPLETED"
    if '\u2E22' in value or '\u2E23' in value:  # ⸢ ⸣
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
    """
    if not value or not isinstance(value, str):
        return False

    # Filter language - keep only Akkadian
    if not language or not isinstance(language, str):
        return False
    if language.lower() not in ["akkadian", "akk"]:
        return False

    # Remove fragmentary words
    if 'x' in value.lower() or 'X' in value:
        return False
    if '...' in value:
        return False
    if value.startswith('[') or value.endswith(']'):
        return False
    if value.startswith('⸢') or value.endswith('⸣'):
        return False

    return True


def process_archibab(input_file: str, output_file: str):
    """
    Process Archibab CSV and save to parquet.

    Args:
        input_file: Path to archibab.csv
        output_file: Path to output parquet file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading Archibab data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Total rows: {len(df):,}")

    # Filter rows
    print("Filtering rows...")
    mask = df.apply(lambda row: should_include_word(
        row.get('value', ''),
        row.get('word_language', '')
    ), axis=1)
    df_filtered = df[mask].copy()
    print(f"Rows after filtering: {len(df_filtered):,}")

    # Apply tokenization
    print("Applying tokenization...")
    tqdm.pandas(desc="Tokenizing")
    df_filtered['value_signs'] = df_filtered['value'].progress_apply(tokenize_to_signs)

    # Create standardized schema
    result = pd.DataFrame({
        'source': 'archibab',
        'fragment_id': df_filtered['fragment_id'],
        'line_num': df_filtered['fragment_line_num'],
        'word_idx': df_filtered['index_in_line'],
        'language': df_filtered['word_language'].str.upper(),
        'value_raw': df_filtered['value'],
        'value_signs': df_filtered['value_signs'],
        'value_clean': df_filtered['clean_value'],
        'lemma': df_filtered['lemma'].astype(str),
        'domain': df_filtered['domain'].astype(str),
        'place_discovery': df_filtered['place_discovery'].astype(str),
        'place_composition': df_filtered['place_composition'].astype(str),
        'certainty': df_filtered['value'].apply(determine_certainty)
    })

    # Save as parquet
    print(f"Saving to {output_path}...")
    result.to_parquet(output_path, index=False)

    print(f"\n✓ Processing complete!")
    print(f"  Total words: {len(result):,}")
    print(f"  Unique fragments: {result['fragment_id'].nunique()}")
    print(f"  Languages: {result['language'].value_counts().to_dict()}")
    print(f"  Certainty: {result['certainty'].value_counts().to_dict()}")
    print(f"  Unique signs: {result['value_signs'].str.split().explode().nunique()}")


if __name__ == '__main__':
    import argparse

    # Get script directory to build relative paths
    script_dir = Path(__file__).parent.parent  # Go up to v_1/

    parser = argparse.ArgumentParser(description='Process Archibab CSV to parquet')
    parser.add_argument(
        '--input_file',
        default=str(script_dir / 'data/raw/archibab.csv'),
        help='Path to archibab.csv'
    )
    parser.add_argument(
        '--output_file',
        default=str(script_dir / 'data/processed/archibab/archibab_corpus.parquet'),
        help='Output parquet file'
    )

    args = parser.parse_args()
    process_archibab(args.input_file, args.output_file)
