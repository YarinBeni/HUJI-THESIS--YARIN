#!/usr/bin/env python3
"""
Process ORACC (Open Richly Annotated Cuneiform Corpus) data into unified parquet format.

Uses pre-processed ORACC data from Akk/data/akk_from_jsons.jsonl which contains
transliterated texts from all ORACC projects.

Input: Akk/data/akk_from_jsons.jsonl (23,211 texts)
Output: v_1/data/processed/oracc/oracc_corpus.parquet

Tokenization follows EvaCun 2025 approach (same as eBL/Archibab processing).
"""

import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm


def tokenize_to_signs(value: str) -> str:
    """
    Apply EvaCun 2025 logo-syllabic tokenization.

    Converts a word like "a-na" into space-separated signs "a na".
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


def should_include_word(value: str) -> bool:
    """
    Filter out fragmentary words.
    """
    if not value or not isinstance(value, str):
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


def parse_raw_text_to_words(raw_text: str, text_id: str, project: str, metadata: dict) -> list:
    """
    Parse raw_text field into individual word records.

    The raw_text contains space-separated words with special markers:
    - \\t, \\m, \\d, \\p = formatting markers (ignore)
    - Words are space-separated
    - Each word may have determinatives in {}

    Returns list of word dictionaries.
    """
    if not raw_text:
        return []

    # Split into lines (\\t often indicates line breaks or labels)
    # Remove formatting markers
    clean_text = re.sub(r'\\[tmdp]', ' ', raw_text)

    # Split on whitespace
    tokens = clean_text.split()

    words = []
    line_num = 0
    word_idx = 0

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Skip if it's purely a determinative or marker
        if token.startswith('{') and token.endswith('}'):
            continue

        if should_include_word(token):
            words.append({
                'text_id': text_id,
                'project': project,
                'line_num': line_num,
                'word_idx': word_idx,
                'value_raw': token,
                'genre': metadata.get('genre', ''),
                'provenience': metadata.get('provenience', ''),
                'period': metadata.get('period', ''),
            })
            word_idx += 1

    return words


def process_oracc_jsonl(input_file: str, output_file: str):
    """
    Process ORACC JSONL file and save to parquet.

    Args:
        input_file: Path to akk_from_jsons.jsonl
        output_file: Path to output parquet file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading ORACC data from: {input_path}")

    all_words = []
    text_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing ORACC texts"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            text_id = data.get('id_text', '')
            project = data.get('sub_project_name', data.get('project_name', ''))
            raw_text = data.get('raw_text', '')

            metadata = {
                'genre': data.get('genre', ''),
                'provenience': data.get('provenience', ''),
                'period': data.get('period', ''),
            }

            words = parse_raw_text_to_words(raw_text, text_id, project, metadata)
            all_words.extend(words)
            text_count += 1

    print(f"Total texts: {text_count:,}")
    print(f"Total words extracted: {len(all_words):,}")

    if not all_words:
        print("No words extracted!")
        return

    # Convert to DataFrame
    print("\nCreating DataFrame...")
    df = pd.DataFrame(all_words)

    # Apply tokenization
    print("Applying tokenization...")
    tqdm.pandas(desc="Tokenizing")
    df['value_signs'] = df['value_raw'].progress_apply(tokenize_to_signs)
    df['certainty'] = df['value_raw'].apply(determine_certainty)

    # Filter out empty tokenizations (e.g., words that were just determinatives)
    df = df[df['value_signs'].str.strip() != '']

    # Standardize schema to match eBL/Archibab
    result = pd.DataFrame({
        'source': 'oracc',
        'fragment_id': df['text_id'],
        'line_num': df['line_num'],
        'word_idx': df['word_idx'],
        'language': 'AKKADIAN',  # ORACC data is pre-filtered for Akkadian
        'value_raw': df['value_raw'],
        'value_signs': df['value_signs'],
        'value_clean': df['value_raw'],  # Use raw as clean for ORACC
        'lemma': '',  # ORACC JSONL doesn't include lemma
        'domain': df['genre'].astype(str),
        'place_discovery': df['provenience'].astype(str),
        'place_composition': df['period'].astype(str),
        'certainty': df['certainty']
    })

    # Save as parquet
    print(f"Saving to {output_path}...")
    result.to_parquet(output_path, index=False)

    print(f"\n✓ Processing complete!")
    print(f"  Total words: {len(result):,}")
    print(f"  Unique texts: {result['fragment_id'].nunique()}")
    print(f"  Certainty distribution:")
    for cert, count in result['certainty'].value_counts().items():
        print(f"    {cert}: {count:,}")
    print(f"  Unique signs: {result['value_signs'].str.split().explode().nunique()}")


if __name__ == '__main__':
    import argparse

    # Get script directory to build relative paths
    script_dir = Path(__file__).parent.parent  # Go up to v_1/
    project_root = script_dir.parent  # Go up to project root (for Akk folder)

    parser = argparse.ArgumentParser(description='Process ORACC JSONL to parquet')
    parser.add_argument(
        '--input_file',
        default=str(project_root / 'Akk/data/akk_from_jsons.jsonl'),
        help='Path to ORACC JSONL file'
    )
    parser.add_argument(
        '--output_file',
        default=str(script_dir / 'data/processed/oracc/oracc_corpus.parquet'),
        help='Output parquet file'
    )

    args = parser.parse_args()
    process_oracc_jsonl(args.input_file, args.output_file)
