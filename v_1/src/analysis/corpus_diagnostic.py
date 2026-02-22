#!/usr/bin/env python3
"""
Corpus Diagnostic Script
========================
Answers key questions for the embedding evaluation approach:

1. Do we have metadata to filter correctly?
   - ARCHIBAB: epistolary + administrative, 2nd millennium
   - ORACC: epistolary + administrative, 1st millennium

2. What are the actual transliteration differences between sources?
   - Subscript conventions
   - Determinative notation
   - Sign reading choices

Author: Diagnostic for Nathan's suggested evaluation approach
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
import json

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# SECTION 1: METADATA ANALYSIS
# =============================================================================

def analyze_metadata():
    """Analyze available metadata fields for filtering."""
    section_header("1. METADATA ANALYSIS FOR FILTERING")

    # Load processed data from each source
    archibab_path = PROCESSED_DIR / "archibab" / "archibab_corpus.parquet"
    oracc_path = PROCESSED_DIR / "oracc" / "oracc_corpus.parquet"
    ebl_path = PROCESSED_DIR / "ebl" / "ebl_corpus.parquet"

    print("\nLoading processed corpora...")

    dfs = {}
    for name, path in [("archibab", archibab_path), ("oracc", oracc_path), ("ebl", ebl_path)]:
        if path.exists():
            dfs[name] = pd.read_parquet(path)
            print(f"  ✓ {name}: {len(dfs[name]):,} words, {dfs[name]['fragment_id'].nunique():,} fragments")
        else:
            print(f"  ✗ {name}: File not found at {path}")

    # Analyze each corpus
    for name, df in dfs.items():
        subsection(f"{name.upper()} Metadata Fields")
        print(f"\nColumns available: {list(df.columns)}")

        # Check each metadata field
        for col in df.columns:
            non_null = df[col].notna().sum()
            non_empty = (df[col].astype(str) != '').sum() if non_null > 0 else 0
            unique = df[col].nunique() if non_null > 0 else 0
            print(f"  {col}: {non_null:,} non-null ({non_null/len(df)*100:.1f}%), {unique} unique values")

        # Domain analysis (key for filtering)
        if 'domain' in df.columns:
            subsection(f"{name.upper()} - Domain/Genre Distribution")
            domain_counts = df.groupby('fragment_id')['domain'].first().value_counts()
            print(f"\nTop 20 domain values (by fragment count):")
            for domain, count in domain_counts.head(20).items():
                # Truncate long domain strings
                domain_str = str(domain)[:70] + "..." if len(str(domain)) > 70 else str(domain)
                print(f"  {count:5d} fragments: {domain_str}")

        # Place composition analysis (potential time period info)
        if 'place_composition' in df.columns:
            subsection(f"{name.upper()} - Place/Period Composition")
            place_counts = df.groupby('fragment_id')['place_composition'].first().value_counts()
            print(f"\nTop 20 place_composition values (by fragment count):")
            for place, count in place_counts.head(20).items():
                place_str = str(place)[:70] + "..." if len(str(place)) > 70 else str(place)
                print(f"  {count:5d} fragments: {place_str}")

    return dfs


def analyze_raw_archibab():
    """Analyze raw ARCHIBAB data for additional metadata."""
    section_header("1b. RAW ARCHIBAB METADATA")

    raw_path = RAW_DIR / "archibab.csv"
    if not raw_path.exists():
        print(f"Raw ARCHIBAB file not found at {raw_path}")
        return None

    print(f"\nReading raw ARCHIBAB from {raw_path}...")
    df = pd.read_csv(raw_path)

    print(f"\nTotal rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Show all column stats
    subsection("Column Statistics")
    for col in df.columns:
        non_null = df[col].notna().sum()
        unique = df[col].nunique() if non_null > 0 else 0
        print(f"  {col}: {non_null:,} non-null ({non_null/len(df)*100:.1f}%), {unique} unique")

    # Check for time/date related columns
    date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['date', 'year', 'period', 'time', 'era', 'millennium'])]
    if date_cols:
        subsection("Date/Period Columns Found")
        for col in date_cols:
            print(f"\n{col} value distribution:")
            print(df[col].value_counts().head(20))
    else:
        print("\n⚠ No explicit date/period columns found in raw ARCHIBAB")

    # Domain analysis
    if 'domain' in df.columns:
        subsection("ARCHIBAB Domain Categories (Raw)")
        print(df['domain'].value_counts().head(30))

    return df


def analyze_oracc_raw():
    """Analyze raw ORACC data for metadata."""
    section_header("1c. RAW ORACC METADATA")

    # Check for ORACC JSONL in external location
    oracc_paths = [
        Path("/Users/yarin.b/git/Akk/data/akk_from_jsons.jsonl"),
        RAW_DIR / "oracc.jsonl",
        RAW_DIR / "akk_from_jsons.jsonl"
    ]

    oracc_path = None
    for p in oracc_paths:
        if p.exists():
            oracc_path = p
            break

    if oracc_path is None:
        print("Raw ORACC JSONL file not found. Checked paths:")
        for p in oracc_paths:
            print(f"  - {p}")
        return None

    print(f"\nReading ORACC from {oracc_path}...")

    # Sample first 100 records to understand structure
    records = []
    with open(oracc_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Sample first 1000 records
                break
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("Could not parse any ORACC records")
        return None

    print(f"\nSampled {len(records)} records")

    # Analyze fields present
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    print(f"\nFields in ORACC records: {sorted(all_keys)}")

    # Check key fields
    subsection("Sample ORACC Record Structure")
    if records:
        print(json.dumps(records[0], indent=2, ensure_ascii=False)[:2000])

    # Look for time/period info
    date_fields = [k for k in all_keys if any(kw in k.lower() for kw in ['date', 'period', 'year', 'era', 'millennium', 'time'])]
    if date_fields:
        subsection("Date/Period Fields in ORACC")
        for field in date_fields:
            values = [r.get(field) for r in records if r.get(field)]
            print(f"\n{field}: {len(values)} non-null values")
            value_counts = Counter(values)
            for v, c in value_counts.most_common(10):
                print(f"  {c}: {v}")
    else:
        print("\n⚠ No explicit date/period fields found in ORACC sample")

    # Genre field
    genre_fields = [k for k in all_keys if any(kw in k.lower() for kw in ['genre', 'category', 'type', 'class'])]
    if genre_fields:
        subsection("Genre/Category Fields in ORACC")
        for field in genre_fields:
            values = [r.get(field) for r in records if r.get(field)]
            print(f"\n{field}: {len(values)} non-null values")
            value_counts = Counter(values)
            for v, c in value_counts.most_common(20):
                print(f"  {c}: {v}")

    return records


# =============================================================================
# SECTION 2: TRANSLITERATION DIFFERENCES
# =============================================================================

def analyze_transliteration_differences(dfs: dict):
    """Compare transliteration conventions between corpora."""
    section_header("2. TRANSLITERATION DIFFERENCES ANALYSIS")

    if not dfs:
        print("No data loaded for analysis")
        return

    # Extract sample texts from each corpus
    for name, df in dfs.items():
        subsection(f"{name.upper()} - Transliteration Samples")

        # Get 10 sample fragments
        sample_fragments = df['fragment_id'].unique()[:10]

        for frag_id in sample_fragments[:3]:
            frag_df = df[df['fragment_id'] == frag_id].sort_values(['line_num', 'word_idx'])

            print(f"\nFragment: {frag_id}")

            # Show raw vs clean vs signs
            raw_values = frag_df['value_raw'].head(15).tolist() if 'value_raw' in frag_df.columns else []
            clean_values = frag_df['value_clean'].head(15).tolist() if 'value_clean' in frag_df.columns else []
            sign_values = frag_df['value_signs'].head(15).tolist() if 'value_signs' in frag_df.columns else []

            if raw_values:
                print(f"  Raw:   {' '.join(str(v) for v in raw_values)}")
            if clean_values:
                print(f"  Clean: {' '.join(str(v) for v in clean_values)}")
            if sign_values:
                print(f"  Signs: {' '.join(str(v) for v in sign_values)}")

    # Compare specific transliteration patterns
    subsection("2b. Subscript Pattern Analysis")

    subscript_patterns = {
        'unicode_subscripts': re.compile(r'[₀₁₂₃₄₅₆₇₈₉]'),
        'ascii_subscripts': re.compile(r'[a-zA-Z][0-9]+(?![0-9])'),  # e.g., ša2, not numbers
        'superscripts': re.compile(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]'),
        'determinatives_curly': re.compile(r'\{[^}]+\}'),  # {d}, {m}, etc.
        'determinatives_super': re.compile(r'[ᵈᵐᶠᵍ]'),  # Superscript determinatives
        'brackets': re.compile(r'[\[\]⸢⸣<>]'),  # Editorial marks
        'x_damage': re.compile(r'\bx\b'),  # Damage markers
    }

    for name, df in dfs.items():
        print(f"\n{name.upper()} - Pattern Counts (in raw values):")
        if 'value_raw' not in df.columns:
            print("  No value_raw column")
            continue

        raw_text = ' '.join(df['value_raw'].dropna().astype(str).tolist())

        for pattern_name, pattern in subscript_patterns.items():
            matches = pattern.findall(raw_text)
            if matches:
                unique_matches = set(matches)
                print(f"  {pattern_name}: {len(matches):,} occurrences, {len(unique_matches)} unique")
                print(f"    Examples: {list(unique_matches)[:10]}")

    # Compare sign vocabularies
    subsection("2c. Sign Vocabulary Comparison")

    sign_vocabs = {}
    for name, df in dfs.items():
        if 'value_signs' in df.columns:
            all_signs = ' '.join(df['value_signs'].dropna().astype(str).tolist())
            signs = all_signs.split()
            sign_vocabs[name] = set(signs)
            print(f"\n{name}: {len(sign_vocabs[name]):,} unique signs")

    if len(sign_vocabs) >= 2:
        subsection("Sign Overlap Analysis")
        names = list(sign_vocabs.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                s1, s2 = sign_vocabs[name1], sign_vocabs[name2]
                overlap = s1 & s2
                only_1 = s1 - s2
                only_2 = s2 - s1
                print(f"\n{name1} vs {name2}:")
                print(f"  Overlap: {len(overlap):,} signs ({len(overlap)/len(s1|s2)*100:.1f}%)")
                print(f"  Only in {name1}: {len(only_1):,}")
                print(f"  Only in {name2}: {len(only_2):,}")

                if only_1:
                    print(f"  Sample {name1}-only: {list(only_1)[:20]}")
                if only_2:
                    print(f"  Sample {name2}-only: {list(only_2)[:20]}")


def analyze_normalization_effectiveness():
    """Check if current preprocessing normalization is sufficient."""
    section_header("2d. NORMALIZATION EFFECTIVENESS")

    # Load unified data
    unified_path = PROCESSED_DIR / "unified" / "unified_corpus.parquet"
    if not unified_path.exists():
        print(f"Unified corpus not found at {unified_path}")
        return

    print("Loading unified corpus...")
    df = pd.read_parquet(unified_path)

    # Check for remaining variations
    subsection("Remaining Variations in Normalized Data")

    # Check value_signs for variations
    if 'value_signs' in df.columns:
        all_signs = ' '.join(df['value_signs'].dropna().astype(str).tolist())
        signs = all_signs.split()

        # Look for potential remaining issues
        issues = {
            'numbers_in_signs': [s for s in set(signs) if re.search(r'\d', s)],
            'brackets_remaining': [s for s in set(signs) if re.search(r'[\[\]⸢⸣<>]', s)],
            'subscripts_remaining': [s for s in set(signs) if re.search(r'[₀₁₂₃₄₅₆₇₈₉]', s)],
            'empty_or_whitespace': [s for s in set(signs) if not s.strip()],
        }

        for issue_name, issue_signs in issues.items():
            if issue_signs:
                print(f"\n{issue_name}: {len(issue_signs)} unique signs")
                print(f"  Examples: {issue_signs[:20]}")
            else:
                print(f"\n✓ {issue_name}: None found")

    # Compare raw vs normalized by source
    subsection("Raw vs Normalized Examples by Source")

    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        sample = source_df.sample(min(5, len(source_df)))

        print(f"\n{source.upper()}:")
        for _, row in sample.iterrows():
            raw = row.get('value_raw', 'N/A')
            clean = row.get('value_clean', 'N/A')
            signs = row.get('value_signs', 'N/A')
            print(f"  {raw} → {clean} → [{signs}]")


# =============================================================================
# SECTION 3: FILTERING RECOMMENDATIONS
# =============================================================================

def generate_filtering_recommendations(dfs: dict):
    """Generate recommendations for corpus filtering."""
    section_header("3. FILTERING RECOMMENDATIONS")

    print("""
Based on the analysis, here are recommendations for creating the evaluation corpora:

CORPUS A: ARCHIBAB (2nd Millennium)
===================================
""")

    if 'archibab' in dfs:
        archibab = dfs['archibab']

        # Check domain values
        if 'domain' in archibab.columns:
            domains = archibab.groupby('fragment_id')['domain'].first().value_counts()
            print("Available domain categories in ARCHIBAB:")
            for d, c in domains.items():
                print(f"  {c:5d} fragments: {d}")

            # Suggest filters
            epistolary_patterns = ['lettre', 'letter', 'épistol']
            admin_patterns = ['admin', 'économ', 'econom', 'juridique', 'legal']

            print("\nSuggested epistolary filter patterns:", epistolary_patterns)
            print("Suggested administrative filter patterns:", admin_patterns)

    print("""
CORPUS B: ORACC (1st Millennium)
================================
""")

    if 'oracc' in dfs:
        oracc = dfs['oracc']

        if 'domain' in oracc.columns:
            domains = oracc.groupby('fragment_id')['domain'].first().value_counts()
            print("Available domain categories in ORACC:")
            for d, c in domains.head(30).items():
                print(f"  {c:5d} fragments: {d}")

    print("""
TIME PERIOD FILTERING STATUS
============================
⚠ IMPORTANT: Based on the analysis, TIME PERIOD metadata is NOT reliably available
in the processed data. You have several options:

1. RELY ON SOURCE ASSUMPTION:
   - ARCHIBAB is predominantly Old Babylonian (2nd millennium)
   - ORACC covers various periods including Neo-Assyrian/Neo-Babylonian (1st millennium)
   - This is an approximation, not a hard filter

2. CHECK RAW DATA:
   - ORACC catalogs may have period info in original JSON files
   - ARCHIBAB source data may have date fields not in our processed version

3. EXTERNAL METADATA:
   - Cross-reference fragment IDs with external period databases
   - Use text-internal dating (year names, king names) - requires expertise

RECOMMENDED ACTION:
------------------
For the embedding evaluation, start with the source-based separation:
- Corpus A = All ARCHIBAB epistolary/administrative texts (assumed 2nd millennium)
- Corpus B = ORACC epistolary/administrative texts (need to verify period)

This gives you genre-controlled comparison, even if period filtering is approximate.
""")


# =============================================================================
# SECTION 4: SUMMARY STATISTICS FOR EMBEDDING EVALUATION
# =============================================================================

def summary_for_embedding_eval(dfs: dict):
    """Generate summary statistics relevant for embedding evaluation."""
    section_header("4. CORPUS STATISTICS FOR EMBEDDING EVALUATION")

    for name, df in dfs.items():
        subsection(f"{name.upper()} Corpus Statistics")

        n_words = len(df)
        n_fragments = df['fragment_id'].nunique()

        # Fragment length distribution
        frag_lengths = df.groupby('fragment_id').size()

        print(f"\nTotal words: {n_words:,}")
        print(f"Total fragments: {n_fragments:,}")
        print(f"\nFragment length distribution (words):")
        print(f"  Mean:   {frag_lengths.mean():.1f}")
        print(f"  Median: {frag_lengths.median():.1f}")
        print(f"  Std:    {frag_lengths.std():.1f}")
        print(f"  Min:    {frag_lengths.min()}")
        print(f"  Max:    {frag_lengths.max()}")
        print(f"  25%:    {frag_lengths.quantile(0.25):.1f}")
        print(f"  75%:    {frag_lengths.quantile(0.75):.1f}")
        print(f"  95%:    {frag_lengths.quantile(0.95):.1f}")

        # Vocabulary in signs
        if 'value_signs' in df.columns:
            all_signs = ' '.join(df['value_signs'].dropna().astype(str).tolist())
            signs = all_signs.split()
            vocab = set(signs)
            print(f"\nVocabulary size (signs): {len(vocab):,}")

            # Top signs
            sign_counts = Counter(signs)
            print(f"Top 20 signs:")
            for sign, count in sign_counts.most_common(20):
                print(f"  {count:7,} ({count/len(signs)*100:5.2f}%): {sign}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all diagnostic analyses."""
    print("\n" + "=" * 80)
    print("  CORPUS DIAGNOSTIC REPORT")
    print("  For Embedding Evaluation Preparation")
    print("=" * 80)

    # 1. Metadata analysis
    dfs = analyze_metadata()

    # 1b. Raw ARCHIBAB analysis
    analyze_raw_archibab()

    # 1c. Raw ORACC analysis
    analyze_oracc_raw()

    # 2. Transliteration differences
    if dfs:
        analyze_transliteration_differences(dfs)

    # 2d. Normalization effectiveness
    analyze_normalization_effectiveness()

    # 3. Filtering recommendations
    if dfs:
        generate_filtering_recommendations(dfs)

    # 4. Summary statistics
    if dfs:
        summary_for_embedding_eval(dfs)

    print("\n" + "=" * 80)
    print("  END OF DIAGNOSTIC REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
