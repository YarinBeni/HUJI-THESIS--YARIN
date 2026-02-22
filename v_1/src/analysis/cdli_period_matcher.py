#!/usr/bin/env python3
"""
CDLI Period Matcher
===================
Uses CDLI (Cuneiform Digital Library Initiative) catalog to get period/genre
metadata for texts in our ORACC corpus.

Our ORACC corpus contains P-numbers (e.g., P224485) which are CDLI identifiers.
CDLI has comprehensive metadata including:
- period: e.g., "Old Babylonian (ca. 1900-1600 BC)", "Neo-Assyrian (ca. 911-612 BC)"
- genre: text genre classification
- provenience: archaeological findspot
- language: Akkadian, Sumerian, etc.

HOW TO GET CDLI CATALOG DATA
============================

Option 1: GitHub Repository (Recommended for bulk data)
-------------------------------------------------------
1. Install Git LFS: https://git-lfs.github.com/
2. Clone: git clone https://github.com/cdli-gh/data
3. The catalog is in: cdli_cat.csv

Option 2: CDLI Search Export
----------------------------
1. Go to https://cdli.earth/search
2. Search for texts (e.g., by P-number range)
3. Export as CSV

Option 3: CDLI Bulk Data Request
--------------------------------
Contact cdli@ucla.edu for custom bulk data exports

KEY PERIOD VALUES IN CDLI
=========================
2nd Millennium (for ARCHIBAB comparison):
- "Old Babylonian (ca. 1900-1600 BC)"
- "Old Assyrian (ca. 1950-1850 BC)"
- "Middle Babylonian (ca. 1400-1100 BC)"
- "Middle Assyrian (ca. 1400-1000 BC)"

1st Millennium (for ORACC corpus):
- "Neo-Assyrian (ca. 911-612 BC)"
- "Neo-Babylonian (ca. 626-539 BC)"
- "Late Babylonian (ca. 539-64 BC)"
- "Achaemenid (ca. 550-331 BC)"
- "Hellenistic (ca. 323-63 BC)"

Author: For embedding evaluation corpus preparation
"""

import pandas as pd
import re
from pathlib import Path
from collections import Counter
from typing import Optional, Set, Dict, List

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def section_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_oracc_pnumbers() -> Set[str]:
    """Load P-numbers from our ORACC corpus."""
    oracc_path = PROCESSED_DIR / "oracc" / "oracc_corpus.parquet"

    if not oracc_path.exists():
        print(f"ORACC corpus not found at {oracc_path}")
        return set()

    print(f"Loading ORACC corpus from {oracc_path}...")
    df = pd.read_parquet(oracc_path)

    fragment_ids = df['fragment_id'].unique()

    # Extract P-numbers
    p_numbers = set()
    for fid in fragment_ids:
        if fid.startswith('P') and len(fid) >= 7:
            # P-numbers are typically P followed by 6 digits
            match = re.match(r'(P\d+)', fid)
            if match:
                p_numbers.add(match.group(1))

    print(f"Found {len(p_numbers)} unique P-numbers in ORACC corpus")
    return p_numbers


def load_cdli_catalog(catalog_path: str) -> Optional[pd.DataFrame]:
    """
    Load CDLI catalog CSV.

    Expected columns include:
    - id_text (P-number)
    - period
    - genre
    - provenience
    - language
    """
    path = Path(catalog_path)
    if not path.exists():
        print(f"CDLI catalog not found at {path}")
        return None

    print(f"Loading CDLI catalog from {path}...")

    # Try different delimiters and encodings
    for sep in [',', '\t']:
        for encoding in ['utf-8', 'latin-1']:
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
                if len(df.columns) > 5:  # Sanity check
                    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
                    return df
            except:
                continue

    print("Could not parse CDLI catalog")
    return None


def analyze_cdli_metadata(df: pd.DataFrame, our_pnumbers: Set[str]) -> Dict:
    """Analyze CDLI metadata for our P-numbers."""
    section_header("CDLI METADATA ANALYSIS")

    # Find the P-number column
    id_cols = [c for c in df.columns if 'id' in c.lower() or c.lower() == 'artifact_id']
    if not id_cols:
        print("Could not find ID column")
        print(f"Available columns: {df.columns.tolist()}")
        return {}

    id_col = id_cols[0]
    print(f"Using ID column: {id_col}")

    # Match our P-numbers
    df['pnumber'] = df[id_col].astype(str).apply(lambda x: re.match(r'(P\d+)', x).group(1) if re.match(r'P\d+', x) else None)

    matched = df[df['pnumber'].isin(our_pnumbers)]
    print(f"\nMatched {len(matched):,} texts from our ORACC corpus")

    if len(matched) == 0:
        return {}

    # Analyze key fields
    results = {}

    # Period analysis
    if 'period' in df.columns:
        period_counts = matched['period'].value_counts()
        results['period'] = period_counts.to_dict()

        print("\nPeriod Distribution in Our ORACC Corpus:")
        for period, count in period_counts.head(20).items():
            period_str = str(period)[:60] if period else "Unknown"
            print(f"  {count:6,}: {period_str}")

        # Classify into 1st/2nd millennium
        first_mill_keywords = ['neo-assyrian', 'neo-babylonian', 'late babylonian',
                               'achaemenid', 'hellenistic', 'seleucid', 'parthian']
        second_mill_keywords = ['old babylonian', 'old assyrian', 'middle babylonian',
                                'middle assyrian', 'kassite']

        first_mill = matched[matched['period'].str.lower().str.contains('|'.join(first_mill_keywords), na=False)]
        second_mill = matched[matched['period'].str.lower().str.contains('|'.join(second_mill_keywords), na=False)]

        print(f"\n1st Millennium texts: {len(first_mill):,}")
        print(f"2nd Millennium texts: {len(second_mill):,}")

        results['first_millennium_pnumbers'] = first_mill['pnumber'].tolist()
        results['second_millennium_pnumbers'] = second_mill['pnumber'].tolist()

    # Genre analysis
    genre_cols = [c for c in df.columns if 'genre' in c.lower()]
    if genre_cols:
        genre_col = genre_cols[0]
        genre_counts = matched[genre_col].value_counts()
        results['genre'] = genre_counts.to_dict()

        print(f"\nGenre Distribution (from '{genre_col}' column):")
        for genre, count in genre_counts.head(15).items():
            genre_str = str(genre)[:50] if genre else "Unknown"
            print(f"  {count:6,}: {genre_str}")

        # Filter for epistolary/administrative
        epistolary_keywords = ['letter', 'epistle', 'correspondence']
        admin_keywords = ['administrative', 'economic', 'legal', 'receipt', 'account']

        epistolary = matched[matched[genre_col].str.lower().str.contains('|'.join(epistolary_keywords), na=False)]
        admin = matched[matched[genre_col].str.lower().str.contains('|'.join(admin_keywords), na=False)]

        print(f"\nEpistolary texts: {len(epistolary):,}")
        print(f"Administrative texts: {len(admin):,}")

    # Language analysis
    lang_cols = [c for c in df.columns if 'language' in c.lower()]
    if lang_cols:
        lang_col = lang_cols[0]
        lang_counts = matched[lang_col].value_counts()
        results['language'] = lang_counts.to_dict()

        print(f"\nLanguage Distribution:")
        for lang, count in lang_counts.head(10).items():
            print(f"  {count:6,}: {lang}")

    return results


def generate_corpus_recommendations(results: Dict):
    """Generate recommendations for corpus creation."""
    section_header("CORPUS CREATION RECOMMENDATIONS")

    first_mill = results.get('first_millennium_pnumbers', [])
    second_mill = results.get('second_millennium_pnumbers', [])

    print(f"""
EVALUATION CORPUS PREPARATION
=============================

Based on CDLI metadata analysis:

CORPUS A: 2nd Millennium (ARCHIBAB)
-----------------------------------
Source: ARCHIBAB corpus (already processed)
Texts available: ~1,310 fragments
Genre filter: "lettre administrative", "lettre politique"
Period: Old Babylonian (assumed from source)

Status: ✅ READY - Use ARCHIBAB data directly

CORPUS B: 1st Millennium (ORACC)
--------------------------------
Source: ORACC corpus (P-numbers matched via CDLI)
1st millennium texts identified: {len(first_mill):,} P-numbers
""")

    if first_mill:
        print(f"""
These P-numbers can be filtered from our ORACC corpus:
{first_mill[:10]}... (showing first 10)

To extract 1st millennium corpus, filter ORACC by these P-numbers.
""")
    else:
        print("""
⚠ No 1st millennium texts identified yet.

To proceed, you need to:
1. Download CDLI catalog (see instructions at top of script)
2. Re-run this script with: --cdli_catalog /path/to/cdli_cat.csv
3. The script will identify 1st millennium texts
""")

    print("""
ALTERNATIVE: PROJECT-BASED FILTERING
------------------------------------
If CDLI catalog is unavailable, use ORACC project names:

SAA (State Archives of Assyria) projects contain:
- Neo-Assyrian letters (saao/saa01, saa05, saa10, saa13, saa15, saa17, saa18)
- Neo-Assyrian administrative/legal (saao/saa06, saa11, saa14, saa16)

These are definitively 1st millennium.

You can filter texts by checking if their project source starts with "saao" or "rinap".
""")


def create_filtered_corpus(oracc_df: pd.DataFrame, pnumbers: List[str], output_path: Path) -> pd.DataFrame:
    """Create filtered corpus from P-numbers."""
    filtered = oracc_df[oracc_df['fragment_id'].isin(pnumbers)]

    print(f"\nFiltered corpus: {len(filtered):,} words, {filtered['fragment_id'].nunique()} fragments")

    filtered.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")

    return filtered


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description='Match ORACC P-numbers to CDLI metadata')
    parser.add_argument('--cdli_catalog', type=str, default=None,
                        help='Path to CDLI catalog CSV (cdli_cat.csv)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save filtered corpora')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  CDLI PERIOD MATCHER")
    print("  Matching ORACC P-numbers to CDLI Metadata")
    print("=" * 80)

    # Load our P-numbers
    section_header("LOADING ORACC P-NUMBERS")
    our_pnumbers = load_oracc_pnumbers()

    if not our_pnumbers:
        print("No P-numbers found in ORACC corpus")
        return

    # Summary of P-numbers
    print(f"\nP-number range: {min(our_pnumbers)} to {max(our_pnumbers)}")
    print(f"Sample P-numbers: {sorted(list(our_pnumbers))[:10]}")

    # If CDLI catalog provided, analyze it
    results = {}
    if args.cdli_catalog:
        cdli_df = load_cdli_catalog(args.cdli_catalog)
        if cdli_df is not None:
            results = analyze_cdli_metadata(cdli_df, our_pnumbers)

    # Generate recommendations
    generate_corpus_recommendations(results)

    # Instructions for next steps
    section_header("NEXT STEPS")
    print("""
1. DOWNLOAD CDLI CATALOG:
   git clone https://github.com/cdli-gh/data
   (requires git-lfs)

   OR export from https://cdli.earth/search

2. RUN WITH CATALOG:
   python3 cdli_period_matcher.py --cdli_catalog /path/to/cdli_cat.csv

3. CREATE FILTERED CORPORA:
   The script will identify 1st millennium P-numbers
   and help you create Corpus B for evaluation.

4. FOR CORPUS A (2nd Millennium):
   Use the ARCHIBAB corpus directly - it's already Old Babylonian.
""")


if __name__ == "__main__":
    main()
