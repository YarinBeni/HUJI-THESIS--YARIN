#!/usr/bin/env python3
"""
ORACC Catalog Explorer
======================
Downloads and analyzes ORACC project catalogs to find period/genre metadata.

ORACC Metadata Structure (from official docs):
- catalogue.json or index-cat.json contains metadata per text
- Fields include: period, provenience, genre, language, archive, etc.

Download URL format: http://oracc.museum.upenn.edu/json/[PROJECT].zip
Sub-projects: http://oracc.museum.upenn.edu/json/[PROJECT]-[SUBPROJECT].zip

Key 1st Millennium Projects:
- saao/saa01-saa21: State Archives of Assyria (Neo-Assyrian letters, admin docs)
- rinap: Royal Inscriptions of Neo-Assyrian Period
- cams/gkab: Geographical Knowledge in Ancient Babylonia
- aemw/alalakh: Alalakh tablets

Author: Diagnostic for embedding evaluation corpus selection
"""

import io
import json
import zipfile
from pathlib import Path
from collections import Counter
import requests
import warnings

# Suppress SSL warnings for old academic websites
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "oracc_catalogs"


def section_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def get_oracc_projects_list():
    """Fetch list of all public ORACC projects."""
    url = "http://oracc.museum.upenn.edu/projects.json"
    try:
        res = requests.get(url, verify=False, timeout=30)
        if res.status_code == 200:
            data = json.loads(res.content)
            return data.get('public', [])
    except Exception as e:
        print(f"Error fetching projects list: {e}")
    return []


def download_project_catalog(project_name: str, output_dir: Path = None) -> dict:
    """
    Download a project's JSON zip and extract catalog metadata.

    Args:
        project_name: e.g., 'saao/saa01', 'rinap', 'cams/gkab'
        output_dir: Optional directory to save the catalog

    Returns:
        Dictionary with catalog data or empty dict on failure
    """
    # Convert project path to URL format
    url_project = project_name.replace('/', '-')
    url = f"http://oracc.museum.upenn.edu/json/{url_project}.zip"

    print(f"  Downloading: {url}")

    try:
        r = requests.get(url, stream=True, timeout=60, verify=False)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}")
            return {}

        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            # List files in ZIP
            file_list = zf.namelist()

            # Look for catalog files
            catalog_files = [f for f in file_list if 'catalogue' in f.lower() or 'index-cat' in f.lower()]

            if not catalog_files:
                print(f"    No catalog found in: {file_list[:10]}...")
                return {}

            # Read the first catalog file found
            catalog_file = catalog_files[0]
            print(f"    Found catalog: {catalog_file}")

            with zf.open(catalog_file) as cf:
                catalog_data = json.load(cf)

            # Optionally save
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f"{url_project}_catalogue.json"
                with open(out_path, 'w') as f:
                    json.dump(catalog_data, f, indent=2)
                print(f"    Saved to: {out_path}")

            return catalog_data

    except zipfile.BadZipFile:
        print(f"    Bad ZIP file")
    except Exception as e:
        print(f"    Error: {e}")

    return {}


def analyze_catalog_metadata(catalog_data: dict, project_name: str) -> dict:
    """
    Analyze catalog metadata to extract period/genre distribution.

    Returns summary statistics.
    """
    if not catalog_data:
        return {}

    # ORACC catalogs typically have structure:
    # { "type": "catalogue", "project": "...", "members": { "P123456": {...}, ... } }
    # or just { "P123456": {...}, ... }

    members = catalog_data.get('members', catalog_data)
    if isinstance(members, dict) and 'type' in members:
        # Remove metadata keys
        members = {k: v for k, v in members.items() if not k.startswith('_') and k not in ['type', 'project']}

    if not members:
        return {}

    print(f"\n  Project: {project_name}")
    print(f"  Total texts: {len(members)}")

    # Sample one entry to see available fields
    sample_key = list(members.keys())[0]
    sample_entry = members[sample_key]

    print(f"\n  Sample entry ({sample_key}):")
    if isinstance(sample_entry, dict):
        for key, value in list(sample_entry.items())[:15]:
            val_str = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
            print(f"    {key}: {val_str}")

    # Collect statistics for key fields
    stats = {
        'total_texts': len(members),
        'fields': {},
    }

    key_fields = ['period', 'genre', 'provenience', 'language', 'date_of_origin',
                  'dates_referenced', 'archive', 'subgenre', 'object_type']

    for field in key_fields:
        values = []
        for entry in members.values():
            if isinstance(entry, dict) and field in entry:
                val = entry[field]
                if val and str(val).strip():
                    values.append(str(val))

        if values:
            counter = Counter(values)
            stats['fields'][field] = {
                'count': len(values),
                'unique': len(counter),
                'top_10': counter.most_common(10)
            }

            print(f"\n  {field}: {len(values)} texts, {len(counter)} unique values")
            for val, count in counter.most_common(5):
                val_str = val[:50] + "..." if len(val) > 50 else val
                print(f"    {count:5d}: {val_str}")

    return stats


def explore_key_projects():
    """Explore key ORACC projects for 1st millennium corpus."""
    section_header("EXPLORING KEY ORACC PROJECTS")

    # Key projects for 1st millennium Akkadian
    key_projects = [
        # Neo-Assyrian/Neo-Babylonian (1st millennium)
        'saao/saa01',  # State Archives of Assyria 01 - Letters
        'saao/saa05',  # SAA 05 - Letters
        'saao/saa10',  # SAA 10 - Letters
        'saao/saa13',  # SAA 13 - Letters
        'saao/saa15',  # SAA 15 - Letters
        'saao/saa16',  # SAA 16 - Legal texts
        'saao/saa17',  # SAA 17 - Letters
        'saao/saa18',  # SAA 18 - Letters
        'rinap/rinap1',  # Royal Inscriptions Neo-Assyrian
        'rinap/rinap3',
        'rinap/rinap4',
        'cams/gkab',   # Geographical Knowledge
        'aemw/alalakh',  # Alalakh
        # General corpora
        'dcclt',       # Digital Corpus of Cuneiform Lexical Texts
        'etcsri',      # Electronic Text Corpus of Sumerian Royal Inscriptions
    ]

    all_stats = {}

    for project in key_projects:
        print(f"\n{'='*60}")
        print(f"Project: {project}")
        print('='*60)

        catalog = download_project_catalog(project, OUTPUT_DIR)
        if catalog:
            stats = analyze_catalog_metadata(catalog, project)
            all_stats[project] = stats

    return all_stats


def check_existing_oracc_data():
    """Check what ORACC data we already have and analyze fragment IDs."""
    section_header("ANALYZING EXISTING ORACC DATA")

    if not HAS_PANDAS:
        print("pandas not available, skipping")
        return

    oracc_path = Path(__file__).parent.parent.parent / "data" / "processed" / "oracc" / "oracc_corpus.parquet"

    if not oracc_path.exists():
        print(f"ORACC corpus not found at {oracc_path}")
        return

    print(f"Loading ORACC corpus from {oracc_path}...")
    df = pd.read_parquet(oracc_path)

    print(f"\nTotal words: {len(df):,}")
    print(f"Total fragments: {df['fragment_id'].nunique():,}")

    # Analyze fragment ID patterns
    # ORACC uses P-numbers (e.g., P123456) from CDLI
    fragment_ids = df['fragment_id'].unique()

    print("\nFragment ID patterns:")

    # Count P-numbers
    p_numbers = [fid for fid in fragment_ids if fid.startswith('P') and fid[1:].isdigit()]
    q_numbers = [fid for fid in fragment_ids if fid.startswith('Q') and fid[1:].isdigit()]
    other = [fid for fid in fragment_ids if fid not in p_numbers and fid not in q_numbers]

    print(f"  P-numbers (CDLI texts): {len(p_numbers)}")
    print(f"  Q-numbers (composites): {len(q_numbers)}")
    print(f"  Other formats: {len(other)}")

    if other:
        print(f"\n  Sample other IDs: {other[:20]}")

    # Check if we can extract project info from fragment IDs
    # Some fragment IDs may encode project info
    print("\nSample fragment IDs:")
    for fid in list(fragment_ids[:30]):
        print(f"  {fid}")

    return fragment_ids


def generate_period_recommendations():
    """Generate recommendations for period-based filtering."""
    section_header("RECOMMENDATIONS FOR PERIOD FILTERING")

    print("""
Based on ORACC documentation and project analysis:

STRATEGY FOR 1ST MILLENNIUM CORPUS
==================================

Option 1: PROJECT-BASED FILTERING (Recommended)
-----------------------------------------------
Use ORACC project names to identify 1st millennium texts:

Neo-Assyrian Period (~934-609 BCE):
  - saao/saa*  : State Archives of Assyria (letters, admin, legal)
  - rinap/*    : Royal Inscriptions of Neo-Assyrian Period
  - aemw/*     : Archives of Egyptologists and Middle West

Neo-Babylonian Period (~626-539 BCE):
  - cams/gkab  : Babylonian astronomical/administrative

These projects are SPECIFICALLY dated to the 1st millennium.

Option 2: CATALOG METADATA FILTERING
------------------------------------
Download catalogue.json from each project and filter by:
  - period: "Neo-Assyrian", "Neo-Babylonian", "Late Babylonian"
  - genre: "Letter", "Administrative", "Legal"

Option 3: P-NUMBER CROSS-REFERENCE
----------------------------------
Use P-numbers to query CDLI (Cuneiform Digital Library Initiative)
which has comprehensive period metadata for all texts.

IMPLEMENTATION PLAN
===================
1. Download catalogs from key 1st millennium projects
2. Extract P-numbers and their period/genre metadata
3. Match against our existing ORACC corpus fragment_ids
4. Create filtered subset for embedding evaluation

For ARCHIBAB (2nd Millennium):
- Already have genre metadata (lettre administrative, etc.)
- Source is specifically Old Babylonian period
- Can use as-is for Corpus A
""")


def main():
    """Run ORACC catalog exploration."""
    print("\n" + "=" * 80)
    print("  ORACC CATALOG EXPLORER")
    print("  Finding Period/Genre Metadata for 1st Millennium Corpus")
    print("=" * 80)

    # Check existing data
    check_existing_oracc_data()

    # Get list of all projects
    section_header("FETCHING ORACC PROJECTS LIST")
    projects = get_oracc_projects_list()
    if projects:
        print(f"Found {len(projects)} public projects")
        print("\nFirst 30 projects:")
        for p in projects[:30]:
            print(f"  {p}")

        # Filter for likely 1st millennium projects
        first_mill_keywords = ['saa', 'rinap', 'neo', 'babylon', 'assyria', 'nineveh']
        likely_1st_mill = [p for p in projects if any(kw in p.lower() for kw in first_mill_keywords)]

        print(f"\nLikely 1st millennium projects ({len(likely_1st_mill)}):")
        for p in likely_1st_mill:
            print(f"  {p}")
    else:
        print("Could not fetch projects list (connection issue)")

    # Explore key projects
    stats = explore_key_projects()

    # Generate recommendations
    generate_period_recommendations()

    # Summary
    section_header("SUMMARY")
    print(f"""
Catalogs downloaded to: {OUTPUT_DIR}

Next steps:
1. Run this script to download ORACC catalogs
2. Use the period/genre metadata to filter texts
3. Match P-numbers against our existing ORACC corpus
4. Create filtered evaluation corpora

For questions about specific projects:
- SAA (State Archives of Assyria): Neo-Assyrian letters & admin docs
- RINAP: Royal inscriptions, 1st millennium
- Check http://oracc.org for full project documentation
""")


if __name__ == "__main__":
    main()
