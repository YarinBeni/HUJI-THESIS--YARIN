import pandas as pd
from pathlib import Path

# Define paths
DATA_DIR = Path('v_1/data/processed/from_chungrong')  # Where the source CSVs are
OUTPUT_DIR = Path('v_1/data/evaluation_corpora')  # Where to save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Creating unified dataset with all 3 groups...")
print("=" * 80)

# ============================================================================
# STEP 1: Load all three corpora
# ============================================================================
group1 = pd.read_csv(DATA_DIR / 'archibab_nor.csv')
group2_full = pd.read_csv(DATA_DIR / 'oracc_let_adm_nor.csv')
group3 = pd.read_csv(DATA_DIR / 'lbl_nor.csv')

print(f"\n✓ Loaded source files:")
print(f"  - archibab_nor.csv: {len(group1):,} words")
print(f"  - oracc_let_adm_nor.csv: {len(group2_full):,} words")
print(f"  - lbl_nor.csv: {len(group3):,} words")

# ============================================================================
# STEP 2: Apply domain merges to Group 1 (archibab)
# ============================================================================
domain_merge_map = {
    'inconnu, lettre cassée': 'Unknown',
    'lettre administratif': 'lettre administrative',
    'UNKNOWN': 'Unknown',
    'lettre privée (personnelle)': 'lettre privée'
}

print(f"\n✓ Applying {len(domain_merge_map)} domain merges to Group 1...")
for original, target in domain_merge_map.items():
    n_merged = len(group1[group1['domain'] == original])
    if n_merged > 0:
        group1.loc[group1['domain'] == original, 'domain'] = target
        print(f"  - '{original}' → '{target}': {n_merged:,} words")

# ============================================================================
# STEP 3: Filter Group 2 (oracc) for letters only
# ============================================================================
letter_domain = [d for d in group2_full['domain'].unique() if 'let' in str(d).lower()]
filter_value = letter_domain[0] if letter_domain else 'NALet'
group2 = group2_full[group2_full['domain'] == filter_value].copy()

print(f"\n✓ Filtered Group 2 for letters (domain == '{filter_value}'):")
print(f"  - Before: {len(group2_full):,} words")
print(f"  - After: {len(group2):,} words")
print(f"  - Removed: {len(group2_full) - len(group2):,} words")

# ============================================================================
# STEP 4: Add metadata columns to all groups
# ============================================================================
print(f"\n✓ Adding metadata columns...")

# Group 1 metadata
group1['temporal_group'] = 'Group 1'
group1['period'] = 'Old Babylonian'
group1['period_approx'] = '2nd millennium BCE (~1800 BCE)'
group1['corpus_source'] = 'archibab'

# Group 2 metadata
group2['temporal_group'] = 'Group 2'
group2['period'] = 'Neo-Assyrian'
group2['period_approx'] = '9-7 cent BCE'
group2['corpus_source'] = 'oracc'

# Group 3 metadata
group3['temporal_group'] = 'Group 3'
group3['period'] = 'Late Babylonian'
group3['period_approx'] = '6-4 cent BCE (~600 BCE)'
group3['corpus_source'] = 'lbl'

# ============================================================================
# STEP 5: Apply domain standardization to all groups
# ============================================================================
print(f"\n✓ Creating domain standardization mappings...")

# Build domain mappings
all_archibab_domains = group1['domain'].unique().tolist()
all_oracc_domains = group2['domain'].unique().tolist()
all_lbl_domains = group3['domain'].unique().tolist()

domain_to_standard = {}
domain_to_finegrained = {}

# Map archibab domains
for d in all_archibab_domains:
    d_str = str(d) if d else 'Unknown'
    d_lower = d_str.lower()
    
    # Standard mapping
    if 'lettre' in d_lower or 'letter' in d_lower:
        domain_to_standard[d] = 'Letter'
    elif 'unknown' in d_lower:
        domain_to_standard[d] = 'Unknown'
    else:
        domain_to_standard[d] = 'Other'
    
    # Fine-grained mapping
    if 'administrative' in d_lower:
        domain_to_finegrained[d] = 'Administrative Letter'
    elif 'politique' in d_lower:
        domain_to_finegrained[d] = 'Political Letter'
    elif 'privée' in d_lower:
        domain_to_finegrained[d] = 'Private Letter'
    elif 'diplomatique' in d_lower:
        domain_to_finegrained[d] = 'Diplomatic Letter'
    elif 'unknown' in d_lower:
        domain_to_finegrained[d] = 'Unknown'
    else:
        domain_to_finegrained[d] = d_str

# Map oracc domains
for d in all_oracc_domains:
    d_str = str(d) if d else 'Unknown'
    if 'NALet' in d_str:
        domain_to_standard[d] = 'Letter'
        domain_to_finegrained[d] = 'Neo-Assyrian Letter'
    elif 'NAAdm' in d_str:
        domain_to_standard[d] = 'Administrative'
        domain_to_finegrained[d] = 'Neo-Assyrian Administrative'
    else:
        domain_to_standard[d] = 'Other'
        domain_to_finegrained[d] = d_str

# Map lbl domains
for d in all_lbl_domains:
    d_str = str(d) if d else 'Unknown'
    if 'letter' in d_str.lower():
        domain_to_standard[d] = 'Letter'
        domain_to_finegrained[d] = 'Late Babylonian Letter'
    else:
        domain_to_standard[d] = 'Other'
        domain_to_finegrained[d] = d_str

# Apply domain mappings
group1['domain_original'] = group1['domain']
group1['domain_standard'] = group1['domain'].map(domain_to_standard)
group1['domain_finegrained'] = group1['domain'].map(domain_to_finegrained)

group2['domain_original'] = group2['domain']
group2['domain_standard'] = group2['domain'].map(domain_to_standard)
group2['domain_finegrained'] = group2['domain'].map(domain_to_finegrained)

group3['domain_original'] = group3['domain']
group3['domain_standard'] = group3['domain'].map(domain_to_standard)
group3['domain_finegrained'] = group3['domain'].map(domain_to_finegrained)

# ============================================================================
# STEP 6: Combine all three groups
# ============================================================================
print(f"\n✓ Combining all three groups...")
combined = pd.concat([group1, group2, group3], ignore_index=True)

# ============================================================================
# STEP 6.5: Fix mixed types for parquet compatibility
# ============================================================================
print(f"✓ Standardizing column types...")

# Convert potentially mixed-type columns to strings (handles NaN gracefully)
string_cols = ['fragment_id', 'fragment_line_num', 'word_language', 'domain',
               'place_discovery', 'place_composition', 'value', 'clean_value', 'lemma',
               'domain_original', 'domain_standard', 'domain_finegrained',
               'temporal_group', 'period', 'period_approx', 'corpus_source']

for col in string_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna('').astype(str)

# Keep numeric columns as numeric
if 'index_in_line' in combined.columns:
    combined['index_in_line'] = pd.to_numeric(combined['index_in_line'], errors='coerce').fillna(0).astype(int)

# ============================================================================
# STEP 7: Statistics and verification
# ============================================================================
g1_words = len(group1)
g1_texts = group1['fragment_id'].nunique()
g2_words = len(group2)
g2_texts = group2['fragment_id'].nunique()
g3_words = len(group3)
g3_texts = group3['fragment_id'].nunique()
total_words = len(combined)
total_texts = combined['fragment_id'].nunique()

print(f"\n" + "=" * 80)
print("UNIFIED DATASET SUMMARY")
print("=" * 80)
print(f"\nBreakdown by group:")
print(f"  - Group 1 (Old Babylonian):  {g1_words:>8,} words, {g1_texts:>5,} texts")
print(f"  - Group 2 (Neo-Assyrian):    {g2_words:>8,} words, {g2_texts:>5,} texts")
print(f"  - Group 3 (Late Babylonian): {g3_words:>8,} words, {g3_texts:>5,} texts")
print(f"  {'─' * 60}")
print(f"  - TOTAL:                     {total_words:>8,} words, {total_texts:>5,} texts")

print(f"\nBalance:")
g1_pct = g1_words / total_words * 100
g2_pct = g2_words / total_words * 100
g3_pct = g3_words / total_words * 100
print(f"  - Group 1: {g1_pct:5.1f}%")
print(f"  - Group 2: {g2_pct:5.1f}%")
print(f"  - Group 3: {g3_pct:5.1f}%")

print(f"\nColumns in unified dataset: {len(combined.columns)}")
print(f"  Original columns: fragment_id, fragment_line_num, index_in_line, word_language,")
print(f"                    domain, place_discovery, place_composition, value, clean_value, lemma")
print(f"  Added metadata:   temporal_group, period, period_approx, corpus_source,")
print(f"                    domain_original, domain_standard, domain_finegrained")

# ============================================================================
# STEP 8: Save to disk
# ============================================================================
# Save as CSV (always works)
output_csv = OUTPUT_DIR / 'unified_3groups_akkadian_letters.csv'
print(f"\n✓ Saving CSV version to: {output_csv}")
combined.to_csv(output_csv, index=False)

# Try to save as Parquet (requires pyarrow)
output_file = OUTPUT_DIR / 'unified_3groups_akkadian_letters.parquet'
try:
    print(f"✓ Saving parquet version to: {output_file}")
    combined.to_parquet(output_file, index=False, compression='snappy')
    parquet_saved = True
except ImportError:
    print(f"⚠️  Skipping parquet (install pyarrow: pip install pyarrow)")
    parquet_saved = False

print(f"\n" + "=" * 80)
print("✓ COMPLETE!")
print(f"  - CSV file: {output_csv} ({output_csv.stat().st_size / 1024 / 1024:.1f} MB)")
if parquet_saved:
    print(f"  - Parquet file: {output_file} ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")
print("=" * 80)