# CDLI-ORACC Metadata Matching: Recovering Period and Genre Information

**Date:** January 2026
**Purpose:** Document how we recovered period/genre metadata for ORACC texts by matching against the CDLI catalog.

---

## 1. The Problem

Our processed ORACC corpus contained **transliterated text** but was missing critical metadata:
- **Period** (e.g., Neo-Assyrian, Old Babylonian)
- **Genre** (e.g., Letter, Administrative, Legal)

This metadata is essential for creating controlled evaluation corpora (same genre, different time periods).

---

## 2. CDLI vs ORACC: Understanding the Relationship

### CDLI (Cuneiform Digital Library Initiative)
- **Role:** Master catalog of cuneiform artifacts worldwide
- **Content:** Metadata only (no text editions)
- **Identifier:** Assigns unique **P-numbers** (e.g., `P224485`) to every physical tablet
- **Metadata fields:** period, genre, provenience, museum, dimensions, etc.
- **Scale:** 353,283 records (as of January 2026)
- **Website:** https://cdli.earth

### ORACC (Open Richly Annotated Cuneiform Corpus)
- **Role:** Text editions with linguistic annotations
- **Content:** Transliterations, translations, lemmatization
- **Identifier:** Uses CDLI P-numbers to reference texts
- **Organization:** Project-based (SAA, RINAP, CAMS, etc.)
- **Website:** https://oracc.org

### The Relationship

```
CDLI (Catalog/Metadata)              ORACC (Text Editions)
┌─────────────────────────┐          ┌─────────────────────────┐
│ P224485                 │          │ P224485                 │
│ ├─ period: Neo-Assyrian │  ←────→  │ ├─ transliteration      │
│ ├─ genre: Letter        │  same    │ ├─ translation          │
│ ├─ provenience: Nineveh │   ID     │ ├─ lemmatization        │
│ ├─ museum: British Mus. │          │ └─ linguistic tags      │
│ └─ date: ca. 650 BC     │          │                         │
└─────────────────────────┘          └─────────────────────────┘
```

**Key insight:** CDLI and ORACC are complementary databases that share P-numbers as a common identifier. CDLI has the metadata; ORACC has the text content.

---

## 3. The Matching Process

### Step 1: Extract P-numbers from ORACC Corpus

Our ORACC corpus uses fragment IDs like `P224485`. We extracted the numeric portion:

```python
# Fragment ID in our data: "P224485"
# Extracted numeric ID:     224485
```

**Result:** 11,126 unique P-numbers in our ORACC corpus

### Step 2: Download CDLI Catalog

Source: https://github.com/cdli-gh/data (requires git-lfs)

File: `cdli_cat.csv` (154 MB, 353,283 records)

Key columns:
- `id_text`: Numeric P-number (e.g., 224485)
- `period`: Time period with dates (e.g., "Neo-Assyrian (ca. 911-612 BC)")
- `genre`: Text genre (e.g., "Letter", "Administrative")
- `provenience`: Archaeological findspot
- `language`: Primary language

### Step 3: Join on P-number

```python
# ORACC fragment_id: "P224485" → extract 224485
# CDLI id_text:       224485
# Match! → Retrieve period="Neo-Assyrian", genre="Letter", etc.
```

**Result:** 11,059 of 11,126 texts matched (99.4% match rate)

---

## 4. Matching Results

### Period Distribution (matched ORACC texts)

| Period | Count | Percentage |
|--------|-------|------------|
| Neo-Assyrian (ca. 911-612 BC) | 5,826 | 52.7% |
| Old Babylonian (ca. 1900-1600 BC) | 2,280 | 20.6% |
| Uruk III (ca. 3200-3000 BC) | 675 | 6.1% |
| Hellenistic (323-63 BC) | 639 | 5.8% |
| Middle Babylonian (ca. 1400-1100 BC) | 417 | 3.8% |
| Neo-Babylonian (ca. 626-539 BC) | 229 | 2.1% |
| Other periods | 993 | 9.0% |

### Millennium Classification

| Classification | Count | Percentage |
|----------------|-------|------------|
| 1st Millennium | 6,786 | 61.4% |
| 2nd Millennium | 2,867 | 25.9% |
| 3rd Millennium or earlier | 830 | 7.5% |
| Unknown/Other | 576 | 5.2% |

### Genre Distribution

| Genre | Count |
|-------|-------|
| Lexical | 3,734 |
| Letter | 2,434 |
| Legal | 896 |
| Administrative | 879 |
| Omen | 810 |
| School | 400 |
| Literary | 221 |
| Royal/Monumental | 149 |
| Other | ~500 |

---

## 5. Evaluation Corpus Extraction

### 1st Millennium Epistolary/Administrative Texts

For the embedding evaluation, we need same-genre texts from different periods:

| Filter | Count |
|--------|-------|
| Total 1st millennium texts | 6,786 |
| Epistolary (Letters) | 2,430 |
| Administrative | 1,345 |
| **Combined (Corpus B candidates)** | **3,775** |

### Final Evaluation Corpora

| Corpus | Source | Period | Genre | Texts |
|--------|--------|--------|-------|-------|
| **A** | ARCHIBAB | 2nd millennium (Old Babylonian) | Epistolary + Administrative | ~1,280 |
| **B** | ORACC (filtered) | 1st millennium (Neo-Assyrian/Neo-Babylonian) | Epistolary + Administrative | ~3,775 |

---

## 6. Output Files

| File | Description |
|------|-------------|
| `v_1/data/processed/oracc_cdli_metadata.parquet` | ORACC texts with matched CDLI metadata |
| `v_1/data/external/cdli_data/cdli_cat.csv` | Full CDLI catalog (353K records) |

---

## 7. Scripts Created

| Script | Purpose |
|--------|---------|
| `v_1/src/analysis/corpus_diagnostic.py` | Initial metadata analysis |
| `v_1/src/analysis/oracc_catalog_explorer.py` | ORACC project catalog exploration |
| `v_1/src/analysis/cdli_period_matcher.py` | CDLI P-number matching |

---

## 8. Summary for Thesis

> "To recover period and genre metadata for ORACC texts, we leveraged the complementary relationship between ORACC and CDLI. While ORACC provides rich text editions with transliterations and linguistic annotations, CDLI maintains a comprehensive catalog of cuneiform artifacts with detailed metadata. Both databases use P-numbers as unique identifiers, enabling a direct join.
>
> We matched 11,059 of our 11,126 ORACC texts (99.4%) against the CDLI catalog, recovering period and genre information. This revealed that 61% of our ORACC corpus dates to the 1st millennium BCE, with 3,775 texts classified as epistolary or administrative—the same genres represented in our 2nd millennium ARCHIBAB corpus. This enables a controlled evaluation comparing embeddings of same-genre texts across different historical periods."

---

## 9. References

- CDLI: https://cdli.earth / https://github.com/cdli-gh/data
- ORACC: https://oracc.org
- CDLI Metadata Fields: https://cdli.earth/docs/artifact-metadata-fields
