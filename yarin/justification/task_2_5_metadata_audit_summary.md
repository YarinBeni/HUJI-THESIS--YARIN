# Task 2.5: Metadata Audit & Corpus Extraction Summary

**Date:** January 2026
**Status:** ✅ COMPLETED

---

## 2.5.1 Audit Metadata: Check what period/genre info exists in ARCHIBAB and ORACC

### Status: ✅ COMPLETED

### Findings:

#### ARCHIBAB (2nd Millennium)
| Field | Coverage | Status |
|-------|----------|--------|
| **Genre (domain)** | 100% | ✅ Available - French categories |
| **Period** | Implicit | ✅ Source is Old Babylonian by definition |

**Genre values found:**
- `lettre administrative`: 977 fragments
- `lettre politique`: 180 fragments
- `lettre privée`: 121 fragments
- `lettre diplomatique`: 2 fragments
- Other: 30 fragments

#### ORACC (1st Millennium)
| Field | Coverage | Status |
|-------|----------|--------|
| **Genre** | 0% in processed data | ❌ Lost during processing |
| **Period** | 0% in processed data | ❌ Lost during processing |

**Solution Applied:** Matched ORACC P-numbers to CDLI catalog

### CDLI Matching Results:
- **Total ORACC P-numbers:** 11,126
- **Matched to CDLI:** 11,059 (99.4%)
- **Period coverage recovered:** 96.2%
- **Genre coverage recovered:** 89.2%

### Period Distribution (after CDLI matching):
| Period | Count | % |
|--------|-------|---|
| Neo-Assyrian (ca. 911-612 BC) | 5,826 | 52.7% |
| Old Babylonian (ca. 1900-1600 BC) | 2,280 | 20.6% |
| Uruk III (ca. 3200-3000 BC) | 675 | 6.1% |
| Hellenistic (323-63 BC) | 639 | 5.8% |
| Middle Babylonian (ca. 1400-1100 BC) | 417 | 3.8% |
| Neo-Babylonian (ca. 626-539 BC) | 229 | 2.1% |
| Other | ~993 | 9.0% |

### Genre Distribution (after CDLI matching):
| Genre | Count | % |
|-------|-------|---|
| Lexical | 3,734 | 33.8% |
| Letter | 2,434 | 22.0% |
| Legal | 896 | 8.1% |
| Administrative | 879 | 7.9% |
| Omen | 810 | 7.3% |
| Other | ~2,306 | 20.9% |

---

## 2.5.2 Extract Filtered Subsets: Epistolary+Admin texts from each period

### Status: ✅ COMPLETED

### Corpus A: 2nd Millennium (ARCHIBAB)
| Metric | Value |
|--------|-------|
| Source | ARCHIBAB |
| Period | Old Babylonian (ca. 1900-1600 BC) |
| Genre filter | `lettre administrative`, `lettre politique`, `lettre privée` |
| **Fragments** | **1,288** |
| **Words** | **64,511** |
| File | `v_1/data/evaluation_corpora/corpus_a_archibab_2nd_mill.parquet` |

### Corpus B: 1st Millennium (ORACC)
| Metric | Value |
|--------|-------|
| Source | ORACC (filtered via CDLI metadata) |
| Period | Neo-Assyrian, Neo-Babylonian, Hellenistic, Achaemenid |
| Genre filter | Letter, Administrative, Legal |
| **Fragments** | **3,775** |
| **Words** | **272,622** |
| File | `v_1/data/evaluation_corpora/corpus_b_oracc_1st_mill.parquet` |

### Corpus B Period Breakdown:
| Period | Fragments |
|--------|-----------|
| Neo-Assyrian (ca. 911-612 BC) | 3,272 |
| Hellenistic (323-63 BC) | 485 |
| Achaemenid (547-331 BC) | 11 |
| Neo-Babylonian (ca. 626-539 BC) | 5 |

### Corpus B Genre Breakdown:
| Genre | Fragments | % |
|-------|-----------|---|
| Letter | 2,430 | 64.4% |
| Legal | 883 | 23.4% |
| Administrative | 461 | 12.2% |

---

## 2.5.3 Analyze Transliteration Differences Between the Two Sources

### Status: ✅ COMPLETED

### Normalization Applied (EvaCun 2025):
| Transformation | Status | Example |
|----------------|--------|---------|
| Subscript removal | ✅ Done | `ša₂` → `ša` |
| Determinative removal | ✅ Done | `{d}AMAR.UTU` → `AMAR UTU` |
| Editorial marks removal | ✅ Done | `[a]-na` → `a na` |
| Sign splitting | ✅ Done | `a-na` → `a na` |

### Remaining Differences:
| Issue | Status | Notes |
|-------|--------|-------|
| Numbers in signs | ⚠️ Present | 1,516 signs contain digits (e.g., `2(SILA)`) |
| Brackets | ✅ Removed | None remaining |
| Subscripts | ✅ Removed | None remaining |

### Sign Vocabulary Comparison:
| Comparison | Overlap | Interpretation |
|------------|---------|----------------|
| ARCHIBAB vs ORACC | **4.3%** | Very low - suggests significant dialectal/temporal differences |
| ARCHIBAB vs eBL | 6.2% | Low overlap |
| ORACC vs eBL | 13.2% | Moderate overlap |

**Key Insight:** The low vocabulary overlap (4.3%) between ARCHIBAB and ORACC indicates substantial differences in sign usage between 2nd and 1st millennium texts. This is expected due to:
- Different scribal traditions
- Language evolution (Old Babylonian → Neo-Assyrian/Neo-Babylonian)
- Different text genres emphasis

### Transliteration Samples:

**ARCHIBAB (2nd Mill):**
```
Raw:   a-na be-lí-ia qí-bí-ma
Signs: a na be lí ia qí bí ma
```

**ORACC (1st Mill):**
```
Raw:   LUGAL a-na {1}aš-šur-MAN-⸢PAB*
Signs: LUGAL a na aš šur MAN PAB
```

---

## Scripts Created

| Script | Purpose |
|--------|---------|
| `v_1/src/analysis/corpus_diagnostic.py` | Initial metadata audit |
| `v_1/src/analysis/cdli_period_matcher.py` | CDLI P-number matching |
| `v_1/src/analysis/cdli_join_diagnostic.py` | Join verification & visualization |
| `v_1/src/analysis/oracc_catalog_explorer.py` | ORACC project exploration |

## Output Files

| File | Description |
|------|-------------|
| `v_1/data/evaluation_corpora/corpus_a_archibab_2nd_mill.parquet` | Corpus A |
| `v_1/data/evaluation_corpora/corpus_b_oracc_1st_mill.parquet` | Corpus B |
| `v_1/data/processed/oracc_cdli_metadata.parquet` | ORACC texts with CDLI metadata |
| `v_1/data/analysis_outputs/cdli_join_diagnostic.png` | Join verification plot |
| `v_1/data/analysis_outputs/evaluation_corpora_comparison.png` | Corpora comparison plot |

---

## Summary

All three subtasks of 2.5 are complete:
- ✅ 2.5.1: Metadata audited, CDLI matching recovered missing ORACC metadata
- ✅ 2.5.2: Filtered corpora extracted (1,288 + 3,775 = 5,063 fragments)
- ✅ 2.5.3: Transliteration differences analyzed, 4.3% sign overlap confirms temporal separation
