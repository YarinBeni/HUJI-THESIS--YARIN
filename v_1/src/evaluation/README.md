# LLM Baseline Evaluation Pipeline

Evaluates how well general-purpose LLMs can classify Akkadian letter texts by temporal period (Old Babylonian / Neo-Assyrian / Late Babylonian) using zero-shot prompting via the OpenRouter API.

---

## Quick Start (New Data)

If you received new/updated CSV files from Chunrong, here's how to rebuild everything:

```bash
# 1. Copy the new CSVs to the expected location
cp archibab_nor.csv       v_1/data/processed/from_chungrong/archibab_nor.csv
cp oracc_let_adm_nor.csv  v_1/data/processed/from_chungrong/oracc_let_adm_nor.csv
cp lbl_nor.csv            v_1/data/processed/from_chungrong/lbl_nor.csv

# 2. Build unified corpus (merges 3 groups, filters ORACC to letters only)
python v_1/src/preprocessing/06_create_test_letters_copra.py

# 3. Reconstruct texts + domain cleanup → ready for LLM evaluation
python v_1/src/evaluation/01_prepare_texts.py

# 4. (Optional) Clear old model cache if data changed significantly
rm v_1/data/evaluation_corpora/cache/*.jsonl

# 5. Set API key and run baseline
export OPENROUTER_API_KEY="your-key"
python v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b --dry-run  # check first
python v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b            # run
```

**All commands should be run from the repo root** (`lititure-review/`).

---

## Pipeline Flow

```
v_1/data/processed/from_chungrong/
├── archibab_nor.csv                    ← Source: Old Babylonian (~1800 BCE)
├── oracc_let_adm_nor.csv              ← Source: Neo-Assyrian (9-7 cent. BCE)
└── lbl_nor.csv                         ← Source: Late Babylonian (~600 BCE)
        │
        ▼
[06_create_test_letters_copra.py]       ← Step 1: Merge + filter to letters only
        │
        ▼
v_1/data/evaluation_corpora/
├── unified_3groups_akkadian_letters.parquet   ← Word-level unified corpus
├── unified_3groups_akkadian_letters.csv       ← Same, CSV format
        │
        ▼
[01_prepare_texts.py]                   ← Step 2: Reconstruct texts + domain cleanup
        │
        ▼
├── texts_for_evaluation.parquet        ← Text-level, cleaned (final corpus)
├── texts_for_evaluation.jsonl          ← Same data, JSONL for API processing
├── texts_token_stats.json              ← Token count estimates
        │
        ▼
[02_llm_baseline.py]                    ← Step 3: Call LLM API (per model)
        │
        ▼
├── cache/
│   ├── gpt-oss-20b.jsonl              ← Cached predictions (resume-safe)
│   ├── llama-4-maverick.jsonl
│   └── ...
        │
        ▼
[03_aggregate_results.py]               ← Step 4: Merge all model predictions
        │
        ▼
├── baseline_predictions.parquet        ← All predictions in one file
        │
        ▼
[04_evaluate_baseline.py]               ← Step 5: Compute metrics + report
        │
        ▼
├── baseline_results_report.md          ← Human-readable report
└── baseline_metrics.json               ← Machine-readable metrics
```

---

## File Details

### Source Data (`v_1/data/processed/from_chungrong/`)

Three CSV files prepared by Chunrong Ni. Each is word-level (one row per token) with columns:
- `fragment_id` — text identifier
- `fragment_line_num`, `index_in_line` — position within text
- `value` — raw transliteration
- `clean_value` — normalized transliteration (signs connected with `-`, subscript numbers as ASCII: `lu₂` → `lu2`)
- `lemma` — lemmatization
- `domain` — genre/type label
- `place_discovery`, `place_composition` — provenance

### Step 1: `v_1/src/preprocessing/06_create_test_letters_copra.py`

**Input:** 3 source CSVs from `v_1/data/processed/from_chungrong/`
**Output:** `v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet` (+ CSV)

What it does:
1. Loads all 3 CSVs
2. Merges variant domain labels in Archibab (e.g., `inconnu, lettre cassée` → `Unknown`)
3. **Filters ORACC to letters only** (`NALet` domain), removing administrative docs (`NAAdm`)
4. Adds metadata columns: `temporal_group`, `period`, `corpus_source`
5. Creates standardized domain columns: `domain_standard`, `domain_finegrained`
6. Saves combined word-level parquet + CSV

### Step 2: `v_1/src/evaluation/01_prepare_texts.py`

**Input:** `v_1/data/evaluation_corpora/unified_3groups_akkadian_letters.parquet`
**Output:** `texts_for_evaluation.parquet` + `.jsonl` + `texts_token_stats.json`

What it does:
1. Reads word-level parquet
2. Groups by `fragment_id`, reconstructs full text (words joined by spaces within each line, lines joined by newlines)
3. Extracts metadata per fragment (period, domain, place, etc.)
4. **Domain label cleanup** — removes texts with `Unknown`, `nan`, or `Other` domain labels (~29 texts, <1%)
5. Saves text-level parquet and JSONL with all fields

The JSONL uses `fragment_id` and `full_text` as field names (along with all metadata).

### Step 3: `v_1/src/evaluation/02_llm_baseline.py`

**Input:** `v_1/data/evaluation_corpora/texts_for_evaluation.jsonl`
**Output:** `v_1/data/evaluation_corpora/cache/<model-name>.jsonl`

Usage:
```bash
# Dry run — shows text count + estimated tokens, no API calls
python v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b --dry-run

# Run predictions
python v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b

# Run a different model
python v_1/src/evaluation/02_llm_baseline.py --model llama-4-maverick
```

Features:
- **Resume-safe**: each prediction is written to cache immediately. Safe to Ctrl+C and restart.
- Rate-limited (0.5s between calls by default)
- Skips already-cached fragment IDs on restart

### Steps 4-5: Aggregation + Evaluation

```bash
python v_1/src/evaluation/03_aggregate_results.py  # merge all model caches
python v_1/src/evaluation/04_evaluate_baseline.py   # compute accuracy, F1, confusion matrices
```

---

## Configuration (`v_1/src/evaluation/config.py`)

All paths, model registry, API settings, and the prompt template are defined here.

### Available Models

| Tier | Models | Cost |
|------|--------|------|
| Free | gpt-oss-20b, gpt-oss-120b, llama-4-maverick, llama-4-scout, gemini-2.5-pro-free, mistral-small-free, deepseek-v3-free, deepseek-r1-free | $0 |
| Open-source (paid) | qwen-2.5-7b/32b/72b, mixtral-8x7b, mistral-small/large, deepseek-chat/v3/r1 | Varies |
| Commercial | gemini-2.0-flash, gpt-4o, gpt-4o-mini, sonnet-3.5, grok-2 | Higher |

To add a new model, add its name and OpenRouter model ID to the appropriate dict in `config.py`.

---

## EDA Notebook

`v_1/notebooks/04_eda_evaluation.ipynb` — **read-only** EDA and verification.

- Corpus statistics, distributions, sample texts per period
- Domain label verification (sections 13-14) — checks the data is already clean
- Does NOT modify any data files — all data preparation is done by the pipeline scripts above

Run it after `01_prepare_texts.py` to inspect the corpus visually.

---

## Troubleshooting

**"Loaded X texts" count doesn't match expected:**
Re-run steps 1 and 2. Both parquet and JSONL are regenerated from scratch by `01_prepare_texts.py`.

**Cache from old data:**
If source CSVs changed, delete model cache files to avoid mixing old/new predictions:
```bash
rm v_1/data/evaluation_corpora/cache/*.jsonl
```

**KeyError on `fragment_id` or `full_text`:**
The JSONL must be generated by the current `01_prepare_texts.py` (which writes all fields). Re-run step 2.

---

## Data Cleaning Documentation

All cleaning decisions are documented in `justification/` (repo root):
- `chunrong_data_cleaning_decisions.md` — all rounds of data cleaning with Chunrong
- `evaluation_corpus_size_5005_texts.md` — why 4,957 texts (not 6,577)
- `domain_label_cleanup.md` — removing Unknown/nan domain labels
- `data_source_summary.md` — unified dataset composition
