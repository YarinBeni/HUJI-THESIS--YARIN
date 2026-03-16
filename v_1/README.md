# v_1 — Akkadian Temporal Dating

Current working directory for the thesis. All active code and data live here.

## Folder Structure

```
v_1/
├── src/
│   ├── evaluation/       # Track A: LLM baseline pipeline (OpenRouter API)
│   ├── preprocessing/    # Corpus preparation (merge, filter, normalize)
│   ├── cluster/          # Schmidt Sciences cluster scripts
│   ├── analysis/         # Embedding + manifold analysis (Track B)
│   └── training/         # Phase 1 MLM training artifacts (superseded)
├── data/
│   ├── evaluation_corpora/   # Final evaluation corpus + LLM predictions
│   │   ├── texts_for_evaluation.parquet / .jsonl   # 4,957 texts, text-level
│   │   ├── unified_3groups_akkadian_letters.parquet # Word-level unified corpus
│   │   ├── baseline_predictions.parquet             # Aggregated LLM predictions
│   │   ├── baseline_metrics.json                    # Accuracy / F1 results
│   │   └── cache/                                   # Per-model prediction caches
│   ├── processed/
│   │   ├── from_chungrong/   # Normalized source CSVs (archibab, oracc, lbl)
│   │   └── unified/          # Full unified corpus (Phase 1 training data)
│   └── raw/                  # Original downloaded data (read-only)
└── notebooks/
    ├── 01_data_exploration.ipynb     # eBL + Archibab EDA
    ├── 02_unified_dataset_eda.ipynb  # Unified dataset EDA
    ├── 03_eda_corpora.ipynb          # Corpus comparison
    └── 04_eda_evaluation.ipynb       # Evaluation corpus verification
```

## Running the Evaluation Pipeline (Track A)

See `src/evaluation/README.md` for full instructions. Quick start:

```bash
export OPENROUTER_API_KEY="your-key"
python v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b --dry-run
python v_1/src/evaluation/02_llm_baseline.py --model gpt-oss-20b
```

All commands run from repo root (`lititure-review/`).

## Key Data Files

| File | Description |
|------|-------------|
| `data/evaluation_corpora/texts_for_evaluation.jsonl` | 4,957 Akkadian texts ready for LLM evaluation |
| `data/processed/from_chungrong/*.csv` | Normalized source data from Chunrong Ni |
| `data/evaluation_corpora/baseline_predictions.parquet` | Aggregated predictions from all models run so far |

## Documentation

- Design decisions: `justification/` (repo root)
- Full project history: `RESEARCH_LOG.md` (repo root)
- Evaluation pipeline details: `src/evaluation/README.md`
