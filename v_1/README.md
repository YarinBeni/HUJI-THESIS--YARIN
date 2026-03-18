# v_1 — Akkadian Temporal Dating

Current working directory for the thesis. All active code and data live here.

## Folder Structure

```
v_1/
├── src/
│   ├── evaluation/       # Track A: LLM baseline pipeline (OpenRouter API)
│   ├── corpus/           # Full Akkadian corpus pipeline (download → unified)
│   ├── cluster/          # Schmidt Sciences cluster scripts
│   └── archive/          # Superseded MLM training code
├── data/
│   ├── raw/
│   │   ├── chungrong/        # Normalized source CSVs from Chungrong (archibab, oracc, lbl)
│   │   ├── cdli/             # CDLI catalogue + ATF dump
│   │   └── zip/              # Original downloaded archives (read-only)
│   ├── processed/
│   │   ├── ebl/              # Processed eBL corpus (one row per word)
│   │   ├── oracc/            # Processed ORACC corpus (one row per word)
│   │   └── archibab/         # Processed Archibab corpus (one row per word)
│   ├── unified/              # Merged corpus + train/val/test splits (one row per word)
│   ├── training_ready/       # Tokenized fragment parquets + vocab.json (model input)
│   ├── evaluation/           # Evaluation data (corpora + baseline results)
│   │   ├── corpora/          # Evaluation test sets (input to LLM pipeline)
│   │   │   ├── texts_for_evaluation.parquet / .jsonl   # 4,957 texts, text-level
│   │   │   └── unified_3groups_akkadian_letters.parquet # Word-level unified corpus
│   │   └── baselines/        # Baseline model outputs
│   │       ├── baseline_predictions.parquet             # Aggregated LLM predictions
│   │       ├── baseline_metrics.json                    # Accuracy / F1 results
│   │       └── cache/                                   # Per-model prediction caches
│   ├── analysis_outputs/     # Plots and JSON from analysis scripts
│   └── external/             # External reference data
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
python v_1/src/evaluation/03_llm_baseline.py --model gpt-oss-20b --dry-run
python v_1/src/evaluation/03_llm_baseline.py --model gpt-oss-20b
```

All commands run from repo root (`lititure-review/`).

## Key Data Files

| File | Description |
|------|-------------|
| `data/evaluation/corpora/texts_for_evaluation.jsonl` | 4,957 Akkadian texts ready for LLM evaluation |
| `data/raw/chungrong/*.csv` | Normalized source data from Chunrong Ni |
| `data/evaluation/baselines/baseline_predictions.parquet` | Aggregated predictions from all models run so far |

## Documentation

- Design decisions: `justification/` (repo root)
- Full project history: `RESEARCH_LOG.md` (repo root)
- Evaluation pipeline details: `src/evaluation/README.md`
