# v_1/data — Data Directory Overview

This directory contains all data for the Akkadian interpretability thesis project.

## Pipeline Flow

```
raw/          →  processed/      →  unified/     →  training_ready/
(source files)   (per-source      (merged +        (tokenized,
                  corpora)         train/val/test)   model-ready)
                                       ↓
                                 evaluation/
                                 ├── corpora/     (LLM eval test sets)
                                 └── baselines/   (model predictions + metrics)
```

## Folder Summary

| Folder | Contents | Stage |
|--------|----------|-------|
| `raw/` | Original untouched source files from Shahar Spencer; Chungrong's evaluation CSVs; CDLI external catalog | Source |
| `processed/` | Per-source cleaned corpora (eBL, Archibab, ORACC) + ORACC-CDLI metadata join | Intermediate |
| `unified/` | Merged corpus (2.45M words, 40K texts) + train/val/test splits + EDA outputs | Intermediate |
| `raw/chungrong/` | 3 CSV files from Chungrong for evaluation (NOT pipeline outputs — externally provided) | Source / External |
| `training_ready/` | Tokenized fragments, vocabulary, and eval subset — ready for model training | Model-ready |
| `evaluation/corpora/` | Curated letter corpora for LLM evaluation (2 period groups) | Evaluation |
| `evaluation/baselines/` | Baseline LLM predictions, metrics, and model cache files | Evaluation outputs |
| `raw/cdli/` | CDLI catalog (cdli_cat.parquet + ATF file) — used to enrich ORACC metadata | External reference |
| `analysis_outputs/` | Diagnostic plots and reports from one-off analysis scripts | Outputs (not inputs) |

## Data Scale

- **Training corpus:** 2,450,094 words across 40,429 texts (eBL 41%, ORACC 57%, Archibab 3%)
- **Train/Val/Test split:** 32,343 / 4,042 / 4,044 texts
- **Vocabulary size:** 14,797 tokens
- **Evaluation corpora:** ~337K words across ~5,063 letter fragments (2 period groups)
