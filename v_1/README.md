# v_1 — Akkadian Temporal Dating

Current working directory for the thesis. All active code and data live here.

## Folder Structure

```
v_1/
├── src/
│   ├── bias_check/       # Pre-Track-A: TF-IDF bias validation (letters corpus)
│   ├── linear_probing/   # Track B: Qwen2.5-7B layer probing + validity tests
│   ├── evaluation/       # Track A: LLM baseline pipeline (OpenRouter API)
│   ├── corpus/           # Full Akkadian corpus pipeline (download → unified)
│   ├── cluster/          # Schmidt Sciences HPC cluster setup + README
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
│   ├── evaluation/           # Evaluation data (corpora + baseline + bias check results)
│   │   ├── corpora/          # Evaluation test sets (input to LLM pipeline)
│   │   │   ├── texts_for_evaluation.parquet / .jsonl   # 4,957 letters, text-level
│   │   │   └── unified_3groups_akkadian_letters.parquet # Word-level unified corpus
│   │   ├── baselines/        # LLM baseline outputs (Track A)
│   │   │   ├── baseline_predictions.parquet
│   │   │   ├── baseline_metrics.json
│   │   │   └── cache/
│   │   └── bias_check/       # TF-IDF bias analysis results + plots + report
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

## Running the Linear Probing Pipeline (Track B)

```bash
# Phase 1 — Extract activations (GPU, ~2 hrs)
sbatch v_1/src/linear_probing/sbatch/01_extract.sh

# Phase 1b — Extract random-weights baseline (GPU, ~30 min)
sbatch v_1/src/linear_probing/sbatch/01b_extract_random.sh

# Phase 2 — Run probe (CPU, ~1 hr)
JOB1=$(sbatch --parsable v_1/src/linear_probing/sbatch/02_probe.sh)

# Phase 2b — Validity tests (CPU, ~30 min, after Phase 1b)
sbatch --dependency=afterok:$JOB1 v_1/src/linear_probing/sbatch/02b_validity.sh
```

Full run log: `src/linear_probing/results/PIPELINE_RUN_LOG.md`

## Key Data Files

| File | Description |
|------|-------------|
| `data/evaluation/corpora/texts_for_evaluation.jsonl` | 4,957 Akkadian letters for evaluation |
| `data/raw/chungrong/*.csv` | Normalized source data from Chunrong Ni |
| `data/evaluation/baselines/baseline_predictions.parquet` | Aggregated LLM predictions |
| `src/linear_probing/results/probe_results_qwen2.5-7b-instruct.json` | Linear probe results (pretrained) |
| `src/linear_probing/results/validity_results_*.json` | Validity experiment results |

## Documentation

- Design decisions: `justification/` (repo root)
- Full project history: `RESEARCH_LOG.md` (repo root)
- Progress snapshot: `PROGRESS.md`
- Evaluation pipeline: `src/evaluation/README.md`
- Bias check: `src/bias_check/README.md`
- Linear probing run log: `src/linear_probing/results/PIPELINE_RUN_LOG.md`
