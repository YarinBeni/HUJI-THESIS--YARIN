# v_1 — Akkadian Temporal Dating

> **See also:** [PROGRESS.md](PROGRESS.md) (current status) · [src/linear_probing/results/PIPELINE_RUN_LOG.md](src/linear_probing/results/PIPELINE_RUN_LOG.md) (Track B results) · [../justification/research_log_phases_0_to_track_a.md](../justification/research_log_phases_0_to_track_a.md) (frozen Phase 0 → Track A history)

Current working directory for the thesis. All active code and data live here.

## Folder Structure

```
v_1/
├── src/
│   ├── bias_check/       # Pre-Track-A: TF-IDF bias validation (letters corpus)
│   ├── linear_probing/   # Track B: Qwen2.5-7B layer probing + validity tests
│   ├── sae/              # Track C: SAE feature analysis (Arditi 131k SAE)
│   ├── evaluation/       # Track A: LLM baseline pipeline (OpenRouter API)
│   ├── corpus/           # Full Akkadian corpus pipeline (download → unified)
│   ├── viz/              # SEAL embedding explorer GUI (seal_eda.html + data pipeline)
│   ├── cluster/          # Schmidt Sciences HPC cluster setup + README
│   └── archive/          # Superseded MLM training code (Akkadian MLM + training scripts)
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
| `data/raw/chungrong/*.csv` | Normalized letter CSVs from Chunrong Ni |
| `data/raw/chungrong/seal_round4/{seal,dll,lbpl}.csv` | Round-4 SEAL/DLL/LBPL word-level CSVs (384 frags / 40,484 words) |
| `data/raw/chungrong/seal_round4/inspection_report.json` | Phase 0 data contract — MD5 hashes + per-task feasibility |
| `data/evaluation/corpora/seal_corpus.parquet` | 384-fragment SEAL corpus (Phase A) — `text`, `text_tier0`, `text_maximal` + metadata |
| `data/evaluation/corpora/seal_tasks_verification.md` | Phase B self-test — confirmed per-task N/classes/k |
| `data/evaluation/baselines/baseline_predictions.parquet` | Aggregated LLM predictions |
| `src/linear_probing/results/letters__probe_cls__period/probe_results_qwen2.5-7b-instruct.json` | Linear probe results (pretrained) |
| `src/linear_probing/results/letters__probe_cls__period/validity_results_*.json` | Validity experiment results |

## SEAL Pipeline (all phases complete as of 2026-04-26)

384 fragments from SEAL/DLL/LBPL corpora. Full plan: `justification/parallel_plans_final.md`.

### Bias check + corpus build
| Phase | Script | Status |
|-------|--------|--------|
| 0 — Inspect raw CSVs | `src/corpus/01_inspect_seal_data.py` | ✅ done |
| A — Corpus build | `src/corpus/02_build_seal_corpus.py` | ✅ done |
| B — Task registry | `src/bias_check/seal_tasks.py` | ✅ done |
| C — Bias check CV | `src/bias_check/06_bias_check_cv.py` | ✅ done (all 12 runs FAIL p=0.001) |

### Embedding Explorer (EDA GUI)
| Step | Script / Job | Status |
|------|-------------|--------|
| TF-IDF coords (local) | `src/viz/01_compute_tfidf_coords.py` | ✅ done |
| Qwen + Random mean extraction (cluster) | jobs 2994–2997, then job 3029 | ✅ done |
| MLM retrain (cluster H100) | `sbatch/seal/train_mlm.sh` (job 2998) | ✅ done — val_loss=2.9777 |
| MLM extraction (cluster) | `sbatch/seal/extract_mlm_embeddings.sh` (job 3028) | ✅ done |
| SEAL last-token extraction (cluster) | jobs 4887–4890 | ✅ done |
| SEAL last-token + UMAP coords (cluster) | `sbatch/seal/compute_umap_coords_last.sh` (job 4906) | ✅ done |
| ORCC corpus build (local) | `src/corpus/03_build_orcc_corpus.py` | ✅ done — 1202 frags |
| ORCC extraction (cluster) | jobs 4900–4903 | ✅ done |
| ORCC mean+last coords (cluster) | `sbatch/orcc/compute_2d_umap_coords.sh` (job 4908) | ✅ done |
| Merge all coords (local) | `src/viz/02_merge_coords.py` | ✅ done — 710 keys, 44MB, 1586 frags |
| Standalone HTML (local) | `src/viz/03_build_standalone_html.py` | ✅ done — 46MB |
| GUI | `src/viz/seal_eda.html` | ✅ live — see `src/viz/README.md` |

## Documentation

- Design decisions: `justification/` (repo root)
- Phase 0 → Track A history: `../justification/research_log_phases_0_to_track_a.md` (frozen snapshot)
- Track B / ORCC / Round 2 history: `src/linear_probing/results/PIPELINE_RUN_LOG.md`
- Progress snapshot: `PROGRESS.md`
- SEAL pipeline plan + verified facts: `justification/seal_round4_pipeline_plan.md`
- Evaluation pipeline: `src/evaluation/README.md`
- Bias check: `src/bias_check/README.md`
- Linear probing run log: `src/linear_probing/results/PIPELINE_RUN_LOG.md`
