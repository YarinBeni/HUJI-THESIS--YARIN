# Akkadian Temporal Dating — Thesis Research

> **See also:** [v_1/README.md](v_1/README.md) (working dir layout) · [v_1/PROGRESS.md](v_1/PROGRESS.md) (current status) · [PLAN_round2_qwen_diagnosis.md](PLAN_round2_qwen_diagnosis.md) (active research plan)

**Research question:** How well can language models understand Akkadian cuneiform, and what can we learn about their internal representations?

## Current Status (2026-05-17)

| Phase | Status |
|-------|--------|
| Phase 0: Design decisions | COMPLETE (Dec 2025) |
| Phase 1: Baseline MLM (37M params) | COMPLETE — superseded by fine-tuning pre-trained LLMs |
| Phase 1.5: Evaluation corpus (4,957 letters) | COMPLETE |
| Pre-Track A: Test data bias check (TF-IDF) | COMPLETE |
| Track A: LLM baseline pipeline | COMPLETE on letters · ORCC sweep planned |
| Track B: Linear probing of Qwen2.5-7B | ROUND 1 COMPLETE on letters + ORCC · ROUND 2 (Qwen-failure diagnosis) PLANNED |
| Track C: SAE interpretability | EXTRACTION IN PROGRESS (see `v_1/src/sae/`) |

For details: [v_1/PROGRESS.md](v_1/PROGRESS.md) (current state) · [v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md](v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md) (Track B numbers) · [PLAN_round2_qwen_diagnosis.md](PLAN_round2_qwen_diagnosis.md) (active plan).

## Quick Links

- **Research plan** → `yarin/research_plan.md` (local only)
- **Project history & key results** → `justification/research_log_phases_0_to_track_a.md` (frozen snapshot of Phase 0 → Track A)
- **Run the evaluation pipeline** → `v_1/src/evaluation/README.md`
- **Data & design decisions** → `justification/`

## Repo Structure

```
v_1/
├── src/
│   ├── evaluation/       # Track A pipeline (LLM baseline via OpenRouter)
│   ├── corpus/           # Full Akkadian corpus pipeline (download → unified)
│   ├── cluster/          # Schmidt Sciences cluster scripts
│   └── archive/          # Phase 1 MLM training code (superseded)
├── data/
│   ├── raw/              # Source data (eBL zips, CDLI dump, Chungrong CSVs)
│   ├── processed/        # Per-source parquets: ebl/, oracc/, archibab/
│   ├── unified/          # Merged corpus + train/val/test splits
│   ├── training_ready/   # Tokenized fragment parquets + vocab.json
│   ├── evaluation/       # corpora/ (test sets) + baselines/ (model outputs)
│   ├── analysis_outputs/ # Plots and JSON from analysis scripts
│   └── external/         # External reference data
└── notebooks/            # EDA notebooks (01–04)
justification/            # Documented decisions for thesis/paper (incl. research_log_phases_0_to_track_a.md)
```
