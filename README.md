# Akkadian Temporal Dating — Thesis Research

**Research question:** How well can language models understand Akkadian cuneiform, and what can we learn about their internal representations?

## Current Status (Mar 2026)

| Phase | Status |
|-------|--------|
| Phase 0: Design decisions | COMPLETE |
| Phase 1: Baseline MLM (37M params) | COMPLETE (superseded — shifted to fine-tuning pre-trained LLMs) |
| Phase 1.5: Evaluation corpus (4,957 texts) | COMPLETE |
| Pre-Track A: Test data bias check | NEXT |
| Track A: LLM baseline pipeline | IN PROGRESS |
| Track B: Temporal representation geometry | PENDING |
| Track C: SAE interpretability | PENDING |

## Quick Links

- **Research plan** → `yarin/research_plan.md` (local only)
- **Project history & key results** → `RESEARCH_LOG.md`
- **Run the evaluation pipeline** → `v_1/src/evaluation/README.md`
- **Data & design decisions** → `justification/`

## Repo Structure

```
v_1/
├── src/
│   ├── evaluation/       # Track A pipeline (LLM baseline via OpenRouter)
│   ├── preprocessing/    # Corpus preparation scripts
│   ├── cluster/          # Schmidt Sciences cluster scripts
│   ├── analysis/         # Analysis scripts
│   └── training/         # Phase 1 MLM training (superseded)
├── data/
│   ├── evaluation_corpora/   # 4,957 texts, LLM predictions, metrics
│   └── processed/            # Chunrong's normalized CSVs + unified dataset
└── notebooks/                # EDA notebooks (01–04)
justification/                # Documented decisions for thesis/paper
RESEARCH_LOG.md               # Condensed milestone + results history
```
