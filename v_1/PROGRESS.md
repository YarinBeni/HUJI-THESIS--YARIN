# Project Progress & Handover Snapshot

> **Status Date:** March 2026
> **Current Phase:** Track B complete on letters corpus. Awaiting full dataset from advisor to scale up.
> **Working Directory:** `v_1/`

## Project Context

Three-track thesis on mechanistic interpretability of LLMs applied to Akkadian cuneiform temporal dating:

- **Track A:** LLM baseline evaluation (OpenRouter API) — complete
- **Track B:** Linear probing of Qwen2.5-7B internal representations — complete on letters corpus
- **Track C:** Sparse Autoencoder (SAE) analysis — planned, awaiting full dataset

## Current Status (March 2026)

### Bias Check: Complete
TF-IDF classification on 4,957 letters: 98.3% (tier0), 91% (maximal cleaning). Signal is genuine diachronic linguistic change, not dataset bias. Full report: `data/evaluation/bias_check/`.

### Track B — Linear Probing: Complete (letters corpus)
Full pipeline on Qwen2.5-7B-Instruct, 4,957 letters, 29 layers, mean + last_token pooling, tier0 + maximal cleaning.

**Key results:**
| Condition | Pretrained | Random | Gap |
|-----------|-----------|--------|-----|
| Mean, tier0 (L4) | 99.1% | 98.3% | +0.8% |
| Mean, maximal (L3) | 96.3% | 90.7% | +5.5% |
| Last_token, tier0 (L28) | 95.5% | 84.8% | +10.7% |
| Last_token, maximal (L28) | 90.0% | 70.1% | +19.9% |

**Validity tests (all complete):**
- Learning curve: mean pooling hits 93% with just 42 texts (1%)
- PCA: top-5 PCs recover 90% accuracy (mean pooling) — very compact signal
- MLP vs linear: MLP < linear everywhere — genuinely linear encoding
- Random baseline: +20% pretraining gap for last_token/maximal (strongest selectivity)

Full log: `src/linear_probing/results/PIPELINE_RUN_LOG.md`

### Track C — SAE: Planned
Waiting for full dataset from advisor (~40k fragments across all genres). Will run same linear probe pipeline first, then SAE on best-layer activations.

## Blocked On
Full dataset delivery from advisor (40k+ Akkadian fragments, multi-genre, not just letters).

## Next Steps
1. Receive full dataset from advisor
2. Run linear probe on full dataset (reuse existing sbatch scripts)
3. Implement SAE on best-layer activations
4. Interpret SAE features in linguistic/historical context
