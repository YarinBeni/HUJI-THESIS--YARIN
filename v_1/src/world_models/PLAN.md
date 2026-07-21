# W — Replicating "Language Models Represent Space and Time" (Gurnee & Tegmark 2023) on our model ladder

> Section prefix: **W** (world-models). Status 2026-07-21: designed + implemented, ready to submit.
> Reference repo: https://github.com/wesg52/world-models (arXiv:2310.02207).

## 1. Why we are doing this

The thesis story probes chronology/geography out of embeddings of **Akkadian** texts and
finds the 400M translation encoder beats the LLM ladder. G&T ran the *mirror* experiment on
**English** entities (six datasets, linear probes over layers) and found LLMs linearly encode
space and time. Running their experiments with **our** ladder gives us three things:

1. **Pipeline validation** — our models + our probing machinery reproduce a published result
   on high-resource data ("we will know it works"). The trained Llama-2 runs are the anchor:
   if our re-run of Llama-2-70B lands near the paper's numbers, the harness is trusted.
2. **The random-init control the paper never ran** — G&T compare trained models only. We add
   random-weights versions of *every causal model including each Llama-2 size*. If random
   Llama-2-70B shows nontrivial "space/time decodability", part of the paper's signal is
   architecture/tokenization prior, not learned world structure — exactly the argument our
   thesis makes with the `random` arm on Akkadian.
3. **Cross-lingual context for the thesis** — where do AKK-300m / cunei-400m / uMT5 (small
   translation-tuned encoders) land on *English* world knowledge? Expected ≈ TF-IDF-level;
   this cleanly separates "small translation model knows Akkadian time" from "small
   translation model is generically good at probing tasks".

## 2. Model ladder (14 extraction arms + TF-IDF)

| method | HF id | arch | layers×d | random | GPU |
|---|---|---|---|---|---|
| `qwen3_1b7` | Qwen/Qwen3-1.7B | causal | 28×2048 | | 1 |
| `qwen3_8b` | Qwen/Qwen3-8B | causal | 36×4096 | | 1 |
| `qwen3_32b` | Qwen/Qwen3-32B | causal | 64×5120 (stride 2) | | 1 |
| `gpt_oss_120b` | openai/gpt-oss-120b | causal | 36×2880 | | 8 (J12b lesson) |
| `thalesian_akk300m` | Thalesian/AKK_300m | encoder | enc layers×d_enc | | 1 |
| `thalesian_cunei400m` | Thalesian/cuneiformBase-400m | encoder | enc layers×d_enc | | 1 |
| `umt5_base` | google/umt5-base | encoder | 12×768 | | 1 |
| `random` | Qwen/Qwen3-8B from_config | causal | 36×4096 | seed 42 | 1 |
| `llama2_7b` | meta-llama/Llama-2-7b-hf | causal | 32×4096 | | 1 |
| `llama2_13b` | meta-llama/Llama-2-13b-hf | causal | 40×5120 | | 1 |
| `llama2_70b` | meta-llama/Llama-2-70b-hf | causal | 80×8192 (stride 2) | | 4 |
| `llama2_7b_random` | 7b config | causal | 32×4096 | seed 42 | 1 |
| `llama2_13b_random` | 13b config | causal | 40×5120 | seed 42 | 1 |
| `llama2_70b_random` | 70b config (built by W0) | causal | 80×8192 (stride 2) | seed 42 | 4 |
| `tfidf` | — | baseline | word 1–2 grams + char_wb 2–5 grams | | CPU |

* `random` keeps the thesis-ladder name (random-init Qwen3-8B, seed 42, same convention as
  `J12c`/`01b_extract_random_baseline`).
* meta-llama repos are gated; the loader falls back to the ungated `NousResearch/Llama-2-*-hf`
  mirrors automatically (same weights). Override with `WM_LLAMA_ORG`.
* `llama2_70b_random` cannot be `from_config`-ed inside a GPU job (137 GB CPU-RAM spike +
  device_map juggling), so **W0** materializes it once (CPU, seed 42, `save_pretrained` with
  safetensors shards) into `WM_MODELS_DIR` and extraction loads it like a normal checkpoint
  with `device_map=auto`.

## 3. Datasets (theirs, verbatim)

The six entity CSVs are vendored under `data/entity_datasets/` (23 MB, copied from the
world-models repo at commit HEAD 2026-07-21; `fetch_data.py` re-downloads + verifies counts).
We use **their `is_test` column** as the split so numbers are directly comparable to the paper.

| entity_type | n | target | feature |
|---|---|---|---|
| `world_place` | 39,585 | (lon, lat) | `coords` |
| `us_place` | 29,997 | (lon, lat) | `coords` |
| `nyc_place` | 19,838 | (lon, lat) | `coords` |
| `historical_figure` | 37,539 | death year | `death_year` |
| `art` | 31,321 | release date → fractional year | `release_date` |
| `headline` | 28,461 | publication date → fractional year | `pub_date` |

Entity strings are built **exactly as in their `feature_datasets/`** (ported into
`wm_lib/entity_data.py`):
world_place: parenthetical `"X (Country)" → "Country's X"`, comma `"A, B" → "B's A"`;
nyc_place: their capitalization normalizer (stop-words lower, NYC abbreviations kept);
art: `"{creator}'s {title}"` (apostrophe rule); headline: full headline incl. final period;
us_place / historical_figure: raw `name`.

Prompt = **`empty`** (the paper's canonical setting; their prompt ablation showed prompts
barely matter). The code keeps a prompt registry so `random`-prefix / question prompts can be
added later without touching extraction.

## 4. Extraction protocol

Adapted from `stress_tests/shared/extract_lib.py` (proven loaders: causal via `model.model`
skipping the LM head, encoder via `get_encoder()`, umt5 config patch, download retries) +
their `save_activations.py`.

* Tokenize per model: `add_special_tokens=False`, prepend BOS iff the tokenizer has one
  (Llama yes; Qwen/gpt-oss no; encoders no), cap 96 tokens (headlines ≈ 30), right-pad per
  batch, length-sorted batches, original order restored on save.
* Forward through the base transformer with `output_hidden_states=True`; skip the embedding
  layer (`hidden_states[1:]`), keep every `stride`-th layer (stride 2 for qwen3_32b and
  llama2_70b*, else 1).
* Pooling sites over **entity tokens only** (non-BOS, non-pad):
  `last` (paper-faithful, causal canonical) and `mean` (thesis canonical, encoder canonical).
  Causal models save `last` by default (`--sites last,mean` to add mean); encoders save both.
* Output: `activations/{method}/{entity_type}/{site}.layer{L}.npz` (fp16, row order = CSV
  order) + committed `metadata.json` (n, d, layers, hfid, truncation count, runtime).
  `*.npz` is already globally gitignored → cluster-local, same convention as J12.

**Disk budget** (fp16, `last`, all 6 datasets = 186.7k vectors/layer): 1.7B 21 GB · 8B 55 GB ·
32B(s2) 61 GB · gpt-oss 39 GB · L7B 49 GB · L13B 77 GB · L70B(s2) 122 GB · randoms mirror
their trained twins · encoders < 10 GB total. Worst-case everything at once ≈ 730 GB —
therefore **W2 probes support `--cleanup`** (delete a method's npz after its probe results
JSON lands) and the README's submission order interleaves extract→probe for the big arms.

## 5. Probing protocol (paper-faithful)

Ported from their `probe_experiment.py` + `probes/evaluation.py` into `wm_lib/probing.py`:

* Train on `~is_test`, evaluate on `is_test` (their split). Targets z-scored on train,
  predictions un-normalized before scoring.
* **RidgeCV** over `np.logspace(-1, 5, 13)` (superset of their per-model ranges; GCV, one fit
  per layer). Places: multi-output ridge on (lon, lat).
* Metrics per layer, train+test: R² (joint + per-axis), MAE, Pearson/Spearman/Kendall, and for
  places haversine MAE/RMSE/R² (lat–lon order corrected when calling haversine) + their
  proximity error.
* Optional `--probe pls` (k=3–20 sweep) to connect to the thesis-canonical PLS pipeline —
  secondary, off by default.
* Output: `results/probes/{method}/{entity_type}.{site}.ridge.json` — per-layer score dict +
  `best_layer` (by test R²) — plus the hero-layer projection CSV (`projections/`) for map/
  timeline figures, and the hero-layer probe direction (npz) for later neuron work.
* `tfidf_baseline.py`: same split/targets/metrics; sparse Ridge (`sparse_cg`) with a manual
  alpha sweep on a 10% validation carve-out of train (RidgeCV's GCV can't do sparse).

## 6. Aggregation & the money table

`aggregate_results.py` builds `results/summary_best_layer.csv` (+ `RESULTS.md` markdown):
rows = 15 arms, cols = 6 datasets, cells = best-layer **test R²** (and a twin Spearman table),
with a **paper-reference row** (`paper_reference.json`: Llama-2-70B ≈ world .911 / us .864 /
nyc .359 / historical .835 / art .885 / headline .746) and layerwise curve plots
(`plot_results.py` → `results/figs/`, the Figure-2 analog with random arms dashed).

**First real numbers (full-data TF-IDF floor, run 2026-07-21 during development):**
world .642 · us .536 · nyc .389 · historical .645 · art .116 · headline .448 — i.e.
surface form alone recovers most of the ordering on several datasets, and on
nyc_place the TF-IDF floor already *exceeds* the paper's Llama-2-70B probe (.359).
The paper has no such control; whatever the trained-vs-random gap turns out to be,
the embedding arms must be read against this floor, not against 0.

Read of the outcome:
* `llama2_70b` ≈ paper row → harness validated.
* `llama2_*_random` / `random` vs trained gap = how much of "space & time" is *learned*;
  random ≈ tfidf ≈ low is the expected/clean outcome, random ≫ tfidf would be the
  interesting finding (surface-form prior in the architecture).
* Qwen3 ladder vs Llama-2 ladder = 2023-open-weights vs 2025-open-weights world models.
* Encoders (akk300m/cunei400m/umt5) ≈ tfidf on English = the cross-lingual control.

## 7. Job graph (sbatch, W prefix — conventions: voltagepark, conda `thesis`, `_common.sh` sync/commit)

```
W0_build_random_llama70b   CPU mem 200G      from_config seed42 → save_pretrained (once)
W1_extract                 gpu:1  array 0-6  qwen3_1b7 · qwen3_8b · qwen3_32b · akk300m ·
                                             cunei400m · umt5_base · random   (× 6 datasets inner loop)
W1b_extract_gptoss         gpu:8             gpt_oss_120b
W1c_extract_llama          gpu:1  array 0-3  llama2_7b · llama2_13b · llama2_7b_random · llama2_13b_random
W1d_extract_llama70b       gpu:4  array 0-1  llama2_70b · llama2_70b_random   (task 1 after W0)
W2_probe                   CPU    array 0-13 ridge probes, all datasets × sites  (--cleanup for 32B/70B arms)
W2b_tfidf                  CPU               TF-IDF baseline, all datasets
W3_aggregate               CPU               summary tables + figs, commit_push
```

Committed artifacts: metadata.json per extraction, probe JSONs, summary CSVs/MD, figs.
Cluster-local: npz activations (gitignored), random-70B checkpoint.

## 8. Explicit non-goals of round 1 (future W-jobs)

Their generalization block-holdouts (held-out country/century), PCA-dimension sweep,
nonlinear MLP probe, prompt ablations, and neuron search/interventions are all supported by
the stored probe directions + framework but **not** in this round. Round 1 answers: layerwise
linear decodability, trained vs random, our ladder vs Llama-2, vs TF-IDF floor.

## 9. Risks / gotchas

* **Llama gating** — auto-fallback to NousResearch mirrors; set `HF_TOKEN` if using meta-llama.
* **Disk** — interleave probe `--cleanup` after each big extraction (README order). 
* **gpt-oss-120b** — bf16 ≈ 240 GB, gpu:8 (J12b OOM'd on gpu:4); MoE hidden 2880.
* **Qwen has no BOS** — handled (BOS only when the tokenizer defines one, mask logic generic).
* **art `is_test` is held-out-by-creator** (not random) — matches paper, remember when reading.
* **headline pub_date is tz-aware** — parsed `utc=True`, fractional-year via their NS_PER_YEAR.
* **NaN targets** (some art release dates) — dropped per-feature with count logged in JSON.
