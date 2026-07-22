# HANDOFF — world_models section (Gurnee & Tegmark replication)

> Written for a fresh context agent picking this up. Read `PLAN.md` for the full
> design rationale and `README.md` for the operator's run order. This file is the
> "what we actually did, where it stands, and what bit us" summary.

## 1. What this section is

A faithful re-implementation of **Gurnee & Tegmark 2023, "Language Models Represent
Space and Time"** (arXiv:2310.02207, repo `wesg52/world-models`) run on the thesis's
own model ladder, plus the **random-init control the paper never ran**. Linear probes
read a place's coordinates or an event's year out of a model's hidden states, per
layer, on held-out entities. Goal (three-fold):

1. **Validate the harness** — reproduce the paper's numbers on trained Llama-2 so we
   trust the probing pipeline.
2. **Add the random-init control** — trained-vs-random tells us how much of "space &
   time decodability" is *learned* vs an architecture/tokenizer prior.
3. **Cross-lingual context for the thesis** — where do our small translation encoders
   (AKK-300M, cunei-400M, uMT5) land on *English* world-knowledge? (Answer: near the
   floor — so their Akkadian strength is language-specific, not generic probing skill.)

This is the **English / high-resource** mirror of the thesis's own Akkadian P1 (year)
and P2 (geography) probes. It does NOT touch Akkadian data.

## 2. Status (update this as it changes)

- **12 of 14 arms complete and committed.** Pending: `llama2_70b` and `gpt_oss_120b`.
  - `llama2_70b` / `llama2_70b_random`: extraction done; the **d=8192 ridge probes are
    slow** (many hours) and were still running at last check.
  - `gpt_oss_120b`: **extraction done** (metadata committed) but its **probe never
    committed** — the big all-arms probe (`W2` array task 3) ran before the acts landed
    or crashed, and silently skipped. Re-kick with:
    `sbatch --array=3 v_1/src/world_models/sbatch/W2_probe.sbatch` then a `W3`.
- **No MLM arm** here (it was a thesis-stress-test arm, never in this ladder's 9-model
  scope). Add only if asked; it would be a mean-pooled encoder-style arm.
- Results committed under `results/probes/<arm>/`, summary tables in
  `results/summary_best_layer_{r2,spearman}.csv` + `results/RESULTS.md`.

## 3. The ladder (14 arms + TF-IDF)

Registry: `wm_lib/registry.py`. `random`/`*_random` = from-config seed 42.

| method | HF id | arch | notes |
|---|---|---|---|
| qwen3_1b7 / 8b / 32b | Qwen/Qwen3-{1.7B,8B,32B} | causal | 32b stride 2 |
| gpt_oss_120b | openai/gpt-oss-120b | causal | gpu:8 |
| thalesian_akk300m / cunei400m | Thalesian/{AKK_300m,cuneiformBase-400m} | encoder | mean-pool |
| umt5_base | google/umt5-base | encoder | mean-pool |
| random | Qwen/Qwen3-8B from-config | causal | seed 42 control |
| llama2_7b / 13b / 70b | NousResearch/Llama-2-*-hf | causal | 70b stride 2, gpu:4 |
| llama2_{7b,13b,70b}_random | from-config seed 42 | causal | the paper's missing control |
| tfidf | — | baseline | char+word n-grams, ridge |

Datasets (theirs, vendored in `data/entity_datasets/`, verified by `fetch_data.py`):
`world_place, us_place, nyc_place` (coords) · `historical_figure, art, headline` (year).

## 4. Protocol (what "faithful" means here)

- **Entity strings** built exactly as their `feature_datasets/*` (ports in
  `wm_lib/entity_data.py`): world-place paren/comma → "Country's X", nyc capitalization
  normalizer, art "creator's title", headline verbatim, us/figure raw name.
- **Prompt** = `empty` (their canonical setting).
- **Pooling**: last entity token for causal models (paper-faithful), mean for encoders
  (they only used decoders; encoders have no causal last-token summary — a necessary
  deviation, noted).
- **Probe**: per-layer `RidgeCV` on `~is_test`, eval on `is_test` (their split). Targets
  z-scored on train. Places = multi-output ridge on (lon, lat).
- **Metrics**: R² (joint; haversine for places) + Spearman (time; places = mean of lat
  & lon rank corr). Both live for both domains.

## 5. File map

- `wm_lib/registry.py` — arms, HF ids, pooling sites, tokenizer overrides.
- `wm_lib/entity_data.py` — dataset loaders, entity-string builders, probe targets.
- `wm_lib/tokenize_lib.py` — generic prompt→ids + entity masks for any tokenizer.
- `wm_lib/extract.py` — model loading + per-layer/site pooling. **The load-bearing,
  most-debugged file** (see §6).
- `wm_lib/probing.py` — ridge/PLS probes + scoring (haversine, spearman).
- `extract_acts.py` / `probe_wm.py` / `tfidf_baseline.py` / `build_random_llama.py` /
  `aggregate_results.py` / `plot_results.py` — CLIs.
- `sbatch/W0..W3` — job graph (build random-70b → extract → probe → aggregate).
- `paper_reference.json` — the paper's reported Llama-2-70B R² (the reference row).

## 6. War stories — bugs we hit, so you don't re-hit them

All four cost real cluster time. The fixes are in the code now; a fresh agent should
just know they exist.

1. **Date targets 1000× too small.** pandas ≥2 parsed release/pub dates as
   `datetime64[us]`; `.view(int64)` then gave microseconds, not nanoseconds → every
   year target was garbage. Fixed in `entity_data.target_values` (force ns).
2. **Llama-2 tokenizer cannot be converted by transformers ≥5.** It routes Llama-2's
   SentencePiece `tokenizer.model` through the **tiktoken** loader and crashes with
   `Error parsing line b'\x0e' in tokenizer.model` on *every* path (fast, slow,
   `LlamaTokenizerFast`). Fix: load the Llama tokenizer from
   **`hf-internal-testing/llama-tokenizer`** (ships a prebuilt `tokenizer.json`, same
   32k vocab) via the `tokenizer_hfid` registry field, tried before the model dir.
   `sentencepiece` alone does **not** fix it (the slow path is also broken).
3. **meta-llama is gated.** `Cannot access gated repo`. Fix: default `WM_LLAMA_ORG` to
   the ungated **NousResearch** mirrors (same weights). Override to `meta-llama` if the
   HF token has access.
4. **random-70B build.** `from_config` needs a ~137 GB CPU-RAM spike, so `W0`
   materializes it once (idempotent: reuses the 130 GB weights if a prior run wrote
   them, only (re)writes the tokenizer, which is best-effort since extraction uses the
   `tokenizer_hfid` override anyway).

Also: `sync_main`/`commit_push` in `../stress_tests/sbatch/_common.sh` serialize all git
ops with an flock; the many concurrent W jobs push to `main` fine through it.

## 7. Results so far (best-layer test R²; Spearman in the summary CSVs)

Paper Llama-2-70B row is the reference; our trained Llamas climb toward it.

| arm | world | us | nyc | figures | media | headlines |
|---|--:|--:|--:|--:|--:|--:|
| Llama-2-70B (paper) | .911 | .864 | .359 | .835 | .885 | .746 |
| llama2_13b | .883 | .808 | .272 | .802 | .780 | .663 |
| llama2_7b | .859 | .788 | .249 | .784 | .770 | .592 |
| qwen3_32b | .838 | .702 | .187 | .806 | .727 | .605 |
| qwen3_8b | .797 | .634 | .117 | .774 | .658 | .557 |
| qwen3_1b7 | .655 | .450 | .080 | .693 | .449 | .476 |
| umt5_base | .438 | .325 | .133 | .494 | .153 | .349 |
| cunei400m | .399 | .344 | .114 | .460 | .126 | .343 |
| akk300m | .381 | .312 | .120 | .448 | .123 | .300 |
| **tfidf (floor)** | .642 | .536 | .389 | .645 | .116 | .448 |
| llama2_13b_random | .282 | .290 | .044 | .284 | .038 | .267 |
| llama2_7b_random | .298 | .297 | .070 | .281 | .046 | .260 |
| random (qwen8b) | .327 | .379 | .059 | .276 | .055 | .196 |

**Reading:** (a) trained Llamas reproduce the paper (7B<13B→70B); (b) both families
scale monotonically; (c) random-init collapses *both* space and time (13B world
.883→.282, media .780→.038) → geometry is learned; (d) trained ≫ TF-IDF ≫ random;
(e) small translation encoders are near-floor on English → Akkadian strength is
language-specific. gpt-oss & 70B rows slot at the top when their probes finish.

## 8. How to resume / check

```bash
git fetch origin main -q && git show origin/main:v_1/src/world_models/results/RESULTS.md
squeue -u $USER          # W2 70B probes are the long pole
# re-aggregate locally any time: python v_1/src/world_models/aggregate_results.py
```

## 9. Not done (candidate follow-ups)

Their phase-2 experiments are supported by the stored probe directions but not run:
PCA-dimension sweep, block holdouts (held-out country/century → generalization),
nonlinear MLP probe, prompt ablations, neuron search + interventions. See PLAN §8.
