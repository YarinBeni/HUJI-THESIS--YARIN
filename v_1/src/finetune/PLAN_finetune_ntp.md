# PLAN — Fine-tune Qwen3 + gpt-oss-120b on Akkadian NTP (Task 5, 03.06 meeting)

**Goal:** test whether continued pretraining (CPT) on Akkadian next-token prediction closes the
gap between Akkadian-naive big LLMs and the Akkadian-trained small models (MLM-37M, Thalesian-400m),
and *where in depth* the training must happen (Gabi's depth ablation). Success metric: balanced
year-PLS Spearman (and ruler-CLS) on the ORCC corpus, before vs after CPT, per layer (fig4-style).

**Pre-EDA verdict** (see `eda/results/TOKENIZER_EDA.md`): **no vocabulary expansion.**
Corpus = 11.25M Qwen3 / 10.29M gpt-oss tokens; domain BPE saturates at 3.85 tokens/word
(only 8–16 % better than stock); the high-value candidate tokens are royal/divine names →
adding them would hard-code the name-leakage channel we control with name-masking.
Plain CPT with the stock tokenizer; embeddings stay frozen except in the full-depth arm.

## Design

### Models & depth cuts ("cut = k" → unfreeze transformer blocks k..N-1 + final norm + lm_head)

Meeting rule: cuts at 0 % / ~33 % / ~67 % / ~90 % of depth ("for a 30-layer model: 0, 10, 20, 27").

| model | HF id | blocks | cuts | fig4 anchor (peak hidden-state) |
|---|---|---|---|---|
| qwen3_1b7 (pilot) | `Qwen/Qwen3-1.7B` | 28 | **0, 9, 19, 25** | hs 9 ≈ block 8 → cut 9 sits at the peak |
| qwen3_8b (main) | `Qwen/Qwen3-8B` | 36 | **0, 12, 24, 32** | hs 16 ≈ block 15 → straddled by cuts 12/24 |
| gpt_oss_120b | `openai/gpt-oss-120b` | 36 (MoE, 128 experts) | **0, 12, 24, 32** | unknown → measured in step FT0 |

Notes:
- Same checkpoints that were probed (NOT `-Base` variants) so before/after is apples-to-apples.
- Qwen3-1.7B has tied embeddings → in cut>0 arms we freeze the tie (lm_head trains only for cut=0). Logged in metadata.
- qwen3_32b is excluded for now (65 blocks, marginal probing gain over 8B, 4× cost). Can be added later with cuts 0/21/43/58.

### Training recipe

- **Objective:** causal LM on tier0 fragment texts (same text construction as the ORCC corpus build:
  space-joined `value_clean` fallback `value_raw`, tier0 cleaning), unified **train** split only
  (1.96M words ≈ 9M tokens); unified **val** split for perplexity.
- **Packing:** tokenize → concat with EOS → chunks of 2048.
- **Qwen3 arms — full FT of unfrozen blocks:** bf16, AdamW, lr 1e-5 (cosine, 3 % warmup), wd 0.01,
  global batch 64×2048 tokens, 3 epochs (~70 steps/epoch), eval+save each epoch, keep best by val loss.
- **gpt-oss-120b arms — LoRA restricted to blocks ≥ cut:** MXFP4 → bf16 dequantized, LoRA r=32/α=64
  on attention + expert projections (`layers_to_transform=[cut..35]`), lr 2e-4, otherwise same schedule.
  Full FT of a 120B MoE is out of budget; LoRA-above-the-cut preserves the depth-ablation design.
  (Method difference vs Qwen3 is a known confound — comparisons are *within* model, across cuts.)
- **Leakage accounting:** 504/1202 ORCC probing fragments are in the unified train split. NTP sees raw
  text only (no labels) and the MLM-37M baseline trained on the same split — kept for comparability,
  reported. Optional later ablation: best arm retrained on train-minus-ORCC.

### Evaluation per checkpoint

1. Val perplexity (per epoch, from training logs).
2. Extract ORCC activations: `03_extract_seal_activations.py --model <ckpt>` × {tier0, maximal} × mean pooling
   (LoRA ckpts loaded via new `--lora-adapter` flag, merged in memory).
3. Probe: `run_mc_probes.py` (balanced 200 MC draws) → year-PLS + ridge + ruler-CLS, `--layers all`.
   New methods (e.g. `qwen3_8b_ft12`, `gpt_oss_120b`) are auto-registered: layer count inferred from
   the activation dir (additive change to `run_mc_probes.py`).
4. Compare: fig4-style layer curves base vs cuts; scoreboard table (best-layer Spearman per arm).

## Job plan (you submit; each job git-pulls main first)

| # | sbatch | what | GPUs | est. |
|---|---|---|---|---|
| FT1 | `sbatch/FT1_prepare_data.sbatch` | build packed NTP train/val parquets → `v_1/data/finetune/` | CPU | ~10 min |
| FT0 | `sbatch/FT0_probe_gptoss_base.sbatch` | **= meeting Task 2.** Extract gpt-oss-120b base acts (tier0+maximal, mean) + balanced probes → its fig4 curve | 4×H100 | ~6–10 h |
| FT2 | `sbatch/FT2_qwen3_1b7_ablation.sbatch` | pilot: 1.7B CPT, array 0-3 over cuts {0,9,19,25} | 1×H100 ×4 | ~2–3 h each |
| FT3 | `sbatch/FT3_probe_qwen3_1b7_ft.sbatch` | extract+probe the 4 pilot checkpoints | 1×H100 | ~4 h |
| — | **GATE:** review pilot scoreboard before scaling up | | | |
| FT4 | `sbatch/FT4_qwen3_8b_ablation.sbatch` | 8B CPT, array 0-3 over cuts {0,12,24,32} | 2×H100 ×4 | ~4–6 h each |
| FT5 | `sbatch/FT5_probe_qwen3_8b_ft.sbatch` | extract+probe 4× 8B checkpoints | 1×H100 | ~6 h |
| FT6 | `sbatch/FT6_gptoss120b_lora_ablation.sbatch` | 120b LoRA CPT, array 0-3 over cuts {0,12,24,32} | 8×H100 ×4 | ~8–12 h each |
| FT7 | `sbatch/FT7_probe_gptoss_ft.sbatch` | extract+probe 4× 120b checkpoints | 4×H100 | ~12 h |

Dependencies: FT2/FT4/FT6 need FT1. FT0 independent (do early — it both unblocks interpretation of FT6/FT7
and completes meeting Task 2). FT3 after FT2, FT5 after FT4, FT7 after FT6 + FT0.

## Decision gates

- **After FT3 (pilot):** does any cut beat base qwen3_1b7 (peak Spearman 0.355 maximal)? If all arms are flat or
  degrade, revisit recipe (lr, epochs) before spending on 8B/120b.
- **After FT0:** gpt-oss-120b base fig4 curve → confirm cuts {0,12,24,32} straddle its peak; adjust if not.
- val perplexity must drop substantially vs base (Akkadian-naive ppl will start very high); if it doesn't, training is broken — stop.

## Outputs / where things land

- checkpoints: `v_1/models/finetune/<model_tag>/cut<NN>/` (cluster only, gitignored)
- NTP data: `v_1/data/finetune/` (cluster, gitignored)
- activations: `v_1/src/linear_probing/results/orcc__embed/activations/<method>_<cleaning>_mean/`
  with methods `gpt_oss_120b`, `qwen3_1b7_ft{00,09,19,25}`, `qwen3_8b_ft{00,12,24,32}`, `gpt_oss_120b_ft{00,12,24,32}`
- probes: `v_1/src/finetune/results/probes/` (JSON summaries committed by jobs)
- scoreboard + figures: `v_1/src/finetune/results/` (built after FT3/FT5/FT7)

## FT8 / "M6" — fold winners into the maximal_figs panel set (after FT5/FT7)

The FT probes are format-identical to the M-track (same draws/config/summary
JSONs), so the fine-tuned models can join the 8-model maximal figures. Planned
job, built once the scoreboard names the winners:

1. Extend the model registries (`make_maximal_figs.py` ALL_MODELS/LAYERED +
   styling/param-count registries in `plot_round3_story_figures.py`) with
   `gpt_oss_120b` (base) and the **best cut per family only** — full 4-cut
   grids stay in `finetune/results/figures/` (20 curves would be unreadable;
   8 → ~11 models in the headline figures).
2. Render fig1/2/4 from the union of `maximal_figs/probes/` +
   `finetune/results/probes/` (fig2 needs gpt-oss's size point: 117B total /
   5.1B active — plot at total params, annotate active).
3. `dump_oof_predictions_balanced.py` + `analyze_per_model.py` for the new
   methods → per-ruler MAE plots include them.
