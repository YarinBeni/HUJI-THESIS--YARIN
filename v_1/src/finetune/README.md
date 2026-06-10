# finetune — Akkadian NTP CPT + depth ablation (Task 5, 03.06 meeting)

Fine-tune Qwen3 + gpt-oss-120b on Akkadian next-token prediction and measure
whether (and *at which depth*) it improves the balanced year-PLS signal.
Full design + rationale: **`PLAN_finetune_ntp.md`**. Tokenizer pre-EDA (why no
vocab expansion): **`eda/results/TOKENIZER_EDA.md`**.

## RUN LOG — submitted 2026-06-11 (update as jobs land)

Phase-1 jobs (submitted; dependency-chained, every job pulls main at start and
commits its summaries back to main):

| Job ID | sbatch | what | depends on | status |
|---|---|---|---|---|
| **9554** | FT1_prepare_data | NTP train/val parquets → `v_1/data/finetune/` | — | submitted |
| **9555** | FT0_extract_gptoss_base | gpt-oss-120b BASE acts, ORCC, tier0+maximal (**= meeting Task 2**) | — | submitted |
| **9556** | FT0b_probe_gptoss_base | gpt-oss base balanced probes → its fig4 layer curve | afterok:9555 | queued |
| **9557_[0-3]** | FT2_qwen3_1b7_ablation | pilot CPT, cuts {0,9,19,25} (array idx 0→cut0, 1→cut9, 2→cut19, 3→cut25) | afterok:9554 | queued |
| **9558** | FT3_probe_qwen3_1b7_ft | extract+probe 4 pilot ckpts, builds scoreboard | afterok:9557 (whole array) | queued |

**⛔ GATE after 9558:** `git pull` → review `results/scoreboard_best.csv` +
`results/train_summaries/qwen3_1b7_cut*.json` (val ppl must drop a lot from
the Akkadian-naive base). Pilot target to beat: base qwen3_1b7 year-PLS
**0.355 @ L9 (maximal)** / 0.397 (tier0). Only then submit phase 2
(FT4→FT5 for Qwen3-8B; FT6→FT7 for gpt-oss-120b LoRA) — commands in §Submit order.

Concurrent (unrelated round, same cluster): **9552** = maximal_figs M4 PLS
k-sweep (re-run without tfidf), **9553** = M5 mlm-fix + re-render all maximal
figures to 8 models (afterok:9552). Those finish the *maximal panel set*
(fig1/2/4 + MAE + k-sweep) whose **fig4 layer peaks are exactly the input that
chose this round's unfreeze cuts** (1b7 peak L9 → cuts {0,9,19,25}; 8b peak
L16 → cuts {0,12,24,32}).

Useful monitoring (on the cluster):

```bash
squeue -u $USER
tail -f v_1/src/finetune/logs/FT2_9557_0.out                     # one pilot arm
sacct -j 9557 --format=JobID,JobName%18,State,Elapsed,MaxRSS     # whole array
grep -h "\[eval\]" v_1/src/finetune/logs/FT2_9557_*.out          # ppl per arm
```

## Layout

```
eda/                     tokenizer pre-EDA (local, no GPU)
prepare_ntp_data.py      unified train/val -> fragment texts (tier0) parquets
train_ntp.py             CPT trainer: --unfreeze-from <block> [--lora]
build_scoreboard.py      base-vs-arms layer curves + best-layer table
sbatch/                  FT0..FT7 job scripts (see PLAN job table)
results/
  train_summaries/       per-arm val-ppl history (committed by FT2/FT4/FT6)
  probes/                balanced-MC probe JSONs (summaries committed)
  scoreboard_*.csv|md    final comparison (built by FT3/FT5/FT7)
logs/                    sbatch stdout
```

## Submit order (from repo root on the cluster)

```bash
sbatch v_1/src/finetune/sbatch/FT1_prepare_data.sbatch        # data (CPU, minutes)
sbatch v_1/src/finetune/sbatch/FT0_extract_gptoss_base.sbatch # gpt-oss base acts (= meeting Task 2)
sbatch v_1/src/finetune/sbatch/FT0b_probe_gptoss_base.sbatch  # after FT0
sbatch v_1/src/finetune/sbatch/FT2_qwen3_1b7_ablation.sbatch  # pilot, after FT1 (array 0-3)
sbatch v_1/src/finetune/sbatch/FT3_probe_qwen3_1b7_ft.sbatch  # after FT2
# --- GATE: check results/scoreboard_best.csv before scaling up ---
sbatch v_1/src/finetune/sbatch/FT4_qwen3_8b_ablation.sbatch   # main (array 0-3)
sbatch v_1/src/finetune/sbatch/FT5_probe_qwen3_8b_ft.sbatch   # after FT4
sbatch v_1/src/finetune/sbatch/FT6_gptoss120b_lora_ablation.sbatch  # after gate + FT0 (array 0-3)
sbatch v_1/src/finetune/sbatch/FT7_probe_gptoss_ft.sbatch     # after FT6 (array 0-3)
```

Every job git-pulls main first and commits its summary JSONs back, so progress
is visible from a local `git pull`.

## Method names in the probing pipeline

New activation dirs follow the standard `<method>_<cleaning>_<pooling>` leaf
under `orcc__embed/activations/`. Methods (`gpt_oss_120b`, `qwen3_1b7_ft09`,
`qwen3_8b_ft24`, `gpt_oss_120b_ft32`, …) are **auto-registered** by
`run_mc_probes.py` (layer count read from the `layer_*.npz` files), so no
hand-edits to the dispatch table per checkpoint.
