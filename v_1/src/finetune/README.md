# finetune — Akkadian NTP CPT + depth ablation (Task 5, 03.06 meeting)

Fine-tune Qwen3 + gpt-oss-120b on Akkadian next-token prediction and measure
whether (and *at which depth*) it improves the balanced year-PLS signal.
Full design + rationale: **`PLAN_finetune_ntp.md`**. Tokenizer pre-EDA (why no
vocab expansion): **`eda/results/TOKENIZER_EDA.md`**.

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
