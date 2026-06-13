# finetune — Akkadian NTP CPT + depth ablation (Task 5, 03.06 meeting)

Fine-tune Qwen3 + gpt-oss-120b on Akkadian next-token prediction and measure
whether (and *at which depth*) it improves the balanced year-PLS signal.
Full design + rationale: **`PLAN_finetune_ntp.md`**. Tokenizer pre-EDA (why no
vocab expansion): **`eda/results/TOKENIZER_EDA.md`**. **Findings: `RESULTS_finetune.md`**.

> **TL;DR (2026-06-14, COMPLETE):** NTP fine-tuning does **not** improve dating
> on the length-controlled (maximal) metric at **any** scale (1.7B/8B/32B/
> gpt-oss-120b) or unfreeze depth — max Δ +0.0013, signal lives in early/frozen
> layers (32B: FT arms byte-identical to base at L6). Only gpt-oss-120b on tier0
> with full-depth training gains (+0.048), i.e. where the length confound lives.
> Scale doesn't win; the 0.4B Akkadian-trained Thalesian-400M does. Full writeup:
> `RESULTS_finetune.md`. Comparison plot: `results/figures/maximal_pls_bestlayer.png`.

## RUN LOG — submitted 2026-06-11 (update as jobs land)

Phase-1 jobs (submitted; dependency-chained, every job pulls main at start and
commits its summaries back to main):

| Job ID | sbatch | what | depends on | status |
|---|---|---|---|---|
| **9554** | FT1_prepare_data | NTP train/val parquets → `v_1/data/finetune/` | — | ✅ done (commit 15ef1e35) |
| **9555** | FT0_extract_gptoss_base | gpt-oss-120b BASE acts (**= meeting Task 2**) | — | ❌ OOM on 4 GPUs → fixed to 8 GPUs, **resubmit** |
| **9556** | FT0b_probe_gptoss_base | gpt-oss base balanced probes → its fig4 layer curve | afterok:FT0 | dep never satisfied (9555 died) → scancel, resubmit after new FT0 |
| **9557_[0-3]** | FT2_qwen3_1b7_ablation | pilot CPT, cuts {0,9,19,25} (array idx 0→cut0, 1→cut9, 2→cut19, 3→cut25) | afterok:9554 | ✅ all 4 trained (FT3 started) |
| **9558** | FT3_probe_qwen3_1b7_ft | extract+probe 4 pilot ckpts, builds scoreboard | afterok:9557 | 🏃 running |

**gpt-oss OOM fix (2026-06-11):** cluster Triton < 3.4.0 → MXFP4 dequantizes to
bf16 (~240GB); 4×H100 (320GB) left no dispatch headroom and `device_map=auto`
overloaded one GPU. FT0/FT7 bumped to **`--gres=gpu:8`** + `mem=512G` +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (FT6 already 8-GPU, same env
added). Resubmitted on 8 GPUs: **FT0=9568** (PD on Resources — waiting for a
free 8-GPU node) → **FT0b=9569** (afterok:9568). Old 9555/9556 dead.

**Pilot training result (9557 arms, all ✅):** base val ppl 11.1 → best 6.65
(cut00) / 7.13 (cut09) / 7.71 (cut19) / 8.69 (cut25). Monotonic in unfrozen
depth, converges by epoch 2. NB base ppl is *low* — Qwen3-1.7B is less
Akkadian-naive than assumed. Dating-probe verdict pending 9558's scoreboard.

**Phase-2 jobs (launched 2026-06-11, gate waived):**
- **FT4=9588** Qwen3-8B ablation → **FT5=9589** probe — ✅ done (flat at maximal)
- **FT6=9590** gpt-oss-120b LoRA → **FT7=9591** probe — ✅ done (tier0 +0.048 full-depth only)
- **FT4b=9596** Qwen3-32B ablation ✅ → **FT5b=9597** probe ✅ done (flat at maximal, Δ=0.000; ft21/43/58 byte-identical to base at L6)
- M4 k-sweep **9586** ✅ + M5 mlm-fix **9648** ✅ + gpt-oss-into-ksweep **9655** ✅.
  Maximal panel set complete: fig1/2/4 + MAE = 8 models, k-sweep = **9 models**
  (gpt-oss added). NB at maximal the 37M mlm lands at 0.311 (≈ random); only
  thalesian_cunei400m (0.411) clearly leads.

**DONE (2026-06-14):** all four families probed (base + depth ablation) across
maximal + tier0. gpt-oss folded into the canonical maximal panels + k-sweep
(9 models). Local comparison plot `plot_maximal_pls.py --with-ft` →
`results/figures/maximal_pls_{bestlayer,layerwise}.png`. Optional polish left:
FT8/M6 to fold the best FT *cut* into the cluster fig1/2/4 (local plot already shows it).

### gpt-oss k-sweep — debugging trail (resolved)
Getting gpt-oss into the k-sweep took three fixes, all now in: (1) renderer
`maximal_ksweep.py` had a hardcoded 8-model `MODELS` list — added `gpt_oss_120b`;
(2) M4 copies the gpt-oss std-probe summary into `probes/` so `--emit-layers`
resolves its best layer; (3) the first ksweep draws (job 9650) were computed
before gpt-oss's best layer L5 was in the union, and `run_mc_probes` resume
skips a draw whole-file — so M4 now **purges** gpt-oss ksweep draws before
recompute. Lesson: when adding a model to a resumable sweep, purge its stale
per-draw files or the layer set silently stays out of date.

**Gate decision (2026-06-11):** gate WAIVED — free grant cluster, redo cost
low, parallelism saves wall-clock. Pilot trained cleanly (ppl 11.1→6.65,
converged epoch 2) so the shared recipe is validated; FT4/FT5 (8B) and
FT6/FT7 (gpt-oss LoRA) launched in parallel with 9558 rather than waiting.
Risk accepted: if 9558's scoreboard later shows the recipe needs a tweak,
rerun the affected arms. Pilot target to beat: base qwen3_1b7 year-PLS
**0.355 @ L9 (maximal)** / 0.397 (tier0).

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
