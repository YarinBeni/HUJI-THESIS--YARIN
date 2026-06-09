# maximal_figs — the 5 requested plots, recreated for **maximal / mean / balanced**

One place for every plot Yarin asked to recreate under the config:
**cleaning = maximal · pooling = mean · regime = balanced (200 MC draws) · year-PLS**,
across the **8-model** set.

This folder is **isolated** — it reads its own probe outputs and writes its own
tables/figures. It does **not** touch the canonical Round-3 pipeline
(`plot_round3_story_figures.py`, `build_experiment_tables.py`, `results/tables/`,
`results/figures/round3_story/`), which stays as the committed thesis result.
The plotting script only *imports the styling registries* (colors/markers/labels)
from `plot_round3_story_figures.py` so the look matches.

## Model set (8)

`tfidf` · `mlm` (37M) · `thalesian_akk300m` · `thalesian_cunei400m` ·
`qwen3_1b7` · `qwen3_8b` · `qwen3_32b` · `random` (random-init Qwen3-8B).

## The 5 figures (final list)

| File in `figures/` | What it is | Source data |
|---|---|---|
| `fig1_maximal_ACD.png` | fig1 panels **A** (year PLS vs Ridge), **C** (ruler surface), **D** (name-mask). Panel B (pooling) dropped. | `probes/` (A,C) + `results/tables/T7_name_masking.csv` maximal rows (D, already exists) |
| `fig2_maximal_AB.png` | fig2 panels **A** (year-PLS Sp vs size), **B** (year-Ridge Sp vs size). Panel C (Isomap) dropped. | `probes/` |
| `fig4_maximal_A.png` | fig4 panel **A** layerwise, **raw layer index** x-axis (1,2,…,max — NOT normalized depth). | `probes/` (all layers) |
| `permodel_mae_ruler.png` | per-ruler MAE, per-model breakdown, 8 models. | `predictions/predictions.csv` |
| `bars_mae_ruler.png` | per-ruler MAE bars, 8 models side by side. | `predictions/predictions.csv` |

## Run order (you run the sbatch; cluster writes + commits)

**Step 1 — supervised probes + fig1/2/4** (`sbatch/M1_supervised_maximal.sbatch`)
Runs `run_mc_probes.py --cleaning maximal --pooling mean --layers all` for the 8
models × {PLS, Ridge, ruler-CLS}, into `probes/`, then runs `make_maximal_figs.py`
→ `tables/` + `figures/{fig1_maximal_ACD,fig2_maximal_AB,fig4_maximal_A}.png`.
> Bottleneck: `qwen3_32b` all-layers × 200 draws. Job uses `--n-jobs=$CPUS`. If it
> walltimes, split 32b into its own sbatch (same command, `--probes qwen3_32b_*`).

**Step 2 — MAE plots** (`sbatch/M2_mae_maximal.sbatch`) — run **after** Step 1.
Reads each model's best maximal year-PLS layer straight from
`probes/<m>_pls__mc_balanced_maximal__summary.json` (so no hand-entered layers),
runs `dump_oof_predictions_balanced.py --cleaning maximal --models <8> --pls-k 3`
→ `predictions/`, then `analyze_per_model.py` → the two MAE plots, copied into
`figures/`.

## Open decision baked in (change if you want)

- **MAE uses fixed k=3** (`--pls-k 3`), NOT best-k-per-draw. This removes the
  selection-bias inflation (each draw was picking its luckiest k post-hoc). To go
  back to the optimistic best-k-per-draw, drop `--pls-k 3` from M2.

## Caveats to expect (real, not bugs)

- Maximal year signal is near-zero for `random`/`qwen` (Sp ~0.02–0.09 vs ~0.4 at
  tier0), so fig2/fig4 maximal will look much flatter than the tier0 versions.
- Needs maximal/mean activations on the cluster for all 8 models. If `mlm` or any
  model lacks a `<model>_maximal*` activation dir, M1 prints `[skip]` for it and
  that model is simply absent from the maximal figures.
