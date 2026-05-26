# Phase 2 RUNBOOK — cluster jobs C1–C13 (Round-3 backfill)

These are the sbatch/launcher commands for Phase 2 of `MASTER_BACKFILL_PLAN.md`
(§4 status matrix, §5 job list C1–C13). **Yarin submits every job himself and
reports the returned job IDs** — nothing here is run by the agent. Every job
that writes results commits + pushes to `origin/main` as Yarin's own cluster
job (each does `git pull --rebase origin main` before pushing); **all pushing is
gated on Yarin's explicit go-ahead** — if you do not want auto-push yet, comment
out the `git push origin main` lines before submitting. Run all commands from
the cluster repo root `~/projects/HUJI-THESIS--YARIN`.

Conda env `thesis`, partition `voltagepark`. Fill the `<…JOBID>` placeholders
with the IDs `sbatch --parsable` / the launchers print.

---

## Prerequisites (activation dirs)

Balanced-MC and balanced-geodesic jobs read the balanced draws at
`v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/{draws_matrix.npy,corpus_fragment_order.json}` (present).

Last-token activation dirs required by **C3** (all present on disk except `random`):

| model | dir | produced by |
|---|---|---|
| qwen3_1b7 | `…/orcc__embed/activations/qwen3_1b7_tier0_last` | `sbatch/orcc/extract_qwen3_1b7_tier0_last.sh` (done) |
| qwen3_8b  | `…/qwen3_8b_tier0_last`  | `sbatch/orcc/extract_qwen3_8b_tier0_last.sh` (done) |
| qwen3_32b | `…/qwen3_32b_tier0_last` | `sbatch/orcc/extract_qwen3_32b_tier0_last.sh` (done) |
| thalesian_akk300m | `…/thalesian_akk300m_tier0_last` | `round2_phase3/sbatch/extract_thalesian_akk300m_tier0.sh` (done — extracts mean+last) |
| thalesian_cunei400m | `…/thalesian_cunei400m_tier0_last` | `round2_phase3/sbatch/extract_thalesian_cunei400m_tier0.sh` (done) |
| random (qwen3-8b-init) | `…/random_tier0_last` | **C1** (this runbook) |

> NOTE: the existing Thalesian `extract_*_tier0.sh` scripts already extract BOTH
> `mean` and `last` poolings, so **no new thalesian `*_tier0_last` script is
> needed**. The `*_tier0_last` dirs are confirmed present.

---

## Wave 0 — C1 (GPU extraction; gates random everywhere)

```bash
cd ~/projects/HUJI-THESIS--YARIN && git pull --rebase origin main
sbatch v_1/src/linear_probing/sbatch/orcc/extract_random_qwen3_8b.sh
# -> C1_JOBID   (~3–6 h; random_{tier0,maximal}_{mean,last} activations)
```

## Wave 1 — everything that does NOT need C1 (submit immediately, parallel)

```bash
cd ~/projects/HUJI-THESIS--YARIN && git pull --rebase origin main

# C3 — balanced-MC LAST-TOKEN for the 5 non-random models (random handled in Wave 2).
#      Launcher fans out chunks + per-model afterok aggregate.
bash v_1/src/linear_probing/sbatch/orcc/submit_mc_lasttoken.sh \
     qwen3_1b7 qwen3_8b qwen3_32b thalesian_akk300m thalesian_cunei400m

# C4 — Ridge-year imbalanced backfill (mlm + thalesian x2).
sbatch v_1/src/linear_probing/sbatch/orcc/probe_ridge_imbalanced.sh

# C5 — Ridge-year balanced backfill for the NON-random models (thalesian x2).
bash v_1/src/linear_probing/sbatch/orcc/submit_mc_backfill.sh \
     thalesian_akk300m thalesian_cunei400m

# C8 — geodesic LORO for qwen3_1b7 (tier0/mean/L1). Launcher prints 11 ruler job ids.
bash v_1/src/geodesic/phase_c/sbatch/submit_loro_qwen3_1b7.sh
# then aggregate (rebuilds loro_robustness.json, commits):
sbatch --dependency=afterok:<C8_RULER_JOBIDS_colon_separated> \
       v_1/src/geodesic/phase_c/sbatch/aggregate_loro_job.sh

# C9 — Phase-D centroid-spline for qwen3_1b7 (tier0/mean/L1).
bash v_1/src/geodesic/phase_d/sbatch/submit_phase_d_qwen3_1b7.sh

# C11 — balanced LORO on 4 best configs (single sequential job, single committer).
sbatch v_1/src/geodesic/phase_c/sbatch/balanced_loro_job.sh

# C12 — elicitation kp0/kp1/kp2 on qwen3 x3 (job array, 3 tasks).
sbatch v_1/src/linear_probing/round2_phase1a/sbatch/run_p1a_kp_qwen3.sbatch

# C13 stage 1 — extract prompted activations pv0..pv3 on qwen3 x3 (GPU array).
sbatch v_1/src/linear_probing/round2_phase1b/sbatch/run_p1b_extract_qwen3.sbatch
# -> C13_EXTRACT_JOBID
```

## Wave 2 — needs C1 (submit after C1 has a job id; `afterok` keeps them queued)

```bash
cd ~/projects/HUJI-THESIS--YARIN && git pull --rebase origin main

# C2 — random imbalanced probes (PLS/PLS-DA/Ridge/CLS) + imbalanced geodesic scan.
sbatch --dependency=afterok:<C1_JOBID> \
       v_1/src/linear_probing/sbatch/orcc/probe_random.sh

# C3 (random) — last-token balanced-MC for random; launcher wires afterok:C1 itself.
C1_JOBID=<C1_JOBID> bash v_1/src/linear_probing/sbatch/orcc/submit_mc_lasttoken.sh random

# C5 (random) — Ridge-year balanced backfill for random; launcher wires afterok:C1.
C1_JOBID=<C1_JOBID> bash v_1/src/linear_probing/sbatch/orcc/submit_mc_backfill.sh random

# C10 — balanced geodesic for all 6 models (single sequential job; needs random acts).
sbatch --dependency=afterok:<C1_JOBID> \
       v_1/src/geodesic/phase_b/sbatch/balanced_scan_job.sh
```

## Wave 3 — depends on Wave-1 producers

```bash
cd ~/projects/HUJI-THESIS--YARIN && git pull --rebase origin main

# C13 stage 2 — reprobe PLS+CLS on the prompted activations (afterok on the
#               whole extract array waits for all 3 tasks). Terminal committer.
sbatch --dependency=afterok:<C13_EXTRACT_JOBID> \
       v_1/src/linear_probing/round2_phase1b/sbatch/run_p1b_reprobe_qwen3.sbatch
```

---

## Dependency DAG

```
C1 ─┬─► C2                (probe_random.sh)
    ├─► C3-random         (submit_mc_lasttoken.sh random)
    ├─► C5-random         (submit_mc_backfill.sh random)
    └─► C10               (balanced_scan_job.sh — needs random acts)

C3 chunks ─► C3 aggregate (per model; wired inside submit_mc_lasttoken.sh)
C5 chunks ─► C5 aggregate (per model; wired inside submit_mc_backfill.sh)
C8 LORO ruler jobs ─► aggregate_loro_job.sh
C13 extract array ─► C13 reprobe array

Independent (submit immediately): C3-non-random, C4, C5-non-random, C8, C9,
C11, C12, C13-extract.
```

## Monitoring & expected outputs

- `squeue -u $USER`
- C1 → `…/orcc__embed/activations/random_{tier0,maximal}_{mean,last}/`
- C2 → `cls/pls/cls_numeric_results_random.json` + `…/phase_b/phase_b_random_*.json`
- C3 → `…/orcc_round2_phase0/probes/<model>_<probe>__mc_balanced_last__*.json` (+ summaries)
- C4 → `…/orcc__probe_cls_numeric/cls_numeric_results_{mlm,thalesian_akk300m,thalesian_cunei400m}.json`
- C5 → `<model>_cls_numeric__mc_balanced__*.json` summaries
- C8 → `…/phase_c/loro_qwen3_1b7_tier0_mean_L01_*.json` → `…/results/loro_robustness.json`
- C9 → `…/phase_d/…qwen3_1b7…` + `T6` source
- C10 → `…/results/geodesic_layer_scoreboard_balanced.json` (6 models merged)
- C11 → `…/results/loro_robustness_balanced.json` (4 configs merged)
- C12 → `…/orcc_round2_phase1a/direct_kp_qwen3_{1b7,8b,32b}/scores/*` (→ T9 in Phase 3)
- C13 → `…/orcc_round2_phase1b_qwen3_{1b7,8b,32b}/reprobing/*` (→ T10 in Phase 3)

---

## TODOs / things that could NOT be fully auto-wired

1. **`run_mc_probes.py` filename collision (mitigated, verify after first C3
   chunk).** The current code writes per-draw files with a fixed
   `method_tag=mc_balanced`, independent of pooling. If the last-token sweep
   used the same tag it would OVERWRITE the existing mean-pooled balanced draws.
   `mc_chunk.sh` / `mc_aggregate.sh` now pass `--method-tag mc_balanced_last`
   for the last-token sweep so files are disjoint. **Confirm after the first
   chunk that `<model>_pls__mc_balanced_last__draw000.json` is written and the
   existing `…__mc_balanced__draw*.json` (mean) files are untouched.** If the
   table builder (Phase-3 `M`/`T1`) does not yet know to read the
   `_last`-tagged summaries, that wiring is a Phase-3 local task.

2. **`scan.py` / `loro.py` balanced files are single-fixed-name + overwrite.**
   `run_balanced` in both writes ONE fixed file (`geodesic_layer_scoreboard_balanced.json`
   / `loro_robustness_balanced.json`) containing only the current run's records;
   `loro.py` even ignores `--output-dir` for it. A per-model fan-out would
   clobber. **C10 and C11 are therefore single sequential jobs** that write each
   model/config into a private subdir (C10) or snapshot the file after each run
   (C11) and merge into the canonical file at the end. If `scan.py`/`loro.py`
   later gain native append/merge, these can be parallelized.

3. **C13 per-model layer sweep is a judgement call.** qwen2.5 used
   `0,4,10,15,22,28` (29 hidden states). qwen3 models have different depths, so
   the scripts use `{0, 4 evenly-spaced, top}` per model
   (1b7=`0,7,14,21,27`, 8b=`0,9,18,27,36`, 32b=`0,16,32,48,63`). If a specific
   layer set is required for the T10 comparison, edit `LAYERSETS` in both
   `run_p1b_extract_qwen3.sbatch` and `run_p1b_reprobe_qwen3.sbatch` (they must
   match).

4. **C13 Qwen3 chat template / system prompt.** `extract_prompted_acts.py` and
   `run_kp.py` were authored for Qwen2.5-Instruct. Qwen3 base vs. instruct chat
   templates / the pv0 "empty system prompt" requirement may differ. **Verify the
   first extract task's metadata.json shows a sane prompt render before trusting
   T9/T10.** (This is harness behaviour, owned by eval-harness-builder — flagged,
   not fixed here.)
```
