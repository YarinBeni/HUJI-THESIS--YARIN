#!/bin/bash
#SBATCH --job-name=C6_hsic
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=chrono/sbatch/logs/C6_hsic_%A_%a.out
# ===========================================================================
# C6 — P2 first step: Chrono-Barlow head + HSIC penalty (hidden layer vs
# provenance one-hot), lambda in {1, 10}, 3 encoders x 3 seeds, tier0
# Akkadian, raw features. Heads saved for the nonlinear-recovery probe.
# ===========================================================================
set -uo pipefail
ENCS=(cunei400m llama2_7b qwen3_8b); LAMS=(1 10)
E=$(( SLURM_ARRAY_TASK_ID / 6 )); R=$(( SLURM_ARRAY_TASK_ID % 6 )); L=$(( R / 3 )); SEED=$(( R % 3 ))
CFG="emin2_${ENCS[$E]}_t0_akk_hsic${LAMS[$L]}_provenance"
echo "=== C6 ${CFG} seed ${SEED} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/tier0/hsic/scores chrono/reports/tier0/hsic/heads
ART=chrono/artifacts_tier0
for FOLD in 0 1 2 3 4; do
    python -u chrono/scripts/train_cjb.py --config "chrono/configs/${CFG}.yaml" --seed "${SEED}" --fold "${FOLD}" \
        --corpus "${ART}/corpus_chrono.parquet" --views "${ART}/views.parquet" \
        --ruler-table "${ART}/ruler_table.parquet" --store-root "${ART}/emb_store" --splits-dir "${ART}/splits" \
        || { echo "FAILED train_cjb ${CFG} seed ${SEED} fold ${FOLD}"; exit 1; }
done
cp chrono/artifacts/scores/"${CFG}"-s"${SEED}"-*.parquet chrono/reports/tier0/hsic/scores/ 2>/dev/null || true
[ "${SEED}" -eq 0 ] && cp chrono/artifacts/heads/"${CFG}"-s0-*.pt chrono/reports/tier0/hsic/heads/ 2>/dev/null || true
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C6: hsic ${CFG} seed ${SEED} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
