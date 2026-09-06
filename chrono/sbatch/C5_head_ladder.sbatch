#!/bin/bash
#SBATCH --job-name=C5_headlad
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-35
#SBATCH --output=chrono/sbatch/logs/C5_headlad_%A_%a.out
# ===========================================================================
# C5 — P1 head ladder: does the Chrono-Barlow head's advantage survive LEACE
# erasure of provenance / period / object type / length from the frozen
# features? 3 encoders x 4 concepts x 3 seeds, tier0 Akkadian arm, 5 folds
# each. The unerased reference is the E-MIN v2 akk arm (C3v2).
# ===========================================================================
set -uo pipefail
ENCS=(cunei400m llama2_7b qwen3_8b); CONS=(provenance period subgenre length)
E=$(( SLURM_ARRAY_TASK_ID / 12 )); R=$(( SLURM_ARRAY_TASK_ID % 12 )); C=$(( R / 3 )); SEED=$(( R % 3 ))
CFG="emin2_${ENCS[$E]}_t0_akk_erase_${CONS[$C]}"
echo "=== C5 ${CFG} seed ${SEED} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/tier0/ladder/head_scores
ART=chrono/artifacts_tier0
for FOLD in 0 1 2 3 4; do
    python -u chrono/scripts/train_cjb.py --config "chrono/configs/${CFG}.yaml" \
        --seed "${SEED}" --fold "${FOLD}" \
        --corpus "${ART}/corpus_chrono.parquet" --views "${ART}/views.parquet" \
        --ruler-table "${ART}/ruler_table.parquet" --store-root "${ART}/emb_store" \
        --splits-dir "${ART}/splits" \
        || { echo "FAILED train_cjb ${CFG} seed ${SEED} fold ${FOLD}"; exit 1; }
done
cp chrono/artifacts/scores/"${CFG}"-s"${SEED}"-*.parquet chrono/reports/tier0/ladder/head_scores/ 2>/dev/null || true
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C5: head ladder ${CFG} seed ${SEED} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
