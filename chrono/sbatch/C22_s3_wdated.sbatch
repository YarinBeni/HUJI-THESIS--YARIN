#!/bin/bash
#SBATCH --job-name=C22_s3b
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=chrono/sbatch/logs/C22_s3b_%A_%a.out
# ===========================================================================
# C13 — S3 fine-tune sweep on the dated texts (tier0 akk arm): 22 configs
# (2 encoders x {no init, barlow init, jepa init} x {25, 50, 100}% labels, plus
# head sizes 128 / 2048) x 3 seeds, 5 folds each. Adapter inits come from the
# C10 heads pushed to chrono/reports/ssl/heads/. Read-out: aggregate_emin.py.
# Needs: afterany on the C10 array.
# ===========================================================================
set -uo pipefail
CFGS=($(ls chrono/configs/s3b/*.yaml | xargs -n1 basename | sed 's/.yaml$//' | sort))
CI=$(( SLURM_ARRAY_TASK_ID / 3 )); SEED=$(( SLURM_ARRAY_TASK_ID % 3 )); CFG=${CFGS[$CI]}
[ -z "${CFG:-}" ] && { echo "no config for task ${SLURM_ARRAY_TASK_ID}"; exit 0; }
echo "=== C22 ${CFG} seed ${SEED} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/tier0/s3_scores
INIT=$(python -c "import yaml;print(yaml.safe_load(open('chrono/configs/s3b/${CFG}.yaml')).get('init_head',''))")
if [ -n "$INIT" ] && [ ! -f "$INIT" ]; then echo "SKIP ${CFG}: init head ${INIT} not available"; exit 0; fi
ART=chrono/artifacts_tier0
for FOLD in 0 1 2 3 4; do
    python -u chrono/scripts/train_cjb.py --config "chrono/configs/s3b/${CFG}.yaml" --seed "${SEED}" --fold "${FOLD}" \
        --corpus "${ART}/corpus_chrono.parquet" --views "${ART}/views.parquet" \
        --ruler-table "${ART}/ruler_table.parquet" --store-root "${ART}/emb_store" --splits-dir "${ART}/splits" \
        || { echo "FAILED ${CFG} seed ${SEED} fold ${FOLD}"; exit 1; }
done
cp chrono/artifacts/scores/"${CFG}"-s"${SEED}"-*.parquet chrono/reports/tier0/s3_scores/ 2>/dev/null || true
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C22: ${CFG} seed ${SEED} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
