#!/bin/bash
#SBATCH --job-name=C3v2c_spread
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=chrono/sbatch/logs/C3v2c_spread_%A_%a.out
# ===========================================================================
# C3v2c — seed spread for the frozen-probe baselines: ridge (orig views and
# all views) refitted 4x with 10% of the train docs dropped per fold, for the
# three Akkadian arms and the three mixed arms. Lets the E-MIN v2 margins be
# quoted with +/- on both sides.
# ===========================================================================
set -uo pipefail
CONFIGS=(emin2_cunei400m_t0_akk emin2_llama2_7b_t0_akk emin2_qwen3_8b_t0_akk
         emin2_cunei400m_t0_mix emin2_llama2_7b_t0_mix emin2_qwen3_8b_t0_mix)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "=== C3v2c spread ${CFG} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
ART=chrono/artifacts_tier0
LANGS=$(python -c "import yaml;print(' '.join(yaml.safe_load(open('chrono/configs/${CFG}.yaml'))['views']['langs']))")
for S in 1 2 3 4; do for TV in orig all; do
    python -u chrono/scripts/baseline_conditions.py --config "chrono/configs/${CFG}.yaml" \
        --probes ridge --train-views ${TV} --langs ${LANGS} --subsample-seed ${S} \
        --views "${ART}/views.parquet" --corpus "${ART}/corpus_chrono.parquet" \
        --splits-dir "${ART}/splits" --store-root "${ART}/emb_store" \
        --out-dir chrono/reports/tier0/scores \
        || { echo "FAILED spread ${CFG} s${S} ${TV}"; exit 1; }
done; done
commit_push_sandbox "C3v2c: baseline seed spread ${CFG} (job ${SLURM_JOB_ID})" chrono/reports/tier0/scores
echo "=== Done. $(date) ==="
