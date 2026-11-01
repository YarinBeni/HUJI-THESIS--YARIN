#!/bin/bash
#SBATCH --job-name=C3v2b_base
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#SBATCH --output=chrono/sbatch/logs/C3v2b_base_%A_%a.out
# ===========================================================================
# C3v2b — the "ridge on ALL views" arm for E-MIN v2 (CPU). The head trains on
# every augmented view; a ridge that also sees every view separates what the
# Barlow objective adds from what plain data augmentation adds. One task per
# (encoder, language arm). Reads chrono/artifacts_tier0 (C1v2).
# ===========================================================================
set -uo pipefail
CONFIGS=(emin2_llama2_7b_t0_akk emin2_llama2_7b_t0_eng emin2_llama2_7b_t0_mix
         emin2_qwen3_8b_t0_akk  emin2_qwen3_8b_t0_eng  emin2_qwen3_8b_t0_mix
         emin2_cunei400m_t0_akk emin2_cunei400m_t0_eng emin2_cunei400m_t0_mix)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "=== C3v2b ${CFG} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs chrono/reports/tier0/scores
ART=chrono/artifacts_tier0
LANGS=$(python -c "import yaml;print(' '.join(yaml.safe_load(open('chrono/configs/${CFG}.yaml'))['views']['langs']))")
python -u chrono/scripts/baseline_conditions.py --config "chrono/configs/${CFG}.yaml" \
    --probes ridge --train-views all --langs ${LANGS} \
    --views "${ART}/views.parquet" --corpus "${ART}/corpus_chrono.parquet" \
    --splits-dir "${ART}/splits" --store-root "${ART}/emb_store" \
    --out-dir chrono/reports/tier0/scores \
    || { echo "FAILED baseline all-views ${CFG}"; exit 1; }
commit_push_sandbox "C3v2b: ridge-all-views ${CFG} (job ${SLURM_JOB_ID})" chrono/reports/tier0/scores
echo "=== Done. $(date) ==="
