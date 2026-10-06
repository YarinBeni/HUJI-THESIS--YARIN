#!/bin/bash
#SBATCH --job-name=C23c_olmo_eng
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=chrono/sbatch/logs/C23c_olmo_eng_%j.out
# C23c — OLMo joined the ladder through the akk arm only; give it the English
# baseline (extract + probe) and the eng25 fairness cell, so its language
# cliff is measurable like the others'.
set -uo pipefail
echo "=== C23c olmo english === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
WB=v_1/src/world_models/akkadian
python -u ${WB}/extract_entity.py --method olmo2_7b --entity-type assyrian_ruler \
    || { echo "FAILED extract"; exit 1; }
python -u ${WB}/probe_entity.py --method olmo2_7b --entity-type assyrian_ruler || echo "FAILED probe eng"
python -u ${WB}/probe_entity.py --method olmo2_7b --entity-type assyrian_ruler_eng25 || echo "FAILED probe eng25"
python -u ${WB}/aggregate_entity.py || echo "WARN aggregate"
commit_push_sandbox "C23c: olmo english entity baseline (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/akkadian/results
echo "=== Done. $(date) ==="
