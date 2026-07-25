#!/bin/bash
#SBATCH --job-name=WB3_agg
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/world_models/logs/WB3_agg_%j.out
# ===========================================================================
# WB3 — roll every WB2 probe JSON up into the slide tables. Run last.
#   sbatch v_1/src/world_models/akkadian/sbatch/WB3_aggregate_entity.sbatch
# ===========================================================================
set -uo pipefail
echo "=== WB3 aggregate === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs

python -u v_1/src/world_models/akkadian/aggregate_entity.py \
    || { echo "FAILED aggregate"; exit 1; }

commit_push "WB3: cell-B entity summary tables (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/akkadian/results/summary_entity_*.csv \
    v_1/src/world_models/akkadian/results/RESULTS_entity.md
echo "=== Done. $(date) ==="
