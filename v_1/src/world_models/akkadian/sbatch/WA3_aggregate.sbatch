#!/bin/bash
#SBATCH --job-name=WA3_aggregate
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WA3_aggregate_%j.out
# ===========================================================================
# WA3 — aggregate Akkadian probe JSONs into per-(target x ruler_set) tables +
# RESULTS_akk.md. Rerunnable; also runs locally:
#   python v_1/src/world_models/akkadian/aggregate_akk.py
#   sbatch v_1/src/world_models/akkadian/sbatch/WA3_aggregate.sbatch
# ===========================================================================
set -uo pipefail
echo "=== WA3 aggregate === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs

python -u v_1/src/world_models/akkadian/aggregate_akk.py \
    || { echo "FAILED aggregate"; exit 1; }
commit_push "WA3: akk summary tables (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/akkadian/results/RESULTS_akk.md \
    v_1/src/world_models/akkadian/results/summary_year_best_test_r2.csv \
    v_1/src/world_models/akkadian/results/summary_year_best_test_spearman.csv \
    v_1/src/world_models/akkadian/results/summary_geo_best_test_r2.csv \
    v_1/src/world_models/akkadian/results/summary_geo_best_test_spearman.csv
echo "=== Done. $(date) ==="
