#!/bin/bash
#SBATCH --job-name=W3_aggregate
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=v_1/src/world_models/logs/W3_aggregate_%j.out
# ===========================================================================
# W3 — aggregate probe JSONs into summary tables + RESULTS.md + figures. Run last
# (rerunnable any time; uses whatever probe results exist). Also runs locally:
#   python v_1/src/world_models/aggregate_results.py
#   sbatch v_1/src/world_models/sbatch/W3_aggregate.sbatch
# ===========================================================================
set -uo pipefail
echo "=== W3 aggregate === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs

python -u v_1/src/world_models/aggregate_results.py \
    || { echo "FAILED aggregate"; exit 1; }
python -u v_1/src/world_models/plot_results.py \
    || { echo "FAILED plots"; exit 1; }

commit_push "W3: world-models summary tables + figs (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/results/RESULTS.md \
    v_1/src/world_models/results/summary_best_layer_r2.csv \
    v_1/src/world_models/results/summary_best_layer_spearman.csv \
    v_1/src/world_models/results/summary_layerwise.csv \
    v_1/src/world_models/results/figs
echo "=== Done. $(date) ==="
