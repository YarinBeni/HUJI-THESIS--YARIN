#!/bin/bash
#SBATCH --job-name=J13_maxking_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J13_maxking_probe_%j.out
# ===========================================================================
# J13 — probe the maximal-with-kings activations (CPU). RUN AFTER J12/J12b/J12c.
# For every model with "_maxking_*" acts on disk, runs the 3-site balanced-MC probe
# (year_group + year_strat + ruler_clf control) on the 5-ruler / k=9 king-found
# subset. Emits p1_gurnee_tegmark/results/maxking/p1_maxking__<model>.json.
#   sbatch v_1/src/stress_tests/sbatch/J13_maxking_probe.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J13 maxking probe === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    if ls v_1/src/linear_probing/results/orcc__embed/activations/${M}_maxking_mean/layer_*.npz >/dev/null 2>&1; then
        echo "--- probe $M ---"
        python -u v_1/src/stress_tests/p1_gurnee_tegmark/probe_maxking.py --method "$M" --n-jobs 16 \
            || echo "WARN probe $M failed"
    else
        echo "skip $M (no maxking acts on disk)"
    fi
done
commit_push "J13: maxking probe results (job ${SLURM_JOB_ID})" \
    v_1/src/stress_tests/p1_gurnee_tegmark/results/maxking/*.json
echo "=== Done. $(date) ==="
