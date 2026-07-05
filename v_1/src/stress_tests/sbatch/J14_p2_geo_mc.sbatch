#!/bin/bash
#SBATCH --job-name=J14_p2_geo_mc
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=v_1/src/stress_tests/logs/J14_p2_geo_mc_%A_%a.out
# ===========================================================================
# J14 — P2 geography under SITE-BALANCED Monte-Carlo (CPU). One model per array
# task; each runs tier0 + maximal from on-disk mean acts. Draws (10 merged sites
# x k=21 x 200, coordinate-merged provenances) are committed in
# balanced_subset_sites/ (built by build_site_balanced_subset.py).
#   sbatch v_1/src/stress_tests/sbatch/J14_p2_geo_mc.sbatch
# ===========================================================================
set -uo pipefail
MODELS=(qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random)
M=${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "=== J14 P2 site-MC [$M] === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

python -u v_1/src/stress_tests/p2_godey_geography/probe_p2_mc.py --method "$M" --n-jobs 16 \
    || { echo "FAIL p2_mc $M"; exit 1; }
commit_push "J14: P2 site-balanced MC $M (job ${SLURM_JOB_ID})" \
    "v_1/src/stress_tests/p2_godey_geography/results/mc/p2_geo_mc__${M}.json"
echo "=== Done [$M]. $(date) ==="
