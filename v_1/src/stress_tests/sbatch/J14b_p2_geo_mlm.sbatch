#!/bin/bash
#SBATCH --job-name=J14b_p2_geo_mlm
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J14b_p2_geo_mlm_%j.out
# ===========================================================================
# J14b — P2 geography balanced-MC for the MLM (it was never geo-probed; the
# deck's roster now includes it on the geo slides). Same protocol as J14:
# 200 site-balanced draws, GroupKFold-by-merged-site, PLS + Ridge, gc-km.
# tier0 + maximal from the on-disk MLM mean activations. CPU.
#   sbatch v_1/src/stress_tests/sbatch/J14b_p2_geo_mlm.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J14b P2 geo MLM === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

D=v_1/src/stress_tests/p2_godey_geography
python -u $D/probe_p2_mc.py --method mlm --n-jobs 16 \
    || { echo "FAIL p2 geo mlm"; exit 1; }
commit_push "J14b P2: geography MC for MLM (job ${SLURM_JOB_ID})" \
    "$D/results/mc/p2_geo_mc__mlm.json"
echo "=== Done. $(date) ==="
