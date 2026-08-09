#!/bin/bash
#SBATCH --job-name=F19_spec
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/phase2/pairs/logs/F19_spec_%j.out
# ===========================================================================
# F19 — W5.6 + W5.5 in one: run the E1 probe at site=last for the headline
# arms (the "mean pooling is not a real state" robustness), then build the
# specification-curve table + figure from EVERY probe result on disk.
# ===========================================================================
set -uo pipefail
echo "=== F19 last+spec === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/pairs/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

FAILED=0
for M in olmo2_7b olmo2_7b_random qwen3_8b; do
    for V in eng_tier0 akk_maximal; do
        if [ ! -d "v_1/src/world_models/akkadian/activations/${M}/${V}" ]; then
            echo "[skip] ${M}/${V}"; continue
        fi
        python -u v_1/src/phase2/pairs/probe_pairs.py --method "$M" \
            --variant "$V" --site last \
            || { echo "FAILED ${M}/${V}"; FAILED=1; }
    done
done
[ "$FAILED" -eq 0 ] || exit 1

python -u v_1/src/phase2/pairs/spec_curve.py || { echo "FAILED spec"; exit 1; }

commit_push "F19: site=last probes + specification curve (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/pairs/results
echo "=== Done. $(date) ==="
