#!/bin/bash
#SBATCH --job-name=O6_layers_pls
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/O6_layers_pls_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# O6 — the LAST missing piece for OLMo: WAl's per-layer R² curve + best-layer
# PLS-k sweep over the Akkadian fragment activations O4 already extracted.
#
#   0 olmo2_7b        1 olmo2_7b_random
#
# This feeds the two deck slides that read akkadian/results/layers_pls/ —
# "Raw Akkadian, layer by layer" and "Raw Akkadian: the signal saturates" —
# which are the only figures in the deck still missing the OLMo arm. Everything
# else (cells A/B/B'/C, all tables, all design figures) already has it.
#
# CPU only, no re-extraction: it walks the npz O4 left on this filesystem.
# Same script and settings as WAl gave the other ten arms.
#
#   sbatch v_1/src/olmo_frequency/sbatch/O6_layers_pls.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b olmo2_7b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== O6 layers+pls ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python -u v_1/src/world_models/akkadian/probe_layers_pls.py --method "$M" \
    || { echo "FAILED layers_pls ${M}"; exit 1; }

# refuse to report success with nothing to show — the silent-zero lesson from O5
N=$(ls v_1/src/world_models/akkadian/results/layers_pls/${M}/*.json 2>/dev/null | wc -l)
echo "[check] ${M}: ${N} layers_pls files"
[ "$N" -gt 0 ] || { echo "FAILED: ${M} produced no layers_pls files"; exit 1; }

commit_push "O6: per-layer R2 + best-layer PLS-k ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/results/layers_pls/${M}
echo "=== Done ${M}. $(date) ==="
