#!/bin/bash
#SBATCH --job-name=O2_probe_olmo
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/world_models/logs/O2_probe_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# O2 — probe the OLMo activations, CPU only. Runs after O1.
#
#   0 olmo2_7b        1 olmo2_7b_random
#
# Three passes, all with the SAME scripts and settings every other arm used —
# the point of this arm is comparability, so no protocol is invented here:
#   probe_wm.py        cell A, English entity datasets (layer sweep, ridge)
#   probe_eng_pls.py   cell A, best-layer PLS-k
#   probe_entity.py    cell B, entity-level MC (200 draws, 20% entities held out)
#
# NOTE cleanup is OFF for both arms. W2/WB2 passed --cleanup on the random arms
# and deleted their npz, which forced a re-extraction later; not repeating that.
#
# Gate before believing any frequency result: OLMo must behave like the other
# arms on cell A (beat its twin and the TF-IDF floor on salient entities). If it
# does not, the dose-response curve means nothing and step 3 should not run.
#
#   sbatch v_1/src/olmo_frequency/sbatch/O2_probe_olmo.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b olmo2_7b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== O2 probe ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

echo "--- cell A: layer sweep ---"
python -u v_1/src/world_models/probe_wm.py --method "$M" \
    || { echo "FAILED probe_wm ${M}"; exit 1; }

echo "--- cell A: best-layer PLS-k ---"
python -u v_1/src/world_models/probe_eng_pls.py --method "$M" \
    || { echo "FAILED eng_pls ${M}"; exit 1; }

echo "--- cell B: entity-level MC ---"
python -u v_1/src/world_models/akkadian/probe_entity.py --method "$M" \
    || { echo "FAILED probe_entity ${M}"; exit 1; }

commit_push "O2: OLMo probes ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/results/probes/${M} \
    v_1/src/world_models/results/eng_pls/${M} \
    v_1/src/world_models/akkadian/results/probes_entity/${M}
echo "=== Done ${M}. $(date) ==="
