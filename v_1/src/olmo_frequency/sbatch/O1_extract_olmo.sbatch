#!/bin/bash
#SBATCH --job-name=O1_extract_olmo
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/world_models/logs/O1_extract_%A_%a.out
#SBATCH --array=0-1
# ===========================================================================
# O1 — extract OLMo activations for BOTH cells, using the existing extractors
# unchanged. The value of this arm is comparability, so nothing about the
# protocol differs from the other 15 arms; only the model is new.
#
#   0 olmo2_7b          the trained arm
#   1 olmo2_7b_random   its random-init twin — the control that decides whether
#                       any later frequency correlation is about learning
#
# Runs cell A (English entity datasets, extract_acts.py) then cell B
# (entity-level, extract_entity.py, 4 pooling sites). Cell B is only ~354 short
# rows so it is minutes; cell A is the longer half.
#
# PREREQ: step 1 must have passed —
#   python v_1/src/olmo_frequency/smoke_test_tokenizer.py --model olmo2_7b
# It confirms OLMo's tokenizer gives usable entity spans. Without that, cell B
# would silently pool over the wrong tokens.
#
#   sbatch v_1/src/olmo_frequency/sbatch/O1_extract_olmo.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b olmo2_7b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"

echo "=== O1 extract ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -c "import transformers,sys; v=transformers.__version__; \
print('transformers', v); sys.exit(0 if tuple(int(x) for x in v.split('.')[:2]) >= (4,47) else 1)" \
    || { echo "FAILED: OLMo 2 needs transformers >= 4.47"; exit 1; }

echo "--- cell A (English entity datasets) ---"
python -u v_1/src/world_models/extract_acts.py --method "$M" \
    || { echo "FAILED cellA extract ${M}"; exit 1; }

echo "--- cell B (entity-level, 4 pooling sites) ---"
python -u v_1/src/world_models/akkadian/extract_entity.py --method "$M" \
    || { echo "FAILED cellB extract ${M}"; exit 1; }

commit_push "O1: OLMo activations ${M} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/activations/${M}/*/metadata.json
echo "=== Done ${M}. $(date) ==="
