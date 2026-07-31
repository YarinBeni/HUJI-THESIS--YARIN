#!/bin/bash
#SBATCH --job-name=WB5d_rand70b
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/WB5d_rand70b_%A_%a.out
#SBATCH --array=0
# ===========================================================================
# WB5 — recover the CELL-B entity activations for the random-init controls. WB2 ran
# them with --cleanup, so their npz were deleted and WB4 (the PLS-k sweep) could not
# run: the controls are the dashed lines the figures are read against. Re-extract with
# cleanup OFF, then run the PLS-k sweep for that arm.
#   0 random (Qwen3-8B rand)  1 llama2_7b_random  2 llama2_13b_random
# NOTE llama2_70b_random needs gpu:4 -> WB5d.
#   sbatch v_1/src/world_models/akkadian/sbatch/WB5_random_recover.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(llama2_70b_random)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "=== WB5d recover ${M} === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
python -c "import sentencepiece" 2>/dev/null || pip install -q sentencepiece
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export WM_MODELS_DIR="${WM_MODELS_DIR:-$HOME/projects/wm_models}"
D=v_1/src/world_models/akkadian
python -u $D/extract_entity.py --method "$M" || { echo "FAILED extract ${M}"; exit 1; }
python -u $D/probe_entity_pls.py --method "$M" || echo "warn: entity_pls ${M}"
commit_push "WB5d: recover ${M} entity acts + PLS-k (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    $D/results/probes_entity_pls
echo "=== Done ${M}. $(date) ==="
