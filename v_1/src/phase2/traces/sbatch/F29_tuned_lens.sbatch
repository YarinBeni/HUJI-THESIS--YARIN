#!/bin/bash
#SBATCH --job-name=F29_lens
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=v_1/src/phase2/traces/logs/F29_lens_%A_%a.out
# ===========================================================================
# F29 — mid-layer-faithful lens (reviewer request on slides 35/36): per-layer
# tuned-lens translators (Belrose et al. 2023) trained on our own corpus,
# then ALL directions — cellA, cellB (ruler axis, cached by E3b), the
# document-year regression axes (eng/akk), both E1 pairwise directions —
# read out raw AND tuned: token ends + full spectroscopy with 50 random
# directions pushed through the SAME translator. Run AFTER E3b so the cellB
# direction npz exists (the script skips it gracefully if absent).
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b qwen3_8b llama2_7b)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "=== F29 ${METHOD} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/traces/logs

python -u v_1/src/phase2/traces/lens_tuned.py --method "${METHOD}" \
    || { echo "FAILED F29 ${METHOD}"; exit 1; }

commit_push "F29: tuned-lens all-directions ${METHOD} (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/traces/results
echo "=== Done. $(date) ==="
