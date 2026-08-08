#!/bin/bash
#SBATCH --job-name=F7_fvu
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/phase2/sae/logs/F7_fvu_%j.out
# ===========================================================================
# F7 — E5.1 FVU gate: can the official Qwen-Scope SAE (Qwen3-8B-Base, 64k)
# reconstruct OUR activations (post-trained checkpoint; Akkadian OOD)?
# Downloads the SAE layers from HF on first run; settles the layer-index
# offset empirically; writes the pass/fail verdict E5.2 consumes.
# ===========================================================================
set -uo pipefail
echo "=== F7 fvu === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/sae/logs

python -u v_1/src/phase2/sae/fvu_gate.py || { echo "FAILED fvu gate"; exit 1; }

commit_push "F7: SAE FVU gate (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/sae/results
echo "=== Done. $(date) ==="
