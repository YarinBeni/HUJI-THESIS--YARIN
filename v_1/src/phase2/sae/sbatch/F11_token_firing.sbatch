#!/bin/bash
#SBATCH --job-name=F11_tokfire
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/phase2/sae/logs/F11_tokfire_%j.out
# ===========================================================================
# F11 — gap fix G-lasttoken (the biggest one): F8 measured feature firing on
# the fragment's LAST token only. This reruns the texts through Qwen3-8B and
# encodes EVERY token through the Scope SAE, asking whether the year features
# fire ANYWHERE in a document. Near-zero -> F8's conclusion survives its own
# audit; a jump -> "fires locally, does not propagate" (a different mechanism).
# ===========================================================================
set -uo pipefail
echo "=== F11 token firing === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/sae/logs

python -u v_1/src/phase2/sae/token_firing.py || { echo "FAILED"; exit 1; }

commit_push "F11: SAE token-level firing audit (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/sae/results
echo "=== Done. $(date) ==="
