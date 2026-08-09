#!/bin/bash
#SBATCH --job-name=F22_sae2
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/phase2/sae2/logs/F22_sae2_%j.out
# ===========================================================================
# F22 — SAE2 pipeline on the Neuronpedia-labeled Karvonen dictionary:
# step 0 (discover release + layer + offset — BLOCKING, fails loudly if the
# repo can't be found), step 1 (FVU gate on FOUR populations, cell B now
# included), step 2 (feature hunt + cross-population firing), step 4
# (token-level fired-anywhere with position profiles, same pre-registered
# F11 rules), then step 3 (Neuronpedia labels; failure of the label fetch
# does not kill the compute results).
# ===========================================================================
set -uo pipefail
echo "=== F22 sae2 pipeline === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/sae2/logs

python -u v_1/src/phase2/sae2/run_pipeline.py || { echo "FAILED pipeline"; exit 1; }
python -u v_1/src/phase2/sae2/fetch_labels.py --source auto \
    || echo "[warn] label fetch failed — compute results stand, labels can be refetched"

commit_push "F22: SAE2 pipeline on Karvonen dictionary (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/sae2/results
echo "=== Done. $(date) ==="
