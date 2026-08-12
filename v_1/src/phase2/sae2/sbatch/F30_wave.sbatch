#!/bin/bash
#SBATCH --job-name=F30_wave
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/phase2/sae2/logs/F30_wave_%j.out
# ===========================================================================
# F30 — the closing wave, three small reviewer items in one job:
#  1. F25b-a: max-activating cards + clamp generations for the FIRST
#     dictionary (Qwen-Scope TopK k=100, layer 24) -> feature_interp.layer24
#  2. F25b-b: the labeled layer-18 peek (hunt + Neuronpedia labels, FVU
#     caveat welded on) -> l18_peek.{csv,json}
#  3. F23b: amplify/ablate on the RULER-NAME prompts (cell B), same
#     features, controls and frozen read-out -> steer.layer9.cellB.json
# ===========================================================================
set -uo pipefail
echo "=== F30 wave === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/phase2/sae2/logs

python -u v_1/src/phase2/sae2/feature_interp.py --sae1 \
    || { echo "FAILED interp sae1"; exit 1; }

python -u v_1/src/phase2/sae2/l18_peek.py \
    || echo "WARN: l18 peek failed (network-dependent) — continuing"

python -u v_1/src/phase2/sae2/feature_steer.py --population cellB \
    || { echo "FAILED steer cellB"; exit 1; }

commit_push "F30: sae1 interp cards + labeled L18 peek + cellB steering (job ${SLURM_JOB_ID})" \
    v_1/src/phase2/sae2/results
echo "=== Done. $(date) ==="
