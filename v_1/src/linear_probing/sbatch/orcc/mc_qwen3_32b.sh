#!/bin/bash
#SBATCH --job-name=r3_e1_mc_32b
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_e1_mc_32b_%j.out
# Submit AFTER qwen3_32b probe job completes (activations must exist).

echo "=== Phase E1: Qwen3-32B balanced MC probes ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc_round2_phase0/probes

python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
    --draws-matrix   v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy \
    --fragment-order v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json \
    --output-dir     v_1/src/linear_probing/results/orcc_round2_phase0/probes \
    --probes         qwen3_32b_pls,qwen3_32b_cls \
    --layers         all \
    || { echo "FAILED: qwen3_32b MC"; exit 1; }

git add v_1/src/linear_probing/results/orcc_round2_phase0/probes/qwen3_32b_*
git commit -m "Phase E1: Qwen3-32B balanced MC probes (job $SLURM_JOB_ID)" || echo "Nothing new to commit"
git push origin main || echo "WARNING: git push failed"

echo ""
echo "=== Done ==="
echo "End: $(date)"
