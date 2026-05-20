#!/bin/bash
#SBATCH --job-name=r2_p1b_extract_v2
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r2_p1b_extract_v2_%j.out

# ===========================================================================
# Round 2 Phase 1b — Extraction v2: 6 layers x 2 poolings, all 4 variants
#
# Replaces the first extraction (3 layers, last-pooling only) so that the
# Phase 1b -> Round 1 comparison is apples-to-apples for BOTH pooling
# strategies (mean and last) at the same set of layers.
#
# Layers: 0, 4, 10, 15, 22, 28   (Qwen 2.5-7B has 29 hidden states)
# Poolings: last (token before <</FRAG>>), mean (over span)
#
# Output layout (NEW):
#   prompted_activations/{variant}/{pooling}/L{NN}.npz
#   prompted_activations/{variant}/{pooling}/layer_{NN}.npz   (R1-compat)
#   prompted_activations/{variant}/metadata.json
#
# Why one job for all 4 variants: Qwen 7B model load is ~30s; doing it once
# instead of 4 times saves a couple minutes and keeps the activations
# directory layout consistent.
#
# Estimated wall: ~110 min compute (4 variants x 1202 fragments x ~1.4s
# per forward pass) + ~30s model load + push. 4h walltime is comfortable.
# ===========================================================================

echo "=== Round 2 Phase 1b extraction v2 ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs

LAYERS="0,4,10,15,22,28"
OUT_ROOT=v_1/src/linear_probing/results/orcc_round2_phase1b

for VARIANT in pv0 pv1 pv2 pv3; do
    echo ""
    echo "--- extract_prompted_acts.py  variant=$VARIANT  layers=$LAYERS  poolings=last,mean ---"
    python -u v_1/src/linear_probing/round2_phase1b/extract_prompted_acts.py \
        --variant "$VARIANT" \
        --layers "$LAYERS" \
        --out_dir "$OUT_ROOT" \
        --corpus v_1/data/evaluation/corpora/orcc_corpus.parquet \
        --prompts_dir v_1/src/linear_probing/results/orcc_round2_phase1b/prompts \
        --draws_matrix v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy \
        --fragment_order v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json \
        || { echo "FAILED: extract_prompted_acts.py variant=$VARIANT"; exit 1; }
done

echo ""
echo "=== Pushing extraction v2 metadata to GitHub (NPZs are gitignored) ==="
git add v_1/src/linear_probing/results/orcc_round2_phase1b/prompted_activations/*/metadata.json
git commit -m "P1b extract v2: 6 layers x 2 poolings (last,mean) — metadata only" || true
git push origin main || true

echo ""
echo "=== Done ==="
echo "End: $(date)"
