#!/bin/bash
#SBATCH --job-name=r3_rand8b_extract
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r3_rand8b_extract_%j.out
# Round-3 wrap-up C1: random-init Qwen3-8B activations -> the "random" baseline.
# Decision 2026-05-26 (Yarin): the random control is now a random-init Qwen3-8B
# (NOT the old qwen2.5-random twin, whose balanced JSONs were empty). Same
# architecture/tokenizer as Qwen3-8B, randomly initialized weights (fixed seed in
# 03b_extract_random_seal_activations.py). Writes to the canonical "random_*"
# dirs under orcc__embed so the probes pick it up as method=random.
# Submit from repo root on the cluster:  sbatch v_1/src/linear_probing/sbatch/orcc/extract_random_qwen3_8b.sh

echo "=== Round-3 C1: random-init Qwen3-8B activations (tier0+maximal x mean+last) ==="
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARNING: git pull failed"

mkdir -p v_1/src/linear_probing/logs
ACT=v_1/src/linear_probing/results/orcc__embed/activations

for cleaning in tier0 maximal; do
    for pooling in mean last; do
        OUT="$ACT/random_${cleaning}_${pooling}"
        echo ""; echo "--- random qwen3_8b / $cleaning / $pooling -> $OUT ---"
        mkdir -p "$OUT"
        python -u v_1/src/linear_probing/03b_extract_random_seal_activations.py \
            --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
            --text-col "text_${cleaning}" \
            --model Qwen/Qwen3-8B \
            --pooling "$pooling" \
            --output-dir "$OUT" \
            || { echo "FAILED: random qwen3_8b / $cleaning / $pooling"; exit 1; }
    done
done

git add "$ACT"/random_tier0_mean "$ACT"/random_tier0_last \
        "$ACT"/random_maximal_mean "$ACT"/random_maximal_last
git commit -m "Round-3 C1: random-init Qwen3-8B activations (job $SLURM_JOB_ID)" || true
git push origin main || echo "WARNING: git push failed"

echo "=== Done ==="; echo "End: $(date)"
