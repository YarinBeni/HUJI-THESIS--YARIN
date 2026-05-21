#!/bin/bash
#SBATCH --job-name=orcc_cunei400m_max
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=v_1/src/linear_probing/logs/orcc_cunei400m_maximal_%j.out

echo "=== ORCC Thalesian/cuneiformBase-400m Activations — maximal (mean + last) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
mkdir -p v_1/src/linear_probing/results/orcc__embed/activations

# Mean pooling pass
python -u v_1/src/linear_probing/round2_phase3/extract_enc_activations.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_maximal \
    --model Thalesian/cuneiformBase-400m \
    --max-length 512 \
    --pooling mean \
    --output-dir v_1/src/linear_probing/results/orcc__embed/activations/thalesian_cunei400m_maximal_mean \
    || { echo "FAILED: orcc cunei400m maximal mean"; exit 1; }

# Last-token pooling pass
python -u v_1/src/linear_probing/round2_phase3/extract_enc_activations.py \
    --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
    --text-col text_maximal \
    --model Thalesian/cuneiformBase-400m \
    --max-length 512 \
    --pooling last \
    --output-dir v_1/src/linear_probing/results/orcc__embed/activations/thalesian_cunei400m_maximal_last \
    || { echo "FAILED: orcc cunei400m maximal last"; exit 1; }

# Auto-push results back to the repo
git add v_1/src/linear_probing/results/orcc__embed/activations/thalesian_cunei400m_maximal_mean/ \
        v_1/src/linear_probing/results/orcc__embed/activations/thalesian_cunei400m_maximal_last/
git commit -m "Add ORCC Thalesian/cuneiformBase-400m activations: maximal mean+last (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally"

echo "=== Done ==="
echo "End: $(date)"
