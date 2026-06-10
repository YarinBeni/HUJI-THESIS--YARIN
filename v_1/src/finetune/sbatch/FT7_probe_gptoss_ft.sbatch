#!/bin/bash
#SBATCH --job-name=ft_FT7_gptoss_probe
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=v_1/src/finetune/logs/FT7_%A_%a.out
#SBATCH --array=0-3

# ===========================================================================
# FT7 — extract + probe the 4 gpt-oss-120b LoRA checkpoints (after FT6+FT0).
# Array over cuts (extraction of a 120B per cleaning is hours, so one cut per
# job). Base gpt-oss-120b is loaded bf16 and the LoRA adapter merged in
# memory (--lora-adapter). Methods gpt_oss_120b_ft{00,12,24,32}.
# ===========================================================================
set -uo pipefail

CUTS2=(00 12 24 32)
CUT2="${CUTS2[$SLURM_ARRAY_TASK_ID]}"
echo "=== FT7 gpt-oss-120b FT probing cut${CUT2} ===  Job ${SLURM_JOB_ID:-NA}  $(hostname)  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p v_1/src/finetune/logs v_1/src/finetune/results/probes
python -c "import peft" 2>/dev/null || pip install -q peft

ACTS=v_1/src/linear_probing/results/orcc__embed/activations
SUBSET=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
OUT=v_1/src/finetune/results/probes

ADAPTER=v_1/models/finetune/gpt_oss_120b/cut${CUT2}/best
[ -d "$ADAPTER" ] || { echo "MISSING adapter $ADAPTER — did FT6 finish?"; exit 1; }

for CLEAN in tier0 maximal; do
    python -u v_1/src/linear_probing/03_extract_seal_activations.py \
        --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
        --text-col text_${CLEAN} \
        --model openai/gpt-oss-120b \
        --lora-adapter "$ADAPTER" \
        --pooling mean \
        --batch-size 2 \
        --output-dir ${ACTS}/gpt_oss_120b_ft${CUT2}_${CLEAN}_mean \
        || { echo "FAILED: extract ${CLEAN}"; exit 1; }
done

m=gpt_oss_120b_ft${CUT2}
PROBES="${m}_pls,${m}_cls_numeric,${m}_cls"
for CLEAN in tier0 maximal; do
    TAG=mc_balanced; [ "$CLEAN" = "maximal" ] && TAG=mc_balanced_maximal
    python -u v_1/src/linear_probing/round2_phase0/run_mc_probes.py \
        --draws-matrix   "$SUBSET/draws_matrix.npy" \
        --fragment-order "$SUBSET/corpus_fragment_order.json" \
        --output-dir     "$OUT" \
        --method-tag     "$TAG" \
        --probes         "$PROBES" \
        --cleaning       "$CLEAN" \
        --pooling        mean \
        --layers         all \
        --pls-k          1,2,3,5 \
        --n-jobs         "${SLURM_CPUS_PER_TASK:-32}" \
        || { echo "FAILED: probes ${CLEAN}"; exit 1; }
done

python -u v_1/src/finetune/build_scoreboard.py || echo "WARN scoreboard failed"

git add "$OUT"/*__summary.json v_1/src/finetune/results/scoreboard* 2>/dev/null || true
git commit -m "finetune FT7: gpt-oss-120b ft${CUT2} probe summaries (job ${SLURM_JOB_ID:-NA})" || echo "Nothing to commit"
git pull --rebase origin main || true
git push origin main || echo "WARN push failed"
echo "=== Done. $(date) ==="
