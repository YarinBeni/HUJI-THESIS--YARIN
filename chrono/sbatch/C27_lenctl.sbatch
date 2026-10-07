#!/bin/bash
#SBATCH --job-name=C27_lenctl
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=chrono/sbatch/logs/C27_lenctl_%j.out
# ===========================================================================
# C27 — model-level length control (advisor): replace every non-space char
# with 'a' (word boundaries kept), run the four alignment models over these
# texts at their Fig-B peak layers, and measure how much akk<->eng alignment
# the models produce from length alone. One GPU, models in sequence — the
# fake corpus is ~2.4k short texts, so each pass is minutes.
# ===========================================================================
set -uo pipefail
echo "=== C27 lenctl === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs
python -c "import umap" 2>/dev/null || pip install -q umap-learn 2>&1 | tail -1

python -u chrono/scripts/length_control.py --build || { echo "FAILED build"; exit 1; }

# Fig-B peak layers (B_language_alignment.csv): llama 20, qwen 27, olmo 9, cunei 4.
for ML in "llama2_7b 20" "qwen3_8b 27" "olmo2_7b 9" "thalesian_cunei400m 4"; do
    set -- $ML
    echo "--- extract $1 layer $2 $(date)"
    # --store-root is mandatory here: the default is chrono/artifacts, but
    # the analysis reads the tier0 store (job 53381 wrote to the wrong one).
    python -u chrono/scripts/extract_embeddings.py \
        --model "$1" --table chrono/artifacts_tier0/lenctl.parquet \
        --id-col id --text-col text --layers "$2" --sites mean \
        --store-root chrono/artifacts_tier0/emb_store \
        || { echo "FAILED extract $1"; exit 1; }
done

python -u chrono/scripts/length_control.py --analyze || { echo "FAILED analyze"; exit 1; }
commit_push_sandbox "C27: model-level length control + bin sweep (job ${SLURM_JOB_ID})" chrono/reports/tier0/atlas
echo "=== Done. $(date) ==="
