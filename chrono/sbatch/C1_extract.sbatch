#!/bin/bash
#SBATCH --job-name=C1_extract
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=chrono/sbatch/logs/C1_extract_%j.out
# ===========================================================================
# C1 — P0.3 extraction cache: build the data artifacts if missing (corpus,
# splits, views — all deterministic, so a rebuild is byte-identical), then
# embed every view + corpus original with the Thalesian/AKK_300m encoder
# (hfid from wm_lib registry), layers 0..12, sites mean+last, into the
# gitignored EmbStore. Only the small extraction meta json is copied into
# chrono/reports/ (tracked) and pushed; the npz shards stay cluster-local.
# Submit ONLY after C0 is green.
# ===========================================================================
set -uo pipefail
echo "=== C1 extract === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox
mkdir -p chrono/sbatch/logs

ART=chrono/artifacts
if [ ! -f "${ART}/corpus_chrono.parquet" ]; then
    python -u chrono/scripts/make_corpus.py \
        || { echo "FAILED make_corpus"; exit 1; }
fi
if [ ! -f "${ART}/splits/gkf_ruler.json" ]; then
    python -u chrono/scripts/make_splits.py \
        || { echo "FAILED make_splits"; exit 1; }
fi
if [ ! -f "${ART}/views.parquet" ]; then
    python -u chrono/scripts/make_views.py \
        --corpus "${ART}/corpus_chrono.parquet" \
        --out "${ART}/views.parquet" \
        --audit-out "${ART}/views_audit.parquet" \
        || { echo "FAILED make_views"; exit 1; }
fi

python -u chrono/scripts/extract_embeddings.py \
    --model thalesian_akk300m --layers 0-12 --sites mean last \
    --views "${ART}/views.parquet" --corpus "${ART}/corpus_chrono.parquet" \
    --store-root "${ART}/emb_store" --batch-size 32 \
    || { echo "FAILED extract_embeddings"; exit 1; }

# chrono/artifacts is gitignored: copy the small committable outputs into
# tracked chrono/reports/ before pushing.
mkdir -p chrono/reports
cp "${ART}/emb_store/extract_meta__thalesian_akk300m.json" chrono/reports/
commit_push_sandbox "C1: thalesian extraction cache meta (job ${SLURM_JOB_ID})" \
    chrono/reports
echo "=== Done. $(date) ==="
