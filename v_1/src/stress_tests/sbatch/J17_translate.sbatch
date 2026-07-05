#!/bin/bash
#SBATCH --job-name=J17_translate
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J17_translate_%j.out
# ===========================================================================
# J17 — Thalesian translation experiment (GPU):
#   1) translate ORCC tier0+maximal -> English with Thalesian/cuneiformBase-400m
#      (umT5 seq2seq; commits translations.parquet + preview).
#      !!! CHECK THE LOG'S "PREFIX PROBE" BLOCK: if the used prefix doesn't give
#      fluent English, set PREFIX below per the model card and resubmit.
#   2) embed the English translations (mean pool, all layers) with
#      qwen3_1b7 / qwen3_8b / random — the comparison set vs their Akkadian rows.
# Probe is J17p (CPU, submit with afterok on this job).
#   sbatch v_1/src/stress_tests/sbatch/J17_translate.sbatch
# ===========================================================================
set -uo pipefail
PREFIX="Translate Akkadian simple transliteration to English: "   # <- adjust per model card if probe looks wrong
echo "=== J17 translate+extract === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

T=v_1/src/stress_tests/translation
python -u $T/translate_thalesian.py --prefix "$PREFIX" --batch-size 16 \
    || { echo "FAIL translate"; exit 1; }
commit_push "J17: Thalesian English translations (job ${SLURM_JOB_ID})" \
    "$T/translations.parquet" "$T/translations_preview.json"

METHODS=(qwen3_1b7 qwen3_8b random)
HFIDS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-8B)
FLAGS=("" "" "--random")
for i in 0 1 2; do
    python -u $T/extract_translation_acts.py --hfid "${HFIDS[$i]}" \
        --method "${METHODS[$i]}" --arch causal ${FLAGS[$i]} \
        || { echo "FAIL extract ${METHODS[$i]}"; exit 1; }
done
echo "=== Done. $(date) ==="
