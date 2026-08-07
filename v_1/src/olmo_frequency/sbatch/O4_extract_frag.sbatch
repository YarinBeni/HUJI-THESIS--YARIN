#!/bin/bash
#SBATCH --job-name=O4_frag_extract
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=v_1/src/world_models/akkadian/logs/O4_frag_%A_%a.out
#SBATCH --array=0-5
# ===========================================================================
# O4 — the half of the matrix OLMo is missing: cells B' and C, the FRAGMENT cells.
#
# O1/O2 gave OLMo cells A and B (entity names). Every other arm also has 24 result
# files under akkadian/results/probes/; OLMo has none. Those files are cells B' and C,
# which is where the collapse the thesis is about actually happens — so until this
# runs, OLMo says nothing at all about the deck's central claim.
#
# Three text variants x two arms. Same extractor, same --max-tokens as every other
# arm, so the numbers are comparable rather than merely similar:
#
#   0 olmo2_7b/akk_maximal          3 olmo2_7b_random/akk_maximal
#   1 olmo2_7b/eng_maximal          4 olmo2_7b_random/eng_maximal
#   2 olmo2_7b/eng_tier0            5 olmo2_7b_random/eng_tier0
#
# eng_tier0 is the long literal gloss (~1.9k chars) and is extracted at 512 tokens,
# matching WA1t; the two maximal variants use WA1's 256.
#
#   sbatch v_1/src/olmo_frequency/sbatch/O4_extract_frag.sbatch
# ===========================================================================
set -uo pipefail
METHODS=(olmo2_7b olmo2_7b_random)
VARIANTS=(akk_maximal eng_maximal eng_tier0)
M="${METHODS[$((SLURM_ARRAY_TASK_ID/3))]}"; V="${VARIANTS[$((SLURM_ARRAY_TASK_ID%3))]}"
# the long gloss needs the same 512-token window WA1t gave every other arm
MAXTOK=256; [ "$V" = "eng_tier0" ] && MAXTOK=512

echo "=== O4 extract ${M}/${V} (max_tokens=${MAXTOK}) === job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
python -c "import sentencepiece" 2>/dev/null || pip install -q sentencepiece
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/akkadian/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u v_1/src/world_models/akkadian/extract_akk.py --method "$M" --variant "$V" \
    --max-tokens "$MAXTOK" \
    || { echo "FAILED extract ${M}/${V}"; exit 1; }

commit_push "O4: akk acts ${M}/${V} (job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})" \
    v_1/src/world_models/akkadian/activations/${M}/${V}/metadata.json
echo "=== Done ${M}/${V}. $(date) ==="
