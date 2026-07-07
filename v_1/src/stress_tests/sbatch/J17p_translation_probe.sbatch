#!/bin/bash
#SBATCH --job-name=J17p_translation_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J17p_translation_probe_%j.out
# ===========================================================================
# J17p — probe the English-translation embeddings (CPU). RUN AFTER J17.
# YEAR: 8-ruler balanced MC; GEO: 10-site balanced MC. PLS(best-k)+Ridge;
# metric = Spearman only. Results: translation/results/trans_mc__<method>.json.
#   sbatch --dependency=afterok:<J17-id> v_1/src/stress_tests/sbatch/J17p_translation_probe.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J17p translation probe === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b thalesian_akk300m thalesian_cunei400m umt5_base random; do
    python -u v_1/src/stress_tests/translation/probe_translation_mc.py --method "$M" --n-jobs 16 \
        || echo "WARN trans probe $M failed"
done
commit_push "J17p: translation-probe results (job ${SLURM_JOB_ID})" \
    v_1/src/stress_tests/translation/results/*.json
echo "=== Done. $(date) ==="
