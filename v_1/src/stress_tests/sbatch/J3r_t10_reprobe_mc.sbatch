#!/bin/bash
#SBATCH --job-name=J3r_t10_mc
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=v_1/src/stress_tests/logs/J3r_t10_mc_%j.out
# ===========================================================================
# J3r (MC) — re-run the T10 reprobe under Monte-Carlo BALANCED draws on the
# prompted activations already extracted by J3a (acts_<model>/prompted_king/,
# gitignored but on cluster NFS). CPU only, no GPU, no re-extraction.
# Runs for whichever models have prompted acts on disk (qwen3 set; gpt-oss only
# if its T10 extraction ever succeeded).
#   sbatch v_1/src/stress_tests/sbatch/J3r_t10_reprobe_mc.sbatch
# ===========================================================================
set -uo pipefail
echo "=== J3r MC T10 reprobe === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
mkdir -p v_1/src/stress_tests/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

RP=v_1/src/stress_tests/redo_t10_prompt/reprobe_king_mc.py
for M in qwen3_1b7 qwen3_8b qwen3_32b gpt_oss_120b; do
    AD=v_1/src/stress_tests/redo_t10_prompt/acts_${M}/prompted_king
    if [ -d "$AD" ] && ls "$AD"/pv*/L*.npz >/dev/null 2>&1; then
        echo "--- T10 MC reprobe: $M ---"
        python -u $RP --acts-dir "$AD" --model "$M" || echo "WARN t10_mc $M failed"
    else
        echo "skip $M (no prompted acts on disk)"
    fi
done
git add v_1/src/stress_tests/redo_t10_prompt/results/*__t10_mc_summary.json 2>/dev/null || true
git commit -m "J3r MC: T10 balanced-MC reprobe (job ${SLURM_JOB_ID})" || echo "nothing to commit"
for _i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 8 + 5)); done
echo "=== Done. $(date) ==="
