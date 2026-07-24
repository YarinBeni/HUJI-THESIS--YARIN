#!/bin/bash
#SBATCH --job-name=Wgpt_eng_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=v_1/src/world_models/logs/Wgpt_eng_probe_%j.out
# ===========================================================================
# Wgpt_eng_probe — re-probe gpt-oss-120B ENGLISH on CPU only, after its extraction
# landed. Same fp16-overflow fix as the Akkadian side; no gpu:8 re-extraction.
#   sbatch v_1/src/world_models/sbatch/Wgpt_eng_probe.sbatch
# ===========================================================================
set -uo pipefail
M=gpt_oss_120b
echo "=== Wgpt_eng_probe ${M} === job ${SLURM_JOB_ID} $(hostname) $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source v_1/src/stress_tests/sbatch/_common.sh
sync_main
mkdir -p v_1/src/world_models/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u v_1/src/world_models/probe_wm.py      --method "$M" || echo "warn: probe_wm ${M}"
python -u v_1/src/world_models/probe_eng_pls.py --method "$M" || echo "warn: eng_pls ${M}"
commit_push "Wgpt_eng_probe: gpt-oss-120B English probes (job ${SLURM_JOB_ID})" \
    v_1/src/world_models/results/probes/${M} \
    v_1/src/world_models/results/eng_pls/${M}
echo "=== Done ${M}. $(date) ==="
