#!/bin/bash
#SBATCH --job-name=C3v2_emin
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-44
#SBATCH --output=chrono/sbatch/logs/C3v2_emin_%A_%a.out
# ===========================================================================
# C3v2 — E-MIN v2 grid: 9 configs (3 encoders x {akk, eng, mix}) x 5 seeds,
# one array task per (config, seed), 5 gkf folds inside. Task 0 of each
# config additionally runs the frozen-probe gate and the baselines-on-
# conditions for that encoder, so every arm ships with its own comparison.
# Reads chrono/artifacts_tier0 (C1v2). Needs: afterok on the C1v2 array.
# ===========================================================================
set -uo pipefail
CONFIGS=(emin2_llama2_7b_t0_akk emin2_llama2_7b_t0_eng emin2_llama2_7b_t0_mix
         emin2_qwen3_8b_t0_akk  emin2_qwen3_8b_t0_eng  emin2_qwen3_8b_t0_mix
         emin2_cunei400m_t0_akk emin2_cunei400m_t0_eng emin2_cunei400m_t0_mix)
CI=$(( SLURM_ARRAY_TASK_ID / 5 )); SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))
CFG=${CONFIGS[$CI]}
echo "=== C3v2 ${CFG} seed ${SEED} === job ${SLURM_JOB_ID} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
source chrono/sbatch/_sandbox.sh
sync_sandbox || { echo "FAILED sync_sandbox — refusing to run stale code"; exit 1; }
enable_log_push
mkdir -p chrono/sbatch/logs

ART=chrono/artifacts_tier0
COMMON=(--corpus "${ART}/corpus_chrono.parquet" --views "${ART}/views.parquet"
        --ruler-table "${ART}/ruler_table.parquet" --store-root "${ART}/emb_store"
        --splits-dir "${ART}/splits")

for FOLD in 0 1 2 3 4; do
    python -u chrono/scripts/train_cjb.py --config "chrono/configs/${CFG}.yaml" \
        --seed "${SEED}" --fold "${FOLD}" "${COMMON[@]}" \
        || { echo "FAILED train_cjb ${CFG} seed ${SEED} fold ${FOLD}"; exit 1; }
done

mkdir -p chrono/reports/tier0/scores
cp chrono/artifacts/scores/"${CFG}"-s"${SEED}"-*.parquet chrono/reports/tier0/scores/ 2>/dev/null || true

if [ "${SEED}" -eq 0 ]; then
    # frozen-probe comparison for this encoder/arm, same folds and read-out
    MODEL=$(python -c "import yaml;print(yaml.safe_load(open('chrono/configs/${CFG}.yaml'))['features']['model'])")
    LAYER=$(python -c "import yaml;print(yaml.safe_load(open('chrono/configs/${CFG}.yaml'))['features']['layer'])")
    LANGS=$(python -c "import yaml;print(' '.join(yaml.safe_load(open('chrono/configs/${CFG}.yaml'))['views']['langs']))")
    python -u chrono/scripts/baseline_conditions.py --config "chrono/configs/${CFG}.yaml" \
        --probes ridge pls --langs ${LANGS} --views "${ART}/views.parquet" \
        --corpus "${ART}/corpus_chrono.parquet" --splits-dir "${ART}/splits" \
        --store-root "${ART}/emb_store" --out-dir chrono/reports/tier0/scores \
        || echo "WARN baseline_conditions failed for ${CFG}"
    for LG in ${LANGS}; do
        python -u chrono/scripts/run_baseline_gate.py --model "${MODEL}" \
            --layers "$(python -c "import json;print(','.join(str(l) for l in json.load(open('${ART}/emb_store/extract_meta__'+{'NousResearch/Llama-2-7b-hf':'llama2_7b','Qwen/Qwen3-8B':'qwen3_8b','Thalesian/cuneiformBase-400m':'thalesian_cunei400m'}['${MODEL}']+'.json'))['layers']))")" \
            --sites mean last --lang "${LG}" --apriori-layer "${LAYER}" \
            --corpus "${ART}/corpus_chrono.parquet" --splits-dir "${ART}/splits" \
            --store-root "${ART}/emb_store" \
            --report-out "chrono/reports/tier0/gate_${CFG}_${LG}.txt" \
            || echo "WARN gate failed for ${CFG}/${LG}"
    done
fi
cp chrono/artifacts/results.parquet chrono/reports/results.parquet
commit_push_sandbox "C3v2: ${CFG} seed ${SEED} (job ${SLURM_JOB_ID})" chrono/reports
echo "=== Done. $(date) ==="
