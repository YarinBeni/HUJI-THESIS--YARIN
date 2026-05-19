#!/bin/bash
#SBATCH --job-name=r2_p1b_pv
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=v_1/src/linear_probing/logs/r2_p1b_pv_%j.out

# ===========================================================================
# Round 2 Phase 1b — Framed prompts on full ORCC + activation capture
#   variants: pv0 (no system), pv1, pv2 (few-shot, holdout from balanced),
#             pv3
#   fragments: 1202 (full ORCC corpus)
#   layers:    L0, L15 (Round 1 best), L-1
#
# Pipeline (per variant):
#   1. SHA-256 verify prompt against APPROVED.md
#   2. require Phase 0's draws_matrix.npy (pv2 holdout selection)
#   3. run_pv.py                → direct_answers/<variant>/<fragment_id>.json
#                                 (one JSON per fragment; 1202 files / variant)
#   4. extract_prompted_acts.py → prompted_activations/<variant>/L{NN}.npz
#                                 + layer_{NN}.npz (Round-1 compat) + metadata.json
#                                 for L = 0, 15, -1
#
# Dependencies (must exist before submission):
#   v_1/src/linear_probing/round2_phase1b/run_pv.py                 (W2.C)
#   v_1/src/linear_probing/round2_phase1b/extract_prompted_acts.py  (W2.C)
#   v_1/src/linear_probing/round2_phase1b/pv_parse.py               (W2.C)
#   v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/
#       draws_matrix.npy             (built by Phase 0)
#       corpus_fragment_order.json   (built by Phase 0)
#
# Model resolution: run_pv.py / extract_prompted_acts.py read env
#   QWEN_MODEL_PATH (default in code: "Qwen/Qwen2.5-7B-Instruct"). We leave it
#   unset here so the HF-hub fallback fires; set it before sbatch if you want
#   a local checkpoint.
# ===========================================================================

echo "=== Round 2 Phase 1b — framed prompts + prompted activations ==="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/linear_probing/logs
PROMPT_DIR=v_1/src/linear_probing/results/orcc_round2_phase1b/prompts
OUT_DIR=v_1/src/linear_probing/results/orcc_round2_phase1b
SUBSET_DIR=v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset
DRAWS_MATRIX="$SUBSET_DIR/draws_matrix.npy"
FRAGMENT_ORDER="$SUBSET_DIR/corpus_fragment_order.json"
CORPUS=v_1/data/evaluation/corpora/orcc_corpus.parquet

mkdir -p "$OUT_DIR/direct_answers" "$OUT_DIR/prompted_activations"

# ── Step 1: SHA-256 verify prompts against APPROVED.md ───────────────────────
echo ""
echo "--- Step 1: verify prompt hashes ---"
declare -A EXPECTED_HASH
EXPECTED_HASH[pv0.md]="624708e3e66a31709b244d2f8ddc77d7556628be31c70e40f30bb9ee043e4d01"
EXPECTED_HASH[pv1.md]="1413becf49cdb9cdcf3a3b66135b821871a73d78b8d38e3c9aab8653c80cfa78"
EXPECTED_HASH[pv2.md]="f6397dcd92cb9dadabe9c8e3ce9e8994e440e9f3e17fd5cca02a801e0f2f57df"
EXPECTED_HASH[pv3.md]="db306a0f4cce10dfbc0acb5fe182db11d72d9e004ddec4d106340082788fd5cd"

for f in pv0.md pv1.md pv2.md pv3.md; do
    actual=$(sha256sum "$PROMPT_DIR/$f" | awk '{print $1}')
    expected=${EXPECTED_HASH[$f]}
    if [ "$actual" != "$expected" ]; then
        echo "FAILED: prompt hash mismatch for $f"
        echo "  expected: $expected"
        echo "  actual  : $actual"
        echo "Re-approval required (see APPROVED.md)."
        exit 1
    fi
    echo "  OK  $f  $actual"
done

# ── Step 2: require Phase 0 artifacts (pv2 holdout selection) ────────────────
echo ""
echo "--- Step 2: verify Phase 0 artifacts exist ---"
for f in "$DRAWS_MATRIX" "$FRAGMENT_ORDER"; do
    if [ ! -f "$f" ]; then
        echo "FAILED: $f not found."
        echo "  Phase 1b requires Phase 0 to finish first (pv2 few-shot holdout"
        echo "  selection reads these artifacts to confirm IDs are outside MC draws)."
        echo "  Submit run_p0_probes.sbatch first, then resubmit this job."
        exit 1
    fi
    echo "  OK  $f"
done

# ── Step 3: generate direct answers (each variant × 1202 fragments) ──────────
for VARIANT in pv0 pv1 pv2 pv3; do
    echo ""
    echo "--- $VARIANT: run_pv.py ---"
    # NOTE: run_pv.py writes per-fragment JSON files to
    #   $OUT_DIR/direct_answers/$VARIANT/{fragment_id}.json
    # (it derives that path from --out_dir and --variant; we don't pass a file).
    python -u v_1/src/linear_probing/round2_phase1b/run_pv.py \
        --variant "$VARIANT" \
        --out_dir "$OUT_DIR" \
        --corpus "$CORPUS" \
        --prompts_dir "$PROMPT_DIR" \
        --draws_matrix "$DRAWS_MATRIX" \
        --fragment_order "$FRAGMENT_ORDER" \
        --max_new_tokens 512 \
        || { echo "FAILED: run_pv.py --variant $VARIANT"; exit 1; }
done

# ── Step 4: extract prompted activations (variant × layer) ───────────────────
for VARIANT in pv0 pv1 pv2 pv3; do
    for LAYER in 0 15 -1; do
        echo ""
        echo "--- $VARIANT: extract_prompted_acts.py layer=$LAYER ---"
        # NOTE: extract_prompted_acts.py writes
        #   $OUT_DIR/prompted_activations/$VARIANT/L{NN}.npz       (rich schema)
        #   $OUT_DIR/prompted_activations/$VARIANT/layer_{NN}.npz  (R1 compat)
        #   $OUT_DIR/prompted_activations/$VARIANT/metadata.json
        # where NN is the resolved positive layer index (e.g. -1 -> 28).
        python -u v_1/src/linear_probing/round2_phase1b/extract_prompted_acts.py \
            --variant "$VARIANT" \
            --layer "$LAYER" \
            --out_dir "$OUT_DIR" \
            --corpus "$CORPUS" \
            --prompts_dir "$PROMPT_DIR" \
            --draws_matrix "$DRAWS_MATRIX" \
            --fragment_order "$FRAGMENT_ORDER" \
            || { echo "FAILED: extract_prompted_acts.py --variant $VARIANT --layer $LAYER"; exit 1; }
    done
done

# ── Step 5: push results to GitHub ───────────────────────────────────────────
echo ""
echo "=== Pushing Phase 1b results to GitHub ==="
git add \
    "$OUT_DIR/direct_answers/" \
    "$OUT_DIR/prompted_activations/"
git commit -m "Add Round 2 Phase 1b results: direct answers + prompted activations pv0/1/2/3 (cluster job $SLURM_JOB_ID)" \
    || echo "Nothing new to commit"
git push origin main \
    || echo "WARNING: git push failed — results saved locally."

echo ""
echo "=== Done ==="
echo "End: $(date)"
