#!/bin/bash
# run_pipeline.sh - Executes the full data prep and training pipeline.

# Ensure script is run from the project root directory (lititure-review/)
# This ensures that paths like 'data/' and 'src/' resolve correctly.
if [ ! -f "RESTORATION_PROJECT_PLAN.md" ]; then
    echo "Error: This script must be run from the project root directory ('lititure-review/')."
    exit 1
fi

set -e  # Exit immediately if a command exits with a non-zero status.

echo "--- Starting Akkadian Restoration Project Pipeline ---"

# --- Environment Setup ---
echo "Step 0: Activating conda environment 'evaCun'..."
# Note: In some setups, conda activate might not work directly in scripts.
# If this fails, run 'conda activate evaCun' in your terminal first, then run this script.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate evaCun
echo "Environment activated."
echo ""

# --- Step 1: Character Vocabulary (Audit) ---
echo "Step 1: Building character statistics (auditing corpus)..."
python src/01_build_vocab.py \
  --input data/eBL_fragments.json \
  --output data/char_vocab.json
echo "Step 1 complete. Stats saved to data/char_vocab.json"
echo ""

# --- Step 2: Preprocess Dataset ---
echo "Step 2: Preprocessing and generating masked dataset..."
# Clean up any previous dataset to ensure a fresh build
if [ -d "data/restoration_dataset" ]; then
    echo "Removing existing dataset at data/restoration_dataset..."
    rm -rf data/restoration_dataset
fi
python src/02_preprocess_dataset.py \
  --fragments data/eBL_fragments.json \
  --out_dir data/restoration_dataset \
  --seed 42
echo "Step 2 complete. Masked dataset saved to data/restoration_dataset/"
echo ""

# --- Step 3: Train Custom Torso Model (Small Test Run) ---
echo "Step 3: Training the custom torso model (small test run)..."
# Clean up any previous model checkpoints
if [ -d "models/torso_restoration" ]; then
    echo "Removing existing model checkpoints at models/torso_restoration..."
    rm -rf models/torso_restoration
fi
echo "DEBUG: About to run training script..."
python src/03_train_torso.py \
  --dataset data/restoration_dataset \
  --output_dir models/torso_restoration \
  --epochs 100 \
  --batch_size 32  # Device auto-detected (MPS on Mac, CUDA on NVIDIA, CPU fallback)
echo "DEBUG: Training script finished, checking exit code: $?"
echo "Step 3 complete. Test model checkpoint saved to models/torso_restoration/"
echo ""

# --- Pipeline Completion ---
echo "--- ✅ Pipeline finished successfully! ---"
echo "To run a full training, edit 'src/run_pipeline.sh' and remove the --max_*_examples flags."
