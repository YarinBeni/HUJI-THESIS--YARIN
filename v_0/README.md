# Akkadian Restoration & Mechanistic Interpretability Repo

This repository contains a complete pipeline for training a transformer-based ancient language restoration model (Akkadian/Sumerian) and analyzing it using Sparse Autoencoders (SAEs) for mechanistic interpretability. The project is designed to run on resource-constrained hardware (e.g., Mac M2 with 18GB RAM).

---

## 📂 Repository Structure

```
lititure-review/
├── data/                        # Raw and processed data storage
│   ├── eBL_fragments.json       # Raw cuneiform fragments (~29k texts)
│   ├── char_vocab.json          # Generated character vocabulary
│   └── restoration_dataset/     # Preprocessed HuggingFace dataset (train/val/test)
├── evaCun_data/                 # Source data/samples from EvaCun paper
│   ├── eBL_fragments.json       # Copy of raw fragments
│   ├── sample_lemmatization.csv # Example lemmatization data
│   └── sample_token_prediction.csv # Example token prediction data
├── models/                      # Model checkpoints and SAE weights
│   ├── torso_restoration/       # Trained RestorationModel checkpoints
│   │   ├── best_model.pt        # Best validation checkpoint (37M params)
│   │   ├── last_model.pt        # Final checkpoint
│   │   └── tl_model.pt          # Converted TransformerLens version
│   └── sae_results/             # Trained SAEs and analysis outputs
│       ├── sae_layer8_optimized.pt # Memory-optimized SAE weights
│       ├── sae_layer8.pt        # (Legacy) Standard SAE weights
│       └── analysis_layer8.json # Feature analysis results
├── papers/                      # Reference literature
│   ├── pdf/                     # Original PDFs
│   └── txt/                     # Extracted text versions
├── src/                         # Source code pipeline
│   ├── 01_build_vocab.py        # Step 1: Build vocabulary from JSON
│   ├── 02_preprocess_dataset.py # Step 2: Clean & mask dataset
│   ├── 03_train_torso.py        # Step 3: Train restoration model
│   ├── 04_convert_to_transformerlens.py # Step 4: Convert format
│   ├── 05_run_sae_analysis.py   # Legacy: High-RAM SAE trainer
│   ├── 06_sae_memory_optimized.py # Recommended: Low-RAM SAE trainer
│   ├── 07_inspect_sae.py        # Analysis: Inspect learned features
│   ├── modeling_restoration.py  # Model architecture definition
│   ├── common_atf.py            # Text cleaning utilities
│   └── run_pipeline.sh          # Shell script to automate steps
├── download_evaCun_dataset.py   # Script to fetch eBL data
├── setup_sae_env.sh             # Environment setup helper
├── environment.yml              # Conda environment specification
├── SAE_TRAINING_GUIDE.md        # Guide for running memory-optimized SAE
├── MODEL_FORMAT_AND_SAELENS_GUIDE.md # Documentation on model formats
├── RESTORATION_PROJECT_PLAN.md  # Original project roadmap
└── PROGRESS.md                  # Chronological progress log
```

---

## 🚀 How to Run the Pipeline

### Prerequisites
1. **Install dependencies** (Python 3.10+):
   ```bash
   pip install torch transformers datasets sae-lens transformer-lens einops tqdm
   ```
   *(Or use the provided `environment.yml`)*

### Step 1: Data Acquisition & Prep
```bash
# 1. Download data (if not already present)
python download_evaCun_dataset.py

# 2. Build vocabulary
python src/01_build_vocab.py --input data/eBL_fragments.json --output data/char_vocab.json

# 3. Preprocess & mask dataset
python src/02_preprocess_dataset.py --fragments data/eBL_fragments.json --vocab data/char_vocab.json --out_dir data/restoration_dataset
```

### Step 2: Train the Restoration Model
```bash
# Train on Mac M2 (MPS) or GPU
python src/03_train_torso.py --dataset data/restoration_dataset --output_dir models/torso_restoration --epochs 1 --batch_size 32
```
*Output: `models/torso_restoration/best_model.pt`*

### Step 3: Train a Sparse Autoencoder (SAE)
To interpret what the model learned in Layer 8 (middle layer):
```bash
# Use the memory-optimized script for laptops
python src/06_sae_memory_optimized.py --layer 8 --max-samples 200 --epochs 50 --batch-size 256 --l1-coef 3e-3 --device cpu
```
*Output: `models/sae_results/sae_layer8_optimized.pt`*

### Step 4: Inspect Learned Features
Analyze the sparse features discovered by the SAE:
```bash
# 1. General histogram check (are features sparse?)
python src/07_inspect_sae.py --histogram --layer 8

# 2. Look for specific linguistic patterns (e.g., 'lugal' = king)
python src/07_inspect_sae.py --pattern 'lugal'

# 3. Inspect a specific feature index
python src/07_inspect_sae.py --feature 1170
```

---

## 📚 Documentation Guide

- **`RESTORATION_PROJECT_PLAN.md`**: The original architectural blueprint.
- **`PROGRESS.md`**: Chronological log of the project's development.
- **`MODEL_FORMAT_AND_SAELENS_GUIDE.md`**: Technical details on `.pt` file formats and TransformerLens integration.
- **`SAE_TRAINING_GUIDE.md`**: Focused guide on running the memory-optimized SAE training.

---

## 🧠 Key Concepts

- **RestorationModel**: A "Torso" (encoder-only) transformer trained to fill in masked characters (`-`) in damaged cuneiform text.
- **Mechanistic Interpretability**: Instead of treating the model as a black box, we use SAEs to decompose its internal activations into interpretable "features" (e.g., a feature that specifically activates for Sumerian genitive markers).
- **Memory Optimization**: The project specifically solves the challenge of running heavy interpretability workloads on consumer hardware (Mac M2) by bypassing standard caching overheads.

