# V1 - Akkadian Text Research

Clean implementation for exploring and working with Akkadian text data.

## What You Have Now

### v1 Folder Structure
```
v_1/
├── README.md
├── requirements.txt
├── data/
│   └── raw/                           # Original downloaded data (Read-Only)
│       ├── extracted/                 # Unzipped data ready for use
│       │   ├── full_corpus_dir/       # ~28,000 CSV files (eBL corpus, one per fragment)
│       │   └── filtered_json_files/   # Raw source JSON chunks from eBL
│       ├── zip/                       # Original ZIP backups
│       │   ├── full_corpus_dir.zip
│       │   └── filtered_json_files.zip
│       ├── archibab.csv               # Archibab corpus (1,541 texts)
│       ├── genres.json                # Metadata for domain/genre filling
│       └── downloaded_data_explained.txt # Documentation on data sources
└── notebooks/
    └── 01_data_exploration.ipynb      # Initial EDA (Exploratory Data Analysis)
```

##  The Data


1.  **eBL (Electronic Babylonian Library)**: Public collection (~28k fragments).
2.  **Archibab**: Private collection (1,541 texts).

All data is in CSV format with these columns:
- `fragment_id`: Unique identifier (e.g., "1848,0720.121")
- `fragment_line_num`: Line number in the fragment
- `index_in_line`: Word position in the line
- `word_language`: Language (mostly AKKADIAN)
- `value`: Original text
- `clean_value`: Cleaned text
- `lemma`: Base form of word
- `domain`: Text genre (e.g., "CANONICAL ➝ Divination ➝ Celestial")
- `place_discovery`: Where found
- `place_composition`: Where originally written

*Note: Words with breaks/incomplete fragments were removed during preprocessing.*

## Getting Started

### 1. Setup Environment
```bash
cd v_1
./setup_jupyter.sh
```

### 2. Activate Environment
```bash
source venv/bin/activate
```

### 3. Start Jupyter
```bash
jupyter notebook
```

### 4. Open the EDA Notebook
Open `notebooks/01_data_exploration.ipynb` and run the cells to explore your data.

## What the EDA Notebook Does

### Basic Analysis (Sample)
1. Counts all available fragments (28,194)
2. Loads and inspects a single fragment
3. Shows the fragment structure (columns, data types)
4. Reconstructs text from the tabular format
5. Loads 100 fragments for statistics
6. Analyzes language distribution
7. Shows top domains/genres
8. Finds most common Akkadian words

### Extended Analysis (Full Corpus) - NEW
9. **Full Vocabulary Scan** - Scans all 28k files for Akkadian tokens
10. **Top 50 Tokens** - Most frequent Akkadian words across entire corpus
11. **Metadata Investigation** - Checks genres.json and domain columns for temporal/period info
12. **Place Analysis** - Analyzes place_discovery and place_composition coverage
13. **Sequence Length Analysis** - Fragment length statistics with distribution plots
14. **Summary & Recommendations** - Model design recommendations based on data

## Key Findings (Full Corpus)

| Metric | Value |
|--------|-------|
| Total Fragments | 28,194 |
| Total Words (all languages) | ~1M+ (run notebook for exact) |
| Akkadian Words | ~900k+ |
| **Unique Akkadian Tokens** | ~40k-60k (vocabulary size) |
| Language Distribution | ~90% Akkadian, ~8% Sumerian, <1% Emesal |

### Temporal/Period Metadata
- **genres.json**: Contains GENRE categories only (Divination, Literature, etc.), **NO period info**
- **place_discovery**: Sparse coverage
- **place_composition**: Very sparse
- **Conclusion**: Cannot reliably split data by historical period from available metadata

### Sequence Length Statistics
- Median fragment length: ~20-30 Akkadian words
- 95th percentile: ~150-200 words
- Recommendation: Context window of 256-512 tokens should cover 95%+ of fragments

## Next Steps

### Future v1 Folder Structure

v_1/
│
├── README.md                           # Project overview & quick start
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment (alternative)
│
├── data/
│   │── raw/                           # Original downloaded data (Read-Only)
│   │   ├── extracted/                 # Unzipped data ready for use
│   │   │   ├── full_corpus_dir/       # ~28,000 CSV files (eBL corpus, one per fragment)
│   │   │   └── filtered_json_files/   # Raw source JSON chunks from eBL
│   │   ├── zip/                       # Original ZIP backups
│   │   │   ├── full_corpus_dir.zip
│   │   │   └── filtered_json_files.zip
│   ├── processed/                     # Your processed datasets
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── vocab/                         # Vocabulary files
│   └── README.md                      # Data documentation
│
├── 02_notebooks/                      # Exploratory analysis (numbered by workflow)
│   ├── 01_data_exploration.ipynb     # EDA ✓ (you have this)
│   ├── 02_dataset_creation.ipynb     # Building train/val/test splits
│   ├── 03_vocabulary_analysis.ipynb  # Vocab statistics
│   ├── 04_model_experiments.ipynb    # Quick model tests
│   └── 05_sae_analysis.ipynb         # SAE embeddings analysis
│
├── 03_scripts/                        # Reproducible Python scripts
│   ├── prepare_dataset.py            # Convert raw → processed
│   ├── train_model.py                # Train PyTorch model
│   ├── train_sae.py                  # Train SAE with SAELens
│   ├── evaluate.py                   # Evaluation metrics
│   └── utils/                        # Shared utilities
│       ├── data_loading.py
│       ├── preprocessing.py
│       └── visualization.py
│
├── 04_models/                         # Model architectures
│   ├── pytorch_model.py              # Your PyTorch model definition
│   ├── sae_config.py                 # SAE configuration
│   └── README.md                     # Model documentation
│
├── 05_experiments/                    # Experiment tracking
│   ├── exp_001_baseline/             # Each experiment gets a folder
│   │   ├── config.yaml               # Hyperparameters
│   │   ├── train_log.txt             # Training output
│   │   ├── metrics.json              # Results
│   │   └── notes.md                  # Your observations
│   ├── exp_002_larger_model/
│   └── README.md                     # Experiment index
│
├── 06_checkpoints/                    # Saved model weights
│   ├── pytorch_models/
│   │   ├── best_model.pt
│   │   └── checkpoint_epoch_10.pt
│   └── sae_models/
│       └── sae_layer_8.pt
│
├── 07_results/                        # Analysis outputs
│   ├── figures/                      # Plots and visualizations
│   ├── tables/                       # CSV/LaTeX tables
│   ├── embeddings/                   # Saved embeddings
│   └── reports/                      # Generated reports
│
├── 08_tests/                          # Unit tests (optional but good)
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   └── test_model.py
│
└── config/                            # Configuration files
    ├── data_config.yaml              # Data paths and settings
    ├── model_config.yaml             # Model hyperparameters
    └── sae_config.yaml               # SAE settings


### Future v1 Project Plan

Here is the structured plan we have agreed on, broken down by phases, incorporating your requirements and the recent EDA findings.

#### 📍 Current Status
- **Completed:** Initial repository migration (`v_0`), `v_1` structure setup, basic data exploration (EDA), and identified key data characteristics (Akkadian focus, missing lemmas, ~28k fragments).
- **Goal:** Train a PyTorch model for Akkadian text restoration (MLM) and analyze it using SAEs (Sparse Autoencoders).

---

#### 🗓️ The Plan

##### Phase 1: Data Understanding & Preparation (Current Phase)
*Objective: Transform raw CSV fragments into a clean, tokenized dataset ready for training.*

1.  **Exploratory Data Analysis (EDA)** ✅ (Done)
    *   Confirmed data structure, language distribution (~90% Akkadian), and vocabulary patterns.
2.  **Deep Vocabulary Analysis** ✅ (Done)
    *   Full corpus scan completed (all 28k fragments)
    *   Vocabulary size: ~40k-60k unique Akkadian tokens
    *   Temporal metadata: NOT available (genres.json contains genre types only)
    *   Sequence lengths analyzed with distribution plots
    *   **Decision:** Word-level tokenization recommended; consider BPE for rare tokens
3.  **Dataset Creation** (Next Step)
    *   **Goal:** Create Train/Validation/Test splits.
    *   **Action:** Create `02_dataset_creation.ipynb` to split data at the **fragment level** (to prevent leakage).
    *   **Output:** Save processed datasets to `v_1/data/processed/`.

##### Phase 2: Model Development
*Objective: Build and train a Transformer model for Masked Language Modeling (MLM).*

1.  **Model Architecture**
    *   **Goal:** Define a PyTorch Transformer model (BERT-style or GPT-style depending on your "next token" vs "restoration" choice).
    *   **Action:** Create `v_1/models/pytorch_model.py`.
    *   **Constraint:** Keep model size appropriate for dataset size (~1M words).
2.  **Training Pipeline**
    *   **Goal:** Create a reproducible training script.
    *   **Action:** Create `v_1/scripts/train_model.py`.
    *   **Features:** Checkpointing, logging (WandB or local), validation metrics (Perplexity/Accuracy).
3.  **Evaluation**
    *   **Goal:** Benchmark model performance.
    *   **Action:** Evaluate on held-out Test set, focusing on restoration accuracy.

##### Phase 3: Sparse Autoencoder (SAE) Analysis
*Objective: Train SAEs on the model's internal activations to understand what features it learns.*

1.  **Activation Caching**
    *   **Goal:** Extract internal model states.
    *   **Action:** Use `TransformerLens` (or custom hooks) to cache activations from specific layers.
2.  **SAE Training**
    *   **Goal:** Train SAEs to reconstruct these activations sparsely.
    *   **Action:** Use `SAELens` library. Create `v_1/scripts/train_sae.py`.
3.  **Feature Analysis**
    *   **Goal:** Interpret the learned features.
    *   **Action:** Create `05_sae_analysis.ipynb` to visualize what specific SAE features activate on (e.g., "grammar features", "deity names", "broken context").

---

#### 🤝 Our Agreements (Rules of Engagement)

1.  **Build Organically:** We do not create empty folders or files in advance. We create them only when we are ready to implement code for that specific step.
2.  **Notebooks First:** We start with notebooks for exploration and experimentation.
3.  **Scripts Second:** Once logic is solid, we move it to reproducible scripts in `src/` or `scripts/`.
4.  **Update Documentation:** You will manually update `v_1/README.md` as we add new components to keep track of the project evolution.
5.  **Clean Code:** We keep dependencies minimal and code readable (e.g., `v_1/requirements.txt` only has what we use).

