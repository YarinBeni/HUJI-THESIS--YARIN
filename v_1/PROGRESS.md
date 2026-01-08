# Project Progress & Handover Snapshot

> **Status Date:** December 2024
> **Current Phase:** Phase 1 - Data Understanding & Preparation
> **Working Directory:** `v_1/`

## 🧠 Project Context
We are building a PyTorch-based Masked Language Model (MLM) for Akkadian text restoration, followed by Sparse Autoencoder (SAE) analysis. We are building the repository structure organically—creating files/folders only when needed.

## 📍 Current Status
- **Repository:** `v_1` folder created, legacy code moved to `v_0`.
- **Data:** Raw data (28k+ fragments) located in `v_1/data/raw/extracted/full_corpus_dir/`.
- **Environment:** Conda env `akkadian-v1` (Python 3.10) set up.
- **Exploration:** `01_data_exploration.ipynb` has been created and initial sample analysis (100 fragments) is complete.

## 📊 Key Findings (From Initial EDA)
Based on `notebooks_results/01_data_exploration_result.txt`:
1.  **Volume:** 28,194 CSV fragments total.
2.  **Language:** Data is **93% Akkadian** (3,502/3,759 words in sample).
    - *Decision:* We will likely filter for **Akkadian only** to ensure homogeneity.
3.  **Vocabulary:** Small sample (100 fragments) -> 1,430 unique tokens.
    - *Need:* Must scan full corpus to get true vocab size (estimated ~40k-60k).
4.  **Data Quality:**
    - `clean_value` is the best target for tokenization.
    - `lemma` column is ~30-40% empty/missing.
    - `domain` metadata exists but has some missing values ("see genres.json").
    - `place_discovery`/`place_composition` are mostly NaN in the sample.

## 📝 Next Immediate Tasks (For Developer)

**Goal:** Extend the existing EDA notebook to finalize data understanding before dataset creation.

### Action Item 1: Extend `01_data_exploration.ipynb`
Do **NOT** create a new notebook. Add new cells at the bottom of `01_data_exploration.ipynb` to perform the following:

1.  **Full Vocabulary Scan (Efficiently):**
    - Iterate through ALL 28,194 CSV files.
    - Filter for `word_language == 'AKKADIAN'`.
    - Count unique `clean_value` tokens.
    - *Output:* Total vocabulary size and top 50 tokens.

2.  **Metadata Investigation (Time/Date):**
    - We need to know if temporal metadata exists (e.g., period, date of composition).
    - Check `domain` column: Does it contain period info (e.g., "Old Babylonian", "Neo-Assyrian")?
    - Check `genres.json` (in `data/raw/extracted/`): Does it link fragment IDs to time periods?
    - *Goal:* Determine if we can split data chronologically or control for time period.

3.  **Sequence Length Analysis:**
    - Calculate the length (number of Akkadian words) per fragment.
    - Plot distribution.
    - *Why:* To decide on model context window size.

### Action Item 2: Update Documentation
- After running the analysis, update `README.md` with the new stats (Total Vocab Size, Periodicity findings).

## 🚫 Constraints & Rules
1.  **No empty folders:** Do not create `processed/` or `models/` until we write the code that fills them.
2.  **Notebooks first:** Verify logic in notebooks before writing `.py` scripts.
3.  **Akkadian Focus:** Assume we are proceeding with a monolingual Akkadian model for now.

