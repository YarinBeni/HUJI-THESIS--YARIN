# Unified Dataset: Source Composition and Characteristics

**Date:** December 2025
**Context:** Thesis / Research Project on Akkadian Language Modeling
**Purpose:** To document the composition, scale, and characteristics of the unified dataset constructed from eBL, ORACC, and Archibab sources.

---

## 1. Dataset Overview

We have constructed a **Unified Akkadian Corpus** by merging data from three primary digital repositories. The dataset is structured at the **token level** (one row per word/token occurrence) to facilitate detailed analysis and modeling.

*   **Total Words:** 2,450,094 words/tokens.
*   **Total Signs:** **4,894,744 individual signs.**
*   **Total Texts:** 40,429 unique text fragments.
*   **Format:** Parquet file with standardized columns for source, ID, positioning, and token representations.

### Comparison to Previous Work (Fetaya et al., 2021)
This dataset represents a significant scaling of available resources for Akkadian NLP. Compared to the dataset used in *Filling the Gaps in Ancient Akkadian Texts* (2021):

| Metric | Paper (2021) | Our Unified Dataset (2025) | Scale Factor |
| :--- | :--- | :--- | :--- |
| **Total Signs** | ~2.3 Million | **4.9 Million** | **> 2x** |
| **Total Words** | ~1.0 Million | **2.45 Million** | **> 2x** |

## 2. Source Distribution

The dataset is dominated by ORACC and eBL, with a smaller contribution from Archibab.

| Source | Word Count | % of Total | Text Count | Avg Words/Text |
| :--- | :--- | :--- | :--- | :--- |
| **ORACC** | 1,385,932 | **56.6%** | 14,210 | 97.5 |
| **eBL** | 998,353 | **40.7%** | 24,909 | 40.1 |
| **Archibab** | 65,809 | **2.7%** | 1,310 | 50.2 |

*   **ORACC (Open Richly Annotated Cuneiform Corpus):** Provides the bulk of the data with longer, well-preserved texts.
*   **eBL (Electronic Babylonian Library):** Contributes the largest number of individual fragments (24k), though they tend to be shorter/more fragmentary. This source was **not included** in the 2021 benchmark.
*   **Archibab:** A smaller, specialized corpus contributing ~2.7% of tokens.

## 3. Data Representations (Columns)

For each token, we maintain three representations:

1.  **`value_raw`:** The original transliteration from the source (e.g., `[a]-na`).
2.  **`value_clean`:** A normalized word form (e.g., `a-na`).
    *   *Note:* Currently **missing for all Archibab data** (contains nulls).
    *   *Note:* Inconsistent creation logic across sources (eBL = pre-cleaned, ORACC = raw copy).
3.  **`value_signs`:** The **sign-level tokenization** (e.g., `a na`).
    *   *Status:* **Available and consistent for ALL sources.**
    *   *Usage:* This is the target column for our "Next Token Prediction" model (see `justification_sign_level_tokenization.md`).

## 4. Key Characteristics

### A. Vocabulary & Tokenization
*   **Sign-Level Vocabulary:** ~16,740 unique signs.
*   **Word-Level Vocabulary:** ~253,262 unique words (highly sparse).
*   **Tokenization Density:** Average of **2.00 signs per word**.
    *   43% of words are 1 sign.
    *   28% of words are 2 signs.

### B. Data Quality & Certainty
*   **SURE Readings:** 95.3% of all tokens are marked as "SURE".
*   **Uncertainty:** ORACC contains the most detailed uncertainty metadata (`BLURRED`, `HAS_DOUBTS`), while eBL and Archibab are mostly binary (Sure/Not).

### C. Overlaps
*   **IDs:** There are **zero** overlapping `fragment_id`s between sources. They use disjoint ID namespaces.
*   **Content:** A content-fingerprint analysis (first 10 words) suggests ~29 potential duplicate texts between eBL and ORACC, but they are treated as distinct in the current dataset.

## 5. Train/Val/Test Splits
The unified dataset has been split to prevent data leakage (no text appears in multiple splits).

*   **Train:** 80% (1.96M words)
*   **Val:** 10% (253k words)
*   **Test:** 10% (235k words)
*   **Leakage Check:** Confirmed 0 overlapping fragment IDs between splits.

---

## Summary for Thesis
"The Unified Akkadian Corpus consolidates 2.45 million tokens and **4.9 million signs** from three major digital libraries: ORACC (57%), eBL (41%), and Archibab (2%). This represents a more than **two-fold increase** in data availability compared to previous state-of-the-art benchmarks (Fetaya et al., 2021). While word-level normalization (`value_clean`) is inconsistent across sources, the sign-level tokenization (`value_signs`) is uniform, enabling a robust training pipeline on the combined ~16,740 unique signs. The data is split 80/10/10 by text ID to ensure rigorous evaluation."
