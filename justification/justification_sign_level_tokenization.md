# Justification for Sign-Level Tokenization Strategy

**Date:** December 2025
**Context:** Thesis / Research Project on Akkadian Language Modeling
**Decision:** Train the Next Token Prediction model on **individual signs** (splitting `value_signs`) rather than whole words (`value_clean` or `value_raw`).

---

## 1. Primary Justification: State-of-the-Art Precedent (EvaCun 2025)
The most recent and relevant benchmark for our task, the **EvaCun 2025 Shared Task**, explicitly adopts this approach. Aligning with their methodology makes our results comparable to the current state-of-the-art.

*   **Source:** *EvaCun 2025 Shared Task: Lemmatization and Token Prediction for Cuneiform Languages* (Paper `2025.alp-1.33.pdf`).
*   **Methodology:** In Section 6.2.2, the authors describe their "BabyLemmatizer" baseline model for token prediction:
    > "We segment the input using BabyLemmatizer’s logo-syllabic tokenizer using **transliterated signs as minimal units**, and generate the output sequence similarly... [The model] relies purely on **sign-to-sign relations**."
*   **Implication:** They treat the "sign" (e.g., `a`, `na`, `lugal`) as the atomic unit of the language model, not the "word" (e.g., `a-na`, `lugal-e`).

## 2. Secondary Justification: Task Definition in Prior Work
Previous foundational work in Akkadian NLP also frames the restoration task at the sign level.

*   **Source:** *Reading Lost Sentences* (Gordin et al., 2021, `2021.emnlp-main.384.pdf`).
*   **Task Definition:**
    > "The task is to predict **missing signs**... The model's output should be the specific **sequence of signs** that fills the gap."
*   **Reasoning:** Since physical damage to tablets often destroys individual signs (or parts of them) rather than whole semantic words, predicting signs is the most "natural" task for the domain.

## 3. Technical Justification: Vocabulary Efficiency & Sparsity
Our own Exploratory Data Analysis (EDA) confirms that word-level modeling would suffer from extreme sparsity, whereas sign-level modeling is efficient.

*   **Observation from EDA:**
    *   **Word Vocabulary (`value_raw`):** ~253,262 unique tokens.
        *   *Issue:* Extremely sparse. A huge "long tail" of words appears only once or twice. Neural models struggle to learn embeddings for rare tokens ("OOV" or "Unknown" token problem).
    *   **Sign Vocabulary (`value_signs` split):** ~16,740 unique tokens.
        *   *Advantage:* This is an optimal vocabulary size for modern Transformers (comparable to BERT's ~30k subwords).
*   **Benefit:** By limiting the vocabulary to ~16k signs, the model sees every token thousands of times during training. It can then **compose** complex words it has never seen before by predicting the sequence `sign1` -> `sign2` -> `sign3`, which is impossible for a word-level model that treats a new word as an `<UNK>` token.

## 4. Practical Constraint: Inconsistent "Clean" Data
A practical data engineering constraint further forces this choice.

*   **The Problem:** The `value_clean` column (normalized words) is **completely missing** for the **Archibab** dataset (~65k words).
*   **The Solution:** The `value_signs` column is present and consistent across all three data sources (eBL, ORACC, Archibab).
*   **Conclusion:** Using sign-level tokens allows us to utilize the **entire unified dataset** (2.45M words) for training. Using word-level `value_clean` would force us to discard Archibab.

---

## Summary for Thesis
"We chose to model the Akkadian language at the **sign level** rather than the **word level**. This decision is justified by three factors:
1.  **Linguistic Structure:** It mirrors the morphosyllabic nature of the cuneiform writing system, where signs are the atomic components.
2.  **SOTA Alignment:** It follows the methodology established in the EvaCun 2025 Shared Task.
3.  **Data Efficiency:** It reduces the vocabulary from a sparse ~250k words to a dense ~16k signs, enabling robust embedding learning and allowing the model to generate previously unseen word forms."

