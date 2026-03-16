# Justification for Selecting Masked Language Modeling (MLM) over Next-Token Prediction

## Executive Summary
This document outlines the reasoning behind selecting **Masked Language Modeling (MLM)** (e.g., BERT-style) over **Causal Language Modeling** (e.g., GPT-style next-token prediction) for the task of restoring ancient Akkadian texts. The decision is driven by the specific nature of the restoration task, architectural precedents set by state-of-the-art multilingual models like MMBERT, and empirical evidence showing the superior performance of MLM in low-resource ancient language settings.

## 1. Alignment with the Restoration Task

The primary goal of our project is to restore missing signs or words in damaged cuneiform tablets—a task philologists refer to as "filling in the gaps."

*   **Bidirectional Context:** Unlike text generation, which is strictly left-to-right, text restoration is inherently **bidirectional**. When a scholar restores a broken tablet, they rely on context from *both* the preceding and succeeding text.
*   **Task Formalization:** As established in *Filling the Gaps in Ancient Akkadian Texts*, the restoration task is a direct formalization of the MLM objective: predicting masked tokens given their surrounding context.
*   **Empirical Validation:** Previous research demonstrates that this formulation allows models to effectively use global context, which is critical when the local context (immediate neighbors) might also be damaged or ambiguous.

> "The paper's core idea is that this scholarly task directly corresponds to the **Masked Language Modeling (MLM)** objective used in modern NLP."  
> — *Filling the Gaps in Ancient Akkadian Texts* (2021)

## 2. Architectural Precedents: MMBERT and ModernBERT

Our model architecture follows the advancements in **MMBERT** (A Modern Multilingual Encoder), which itself is based on **ModernBERT**.

*   **Encoder-Only Architecture:** MMBERT is explicitly an **encoder-only** model designed for tasks like classification and retrieval, utilizing an MLM objective.
*   **State-of-the-Art Performance:** MMBERT demonstrates that encoder-only MLM models can outperform significantly larger decoder-only (next-token prediction) models on understanding tasks. For instance, MMBERT Small often outperforms larger generative models in classification benchmarks.
*   **Inverse Masking Schedule:** MMBERT utilizes a novel "inverse masking rate learning schedule" (decaying from 30% to 5% masking), optimizing the model's ability to learn from multilingual data efficiently. Adopting MLM allows us to leverage these specific training innovations.

> "Encoder-only languages models are frequently used for a variety of standard machine learning tasks... We introduce MMBERT, an encoder-only language model pretrained on 3T tokens... [using] an inverse mask ratio schedule."  
> — *MMBERT: A Modern Multilingual Encoder with Annealed Language Learning*

## 3. Superiority in Low-Resource Settings

Ancient languages like Akkadian are extremely low-resource (approx. 1 million tokens available). MLM has distinct advantages in this regime:

*   **Multilingual Transfer:** The *Filling the Gaps* paper showed that a **zero-shot Multilingual BERT (M-BERT)**—which was never trained on Akkadian but used MLM on other languages—outperformed a monolingual BERT trained *only* on Akkadian.
*   **Data Efficiency:** The bidirectional nature of MLM allows the model to learn richer representations from small amounts of data compared to the unidirectional constraint of causal language models.
*   **Fine-tuning Success:** The study confirmed that fine-tuning a multilingual MLM model (MBERT+Akk) yielded state-of-the-art results (89% Hit@5), significantly outperforming LSTM baselines.

> "Strikingly, the **zero-shot performance** of a multilingual model (never trained on Akkadian) **surpasses a monolingual Akkadian model** trained from scratch, suggesting the pretraining signal is more important than the small target dataset."  
> — *Filling the Gaps in Ancient Akkadian Texts* (2021)

## References

1.  **Fetaya et al. (2021).** *Filling the Gaps in Ancient Akkadian Texts: A Masked Language Modelling Approach.* EMNLP 2021.
    *   *Key Finding:* Establishes restoration as an MLM task and demonstrates the superiority of multilingual pretraining over monolingual training.
2.  **Marone et al. (2025).** *MMBERT: A Modern Multilingual Encoder with Annealed Language Learning.*
    *   *Key Finding:* Validates the modern encoder-only MLM architecture (ModernBERT-based) as a powerful and efficient standard for multilingual understanding, outperforming larger generative models.

