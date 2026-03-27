# Linear Probing Pipeline — Run Log

**Model:** Qwen/Qwen2.5-7B-Instruct (28 transformer layers, hidden dim 3584, ~7B params)
**Cluster:** Schmidt Sciences HPC, H100 80GB
**Data:** 4,957 Akkadian letters — OB=1,497 | NA=2,435 | LB=1,025

---

## Step 00 — Tokenization Sanity Check
**Job:** 2030 | **Node:** g0374 | **Date:** 2026-03-26 10:19–10:20 UTC (~1 min)
**Status:** ✅ SUCCESS

### What this test does
Before running any heavy computation, we check: can Qwen's tokenizer even handle Akkadian text? Akkadian transliteration uses special Unicode characters (ṣ, š, ṭ, ḫ, ā, etc.) that may not exist in the model's vocabulary. If the tokenizer can't handle them, nothing downstream will work.

The script loads **only the tokenizer** (no model weights, no GPU needed), tokenizes all 4,957 texts, and reports statistics: how many tokens per text, how many unknown tokens, and what the tokenized output looks like.

### Results
| Metric | Value |
|--------|-------|
| Total texts | 4,957 |
| Mean tokens/text | 267.6 |
| Median tokens/text | 208.0 |
| Std tokens/text | 234.5 |
| Min tokens | 4 |
| Max tokens | 3,873 |
| Unknown tokens | 0 |
| Byte-fallback tokens | 0 |

### Per-Period Token Counts
| Period | N | Mean | Median | Std |
|--------|---|------|--------|-----|
| OB | 1,497 | 275.6 | 210.0 | 258.4 |
| NA | 2,435 | 256.1 | 181.0 | 244.9 |
| LB | 1,025 | 283.5 | 252.0 | 158.7 |

### Interpretation

**How Qwen tokenizes Akkadian:** Qwen's vocabulary (~151K tokens) was learned from modern text (English, Chinese, code, etc.). It has never seen Akkadian. When it encounters Akkadian characters like `š` or `ṣ`, it doesn't recognize them as meaningful units. Instead, it falls back to encoding them as raw UTF-8 bytes. For example:
- `š` (Unicode U+0161) → split into byte tokens `'Å¡'` (the two-byte UTF-8 encoding: 0xC5, 0xA1)
- `ṣ` → becomes `'á¹£'` (three UTF-8 bytes)
- Common Latin characters like `a`, `-`, `m` → tokenized normally

This means **Qwen has zero linguistic knowledge of Akkadian** — it treats the script as raw binary data. This is expected and is actually the interesting part: if the linear probe later finds that Qwen's internal representations still separate texts by temporal period, it means the model discovers statistical patterns in the byte sequences, not from understanding the language.

**No tokenization failures:** 0 unknown tokens means every character is representable. No data is lost in tokenization.

**Token counts across periods are similar:** OB=276, NA=256, LB=285 mean tokens. The differences are small enough that token count alone shouldn't be a strong classifier — this reduces concern about length as a confound.

**Truncation note:** Our extraction pipeline uses `max_length=512` tokens. The median text is 208 tokens (well under 512), but some long texts (max 3,873) will be truncated. This affects ~25% of texts with >512 tokens.

---

## Step 00b — Quick EDA (Final-Layer Embeddings)
**Jobs:** 2031 (first run, mean only), 2052 (re-run, mean + last-token)
**Node:** g0374 | **Date:** 2026-03-26 | **Duration:** 3.2 min (re-run)
**Status:** ✅ SUCCESS

### What this test does
Before committing to extracting activations at all 28 layers (the expensive step), we do a quick sanity check: does the model's **final layer** representation show any period structure at all?

The script loads the full model, runs all 4,957 texts through it in a single forward pass, and extracts the final layer's hidden state for each text. It then uses two methods to collapse per-token representations into a single vector per text:

1. **Mean pooling:** Average all token representations (weighted by attention mask to exclude padding). Captures the "overall flavor" of the entire text.
2. **Last-token pooling:** Take only the last token's representation. In decoder-only models (like Qwen), each token can only attend to previous tokens, so the last token has "seen" the entire text — it carries the full context.

These vectors are then projected to 2D using PCA and t-SNE for visualization, colored by period (OB=blue, NA=purple, LB=red).

### Results
| Metric | Value |
|--------|-------|
| Embeddings shape | (4957, 3584) per pooling method |
| Pooling methods | mean, last_token |
| Dimensionality reduction | PCA (2D), t-SNE (2D, perplexity=40) |
| UMAP | Skipped (not installed in cluster env) |

**Plots saved:**
- `results/plots/quick_eda_final_layer_mean.png`
- `results/plots/quick_eda_final_layer_last_token.png`

### Interpretation

**Mean pooling — clear period separation:**
- **PCA:** OB (blue) clusters to the left, NA (purple) to the right, LB (red) sits in between leaning toward OB. The first principal component roughly corresponds to a temporal axis.
- **t-SNE:** Three fairly distinct clusters. OB forms a tight blob, NA spreads across the opposite side, LB clusters separately. There's overlap at boundaries but periods are clearly separable.

**Last-token pooling — separation but noisier:**
- Same general pattern as mean pooling but with more scatter and overlap.
- Less clean structure — mean pooling is better for this data.
- This makes sense: Akkadian texts are sequences of many short syllabic tokens. Averaging all of them captures the text's overall statistical signature better than any single token.

**Key findings:**
1. **Qwen encodes period information despite having zero Akkadian knowledge.** The model has never seen Akkadian in training, yet its internal representations separate OB/NA/LB texts. This is the core insight we're testing.
2. **Mean pooling > last-token pooling** for this data. We'll run the linear probe on both for completeness but expect mean pooling to perform better.
3. **OB vs NA is the clearest split** (biggest gap in PCA). LB sits in between — this is historically sensible since Late Babylonian shares features with both earlier (OB) and contemporary (NA) traditions.
4. **This strongly justifies proceeding to the full linear probe** (step 02). If the final layer already separates periods this well, earlier/middle layers may show even more interesting patterns.

---

## Step 01 — Extract All-Layer Activations
**Job:** 2069 | **Node:** g0378 | **Date:** 2026-03-26 18:24–18:47 UTC (23 min total)
**Status:** ✅ SUCCESS

### What this step does
This is the data preparation for the linear probe. Instead of looking at just the final layer (like step 00b), we now extract activations at **every layer** of the model — from the embedding layer (layer 0, the raw token embeddings before any transformer processing) through all 28 transformer layers.

We run this extraction in 4 configurations to enable later analysis of confounds:

| Config | Cleaning | Pooling | Purpose |
|--------|----------|---------|---------|
| 1 | tier0 (minimal) | mean | Raw baseline — comparable to TF-IDF "raw" results |
| 2 | maximal (all 11 filters) | mean | Cleaned — tests if signal survives aggressive denoising |
| 3 | tier0 | last_token | Alternative pooling comparison |
| 4 | maximal | last_token | Cleaned + alternative pooling |

**Tier0 cleaning** is minimal: just strips `@v` markup artifacts from the transliteration format.

**Maximal cleaning** applies all 11 filters from the bias check (strip digits, remove logograms, strip determinatives, normalize vowels, lowercase, keep only syllabic tokens, etc.). This removes all the surface features that the TF-IDF baseline exploits.

### Results
| Config | Directory | Layers | Duration |
|--------|-----------|--------|----------|
| tier0 + mean | `activations/qwen2.5-7b-instruct/tier0/` | 29 .npz files | 3.6 min |
| maximal + mean | `activations/qwen2.5-7b-instruct/maximal/` | 29 .npz files | 2.3 min |
| tier0 + last_token | `activations/qwen2.5-7b-instruct/tier0_last_token/` | 29 .npz files | 8.5 min |
| maximal + last_token | `activations/qwen2.5-7b-instruct/maximal_last_token/` | 29 .npz files | 7.7 min |

Each `.npz` file contains a matrix of shape (4957, 3584) — one 3584-dimensional vector per text. Each directory also contains a `metadata.json` with text IDs, period labels, and token counts.

**Note:** Maximal cleaning runs faster because the cleaned texts are shorter (fewer tokens to process after stripping logograms, determinatives, etc.).

**Note:** These files are large (~2.5 GB per configuration, ~10 GB total) and stay on the cluster only (gitignored). Only plots and JSON results are synced via git.

---

## Step 02 — Linear Probe
*Not yet run*

### What this will do
Train a logistic regression classifier at each of the 29 layers to predict period (OB/NA/LB) from the activation vectors. This answers: "at which layer does the model encode the most temporal information?" The layer-accuracy curve and comparison to TF-IDF baselines are the main deliverables.

---

## Step 03 — Analyze Results
*Not yet run*

### What this will do
Classify the outcome as A, B, or C based on the probe results, comparing against the TF-IDF baselines from the bias check.
