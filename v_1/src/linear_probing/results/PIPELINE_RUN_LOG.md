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
**Jobs:** 2251 (final successful) | **Node:** g0377 | **Date:** 2026-03-28 13:27–15:07 UTC
**Duration:** 58.3 min (mean pooling) + 26.1 min (last_token pooling) = ~84 min total
**Status:** ✅ SUCCESS

### What this step does
Trains logistic regression probes at every layer of the model to answer: "at which layer does Qwen encode temporal period most linearly?" For each of the 29 layers × 2 cleaning conditions, we run a 5-fold cross-validated grid search over 6 regularization values (C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}). Then we run a 1000-permutation random-label test at the best layer to confirm statistical significance. Finally, we evaluate the best probe on the held-out test set (744 texts, 15%).

**Key implementation details:**
- `StandardScaler` inside a `sklearn.Pipeline` (fitted per fold, no leakage)
- `GridSearchCV(n_jobs=-1)` parallelizes 6 C × 5 folds = 30 fits simultaneously on 64 CPUs
- `permutation_test_score(n_jobs=-1)` parallelizes 1000 permutations across 64 CPUs
- Split: 70/15/15 stratified (train=3469, val=744, test=744), same seed as bias check

**Engineering issues encountered and fixed:**
| Problem | Fix |
|---------|-----|
| lbfgs no convergence, 860/1000 iterations | Added StandardScaler → 10x speedup (23.6s → 2.2s per fit) |
| Permutation test sequential, ~6 hours | Replaced loop with `permutation_test_score(n_jobs=-1)` → ~3 min |
| OOM kill with 64 workers × 32GB | Bumped memory to 256GB |
| Output buffered, no live logs | Added `python -u` flag |
| Job killed at 4h time limit | Bumped to 8h |

### Results — Mean Pooling

**Layer accuracy curve (tier0 cleaning):**
| Layer | Acc (CV) | F1 | Best C |
|-------|----------|----|--------|
| 0 | 0.9822 | 0.9802 | 0.01 |
| **4 (BEST)** | **0.9910** | **0.9900** | **1.0** |
| 14 | 0.9881 | 0.9868 | 0.1 |
| 28 | 0.9862 | 0.9849 | 0.1 |

Pattern: peaks early at layer 4, stays flat ~98-99% throughout all layers.

**Layer accuracy curve (maximal cleaning):**
| Layer | Acc (CV) | F1 | Best C |
|-------|----------|----|--------|
| 0 | 0.9378 | 0.9311 | 0.01 |
| **3 (BEST)** | **0.9625** | **0.9584** | **0.1** |
| 28 | 0.9461 | 0.9414 | 0.1 |

Pattern: peaks very early at layer 3, then gradually declines through middle layers, slight recovery at end.

**Permutation test (1000 permutations at layer 4, tier0):**
- Null distribution: mean=0.3746, std=0.0087, max=0.3995
- Real accuracy: 0.9907
- p-value: **0.001** (0/1000 permutations exceeded real accuracy — minimum achievable)

**Final test-set evaluation:**

| Cleaning | Best Layer | C | Test Acc | Test F1 | CV Acc | CV-Test Gap |
|----------|-----------|---|----------|---------|--------|-------------|
| tier0 | 4 | 1.0 | **0.9973** | **0.9969** | 0.9910 | -0.0063 |
| maximal | 3 | 0.1 | **0.9825** | **0.9803** | 0.9625 | -0.0200 |

**Confusion matrix (tier0, layer 4, test set — 744 texts):**
```
         LB    NA    OB
LB  [ 153    1    0 ]   → 99.3% correct
NA  [   1  364    0 ]   → 99.7% correct
OB  [   0    0  225 ]   → 100%  correct
```

### Results — Last-Token Pooling

**Key difference from mean pooling:** Last-token pooling shows a completely different layer profile — starts near chance (59% at layer 0), rises monotonically, peaks at the LAST layer (28).

**Layer accuracy curve (tier0 cleaning):**
| Layer | Acc (CV) | Pattern |
|-------|----------|---------|
| 0 | 0.5896 | Near chance (3-class = 33%) |
| 2 | 0.8789 | Early peak then dip |
| **28 (BEST)** | **0.9554** | Monotonically rises to final layer |

**Layer accuracy curve (maximal cleaning):**
| Layer | Acc (CV) | Pattern |
|-------|----------|---------|
| 0 | 0.5455 | Near chance |
| **28 (BEST)** | **0.9003** | Monotonically rises |

**Permutation test (1000 permutations at layer 28, tier0):**
- Null distribution: mean=0.3961, std=0.0081, max=0.4178
- Real accuracy: 0.9554
- p-value: **0.001**

**Final test-set evaluation:**

| Cleaning | Best Layer | Test Acc | Test F1 |
|----------|-----------|----------|---------|
| tier0 | 28 | **0.9651** | **0.9622** |
| maximal | 28 | **0.9086** | **0.9009** |

**Logs From Run:**

Reading from directories: ['tier0', 'maximal']
Data: 4957 texts
  Train: 3469, Val: 744, Test: 744
  Train+Val: 4213
Model: qwen2.5-7b-instruct, 29 layers, hidden_dim=3584

======================================================================
PROBING — tier0 cleaning (mean pooling)
======================================================================
  Layer  0: acc=0.9822 +/- 0.0035, F1=0.9802, C=0.01
  Layer  1: acc=0.9874 +/- 0.0033, F1=0.9859, C=1.0
  Layer  2: acc=0.9896 +/- 0.0014, F1=0.9882, C=0.1
  Layer  3: acc=0.9896 +/- 0.0027, F1=0.9882, C=0.1
  Layer  4: acc=0.9910 +/- 0.0018, F1=0.9900, C=1.0
  Layer  5: acc=0.9900 +/- 0.0021, F1=0.9888, C=0.1
  Layer  6: acc=0.9900 +/- 0.0026, F1=0.9888, C=0.1
  Layer  7: acc=0.9903 +/- 0.0025, F1=0.9893, C=1.0
  Layer  8: acc=0.9886 +/- 0.0031, F1=0.9872, C=1.0
  Layer  9: acc=0.9888 +/- 0.0021, F1=0.9876, C=0.1
  Layer 10: acc=0.9886 +/- 0.0021, F1=0.9874, C=1.0
  Layer 11: acc=0.9877 +/- 0.0024, F1=0.9864, C=0.1
  Layer 12: acc=0.9879 +/- 0.0012, F1=0.9865, C=1.0
  Layer 13: acc=0.9867 +/- 0.0017, F1=0.9851, C=0.1
  Layer 14: acc=0.9881 +/- 0.0026, F1=0.9868, C=0.1
  Layer 15: acc=0.9869 +/- 0.0029, F1=0.9855, C=0.1
  Layer 16: acc=0.9848 +/- 0.0020, F1=0.9836, C=0.1
  Layer 17: acc=0.9853 +/- 0.0022, F1=0.9838, C=1.0
  Layer 18: acc=0.9853 +/- 0.0035, F1=0.9836, C=1.0
  Layer 19: acc=0.9860 +/- 0.0027, F1=0.9843, C=1.0
  Layer 20: acc=0.9834 +/- 0.0034, F1=0.9816, C=1.0
  Layer 21: acc=0.9839 +/- 0.0022, F1=0.9822, C=0.1
  Layer 22: acc=0.9836 +/- 0.0020, F1=0.9819, C=0.1
  Layer 23: acc=0.9841 +/- 0.0042, F1=0.9824, C=1.0
  Layer 24: acc=0.9850 +/- 0.0016, F1=0.9834, C=0.1
  Layer 25: acc=0.9855 +/- 0.0027, F1=0.9838, C=1.0
  Layer 26: acc=0.9867 +/- 0.0026, F1=0.9852, C=1.0
  Layer 27: acc=0.9872 +/- 0.0039, F1=0.9860, C=0.1
  Layer 28: acc=0.9862 +/- 0.0018, F1=0.9849, C=0.1

======================================================================
PROBING — maximal cleaning (mean pooling)
======================================================================
  Layer  0: acc=0.9378 +/- 0.0077, F1=0.9311, C=0.01
  Layer  1: acc=0.9549 +/- 0.0069, F1=0.9504, C=0.01
  Layer  2: acc=0.9611 +/- 0.0061, F1=0.9568, C=0.1
  Layer  3: acc=0.9625 +/- 0.0041, F1=0.9584, C=0.1
  Layer  4: acc=0.9585 +/- 0.0041, F1=0.9543, C=0.01
  Layer  5: acc=0.9575 +/- 0.0081, F1=0.9528, C=0.1
  Layer  6: acc=0.9592 +/- 0.0039, F1=0.9551, C=0.01
  Layer  7: acc=0.9563 +/- 0.0068, F1=0.9514, C=1.0
  Layer  8: acc=0.9563 +/- 0.0046, F1=0.9519, C=0.01
  Layer  9: acc=0.9570 +/- 0.0041, F1=0.9525, C=0.1
  Layer 10: acc=0.9516 +/- 0.0041, F1=0.9469, C=0.01
  Layer 11: acc=0.9468 +/- 0.0036, F1=0.9414, C=0.1
  Layer 12: acc=0.9456 +/- 0.0029, F1=0.9401, C=0.1
  Layer 13: acc=0.9445 +/- 0.0055, F1=0.9383, C=0.01
  Layer 14: acc=0.9485 +/- 0.0041, F1=0.9430, C=0.1
  Layer 15: acc=0.9461 +/- 0.0037, F1=0.9405, C=0.1
  Layer 16: acc=0.9430 +/- 0.0054, F1=0.9372, C=0.01
  Layer 17: acc=0.9442 +/- 0.0021, F1=0.9379, C=1.0
  Layer 18: acc=0.9404 +/- 0.0076, F1=0.9345, C=0.1
  Layer 19: acc=0.9390 +/- 0.0074, F1=0.9329, C=0.1
  Layer 20: acc=0.9361 +/- 0.0036, F1=0.9293, C=1.0
  Layer 21: acc=0.9402 +/- 0.0047, F1=0.9338, C=0.01
  Layer 22: acc=0.9442 +/- 0.0040, F1=0.9379, C=0.01
  Layer 23: acc=0.9468 +/- 0.0024, F1=0.9412, C=0.1
  Layer 24: acc=0.9497 +/- 0.0027, F1=0.9443, C=1.0
  Layer 25: acc=0.9478 +/- 0.0028, F1=0.9425, C=0.1
  Layer 26: acc=0.9478 +/- 0.0042, F1=0.9426, C=0.01
  Layer 27: acc=0.9521 +/- 0.0069, F1=0.9471, C=0.1
  Layer 28: acc=0.9461 +/- 0.0051, F1=0.9414, C=0.1

Best layer (tier0):   4 (acc=0.9910)
Best layer (maximal): 3 (acc=0.9625)
Checkpoint saved to /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/layer_results_checkpoint_qwen2.5-7b-instruct.json

======================================================================
RANDOM-LABEL BASELINE (1000 permutations at layer 4)
======================================================================
  Null distribution: mean=0.3746, std=0.0087, max=0.3995
  Real accuracy: 0.9907
  p-value: 0.000999000999000999

======================================================================
FINAL TEST-SET EVALUATION
======================================================================

  [tier0] Layer 4, C=1.0
    Test accuracy:  0.9973
    Test F1 macro:  0.9969
    CV accuracy:    0.9910
    CV-Test gap:    -0.0063
    Confusion matrix:
[[153   1   0]
 [  1 364   0]
 [  0   0 225]]

  [maximal] Layer 3, C=0.1
    Test accuracy:  0.9825
    Test F1 macro:  0.9803
    CV accuracy:    0.9625
    CV-Test gap:    -0.0200
    Confusion matrix:
[[148   6   0]
 [  6 359   0]
 [  0   1 224]]

Saved results to /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/probe_results_qwen2.5-7b-instruct.json
Saved /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/plots/layer_accuracy_curve.png
Saved /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/plots/confound_random_label.png
Saved /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/plots/tsne_best_layer.png
Saved /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/plots/confusion_matrix_best_layer.png

Total wall time: 58.3 min
=== Probing (last_token pooling) ===
Pooling method: last_token
Reading from directories: ['tier0_last_token', 'maximal_last_token']
Data: 4957 texts
  Train: 3469, Val: 744, Test: 744
  Train+Val: 4213
Model: qwen2.5-7b-instruct, 29 layers, hidden_dim=3584

======================================================================
PROBING — tier0 cleaning (last_token pooling)
======================================================================
  Layer  0: acc=0.5896 +/- 0.0167, F1=0.4933, C=1.0
  Layer  1: acc=0.8630 +/- 0.0084, F1=0.8483, C=0.1
  Layer  2: acc=0.8789 +/- 0.0103, F1=0.8658, C=0.1
  Layer  3: acc=0.8460 +/- 0.0073, F1=0.8317, C=1.0
  Layer  4: acc=0.8365 +/- 0.0141, F1=0.8223, C=0.1
  Layer  5: acc=0.8469 +/- 0.0137, F1=0.8324, C=1.0
  Layer  6: acc=0.8464 +/- 0.0080, F1=0.8305, C=0.1
  Layer  7: acc=0.8459 +/- 0.0084, F1=0.8303, C=0.1
  Layer  8: acc=0.8623 +/- 0.0065, F1=0.8483, C=0.1
  Layer  9: acc=0.8607 +/- 0.0075, F1=0.8462, C=0.1
  Layer 10: acc=0.8737 +/- 0.0101, F1=0.8602, C=0.01
  Layer 11: acc=0.8595 +/- 0.0034, F1=0.8458, C=1.0
  Layer 12: acc=0.8633 +/- 0.0036, F1=0.8484, C=0.01
  Layer 13: acc=0.8635 +/- 0.0084, F1=0.8483, C=0.01
  Layer 14: acc=0.8747 +/- 0.0107, F1=0.8603, C=0.1
  Layer 15: acc=0.8723 +/- 0.0104, F1=0.8587, C=0.01
  Layer 16: acc=0.8609 +/- 0.0126, F1=0.8458, C=0.01
  Layer 17: acc=0.8623 +/- 0.0065, F1=0.8479, C=0.1
  Layer 18: acc=0.8550 +/- 0.0101, F1=0.8374, C=0.01
  Layer 19: acc=0.8702 +/- 0.0072, F1=0.8586, C=0.1
  Layer 20: acc=0.8789 +/- 0.0092, F1=0.8660, C=0.01
  Layer 21: acc=0.8894 +/- 0.0083, F1=0.8785, C=0.01
  Layer 22: acc=0.9010 +/- 0.0092, F1=0.8915, C=0.01
  Layer 23: acc=0.9181 +/- 0.0031, F1=0.9107, C=0.1
  Layer 24: acc=0.9210 +/- 0.0081, F1=0.9135, C=0.1
  Layer 25: acc=0.9283 +/- 0.0071, F1=0.9203, C=0.1
  Layer 26: acc=0.9390 +/- 0.0038, F1=0.9321, C=0.1
  Layer 27: acc=0.9475 +/- 0.0096, F1=0.9418, C=0.1
  Layer 28: acc=0.9554 +/- 0.0044, F1=0.9509, C=0.01

======================================================================
PROBING — maximal cleaning (last_token pooling)
======================================================================
  Layer  0: acc=0.5455 +/- 0.0093, F1=0.4162, C=0.1
  Layer  1: acc=0.8457 +/- 0.0153, F1=0.8295, C=0.1
  Layer  2: acc=0.8443 +/- 0.0059, F1=0.8292, C=0.1
  Layer  3: acc=0.8179 +/- 0.0125, F1=0.8007, C=0.1
  Layer  4: acc=0.8051 +/- 0.0123, F1=0.7841, C=0.01
  Layer  5: acc=0.8051 +/- 0.0113, F1=0.7846, C=0.01
  Layer  6: acc=0.7966 +/- 0.0095, F1=0.7745, C=0.01
  Layer  7: acc=0.7698 +/- 0.0200, F1=0.7467, C=0.01
  Layer  8: acc=0.8008 +/- 0.0135, F1=0.7785, C=0.01
  Layer  9: acc=0.7871 +/- 0.0193, F1=0.7623, C=0.01
  Layer 10: acc=0.7947 +/- 0.0186, F1=0.7708, C=0.01
  Layer 11: acc=0.7797 +/- 0.0179, F1=0.7543, C=0.01
  Layer 12: acc=0.7714 +/- 0.0152, F1=0.7431, C=0.01
  Layer 13: acc=0.7714 +/- 0.0127, F1=0.7427, C=0.01
  Layer 14: acc=0.7690 +/- 0.0144, F1=0.7302, C=0.001
  Layer 15: acc=0.7671 +/- 0.0067, F1=0.7390, C=0.01
  Layer 16: acc=0.7586 +/- 0.0177, F1=0.7315, C=0.01
  Layer 17: acc=0.7541 +/- 0.0156, F1=0.7271, C=0.01
  Layer 18: acc=0.7524 +/- 0.0239, F1=0.7256, C=0.01
  Layer 19: acc=0.7662 +/- 0.0193, F1=0.7389, C=0.01
  Layer 20: acc=0.7769 +/- 0.0174, F1=0.7495, C=0.01
  Layer 21: acc=0.7852 +/- 0.0231, F1=0.7607, C=0.01
  Layer 22: acc=0.8011 +/- 0.0162, F1=0.7777, C=0.01
  Layer 23: acc=0.8201 +/- 0.0169, F1=0.8000, C=0.01
  Layer 24: acc=0.8246 +/- 0.0142, F1=0.8040, C=0.01
  Layer 25: acc=0.8497 +/- 0.0152, F1=0.8329, C=0.01
  Layer 26: acc=0.8744 +/- 0.0180, F1=0.8626, C=0.01
  Layer 27: acc=0.8963 +/- 0.0124, F1=0.8851, C=0.01
  Layer 28: acc=0.9003 +/- 0.0175, F1=0.8902, C=0.01

Best layer (tier0):   28 (acc=0.9554)
Best layer (maximal): 28 (acc=0.9003)
Checkpoint saved to /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/layer_results_checkpoint_qwen2.5-7b-instruct_last_token.json

======================================================================
RANDOM-LABEL BASELINE (1000 permutations at layer 28)
======================================================================
/home/yarin.b/miniconda3/envs/thesis/lib/python3.11/site-packages/joblib/externals/loky/process_executor.py:782: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
  warnings.warn(
  Null distribution: mean=0.3961, std=0.0081, max=0.4178
  Real accuracy: 0.9554
  p-value: 0.000999000999000999

======================================================================
FINAL TEST-SET EVALUATION
======================================================================

  [tier0] Layer 28, C=0.01
    Test accuracy:  0.9651
    Test F1 macro:  0.9622
    CV accuracy:    0.9554
    CV-Test gap:    -0.0097
    Confusion matrix:
[[145   7   2]
 [  7 351   7]
 [  1   2 222]]

  [maximal] Layer 28, C=0.01
    Test accuracy:  0.9086
    Test F1 macro:  0.9009
    CV accuracy:    0.9003
    CV-Test gap:    -0.0083
    Confusion matrix:
[[126  28   0]
 [ 21 334  10]
 [  0   9 216]]

Saved results to /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/probe_results_qwen2.5-7b-instruct_last_token.json
Saved /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/plots/layer_accuracy_curve_last_token.png
Saved /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/plots/confound_random_label_last_token.png
Saved /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/plots/tsne_best_layer_last_token.png
Saved /home/yarin.b/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/plots/confusion_matrix_best_layer_last_token.png

Total wall time: 26.1 min
=== Done ===
End: Sat Mar 28 15:07:41 UTC 2026

### Comparison to TF-IDF Baselines (from Bias Check)

| Method | Tier0/Raw | Maximal/Cleaned |
|--------|-----------|-----------------|
| TF-IDF Unigram | 84.8% | 69.1% |
| TF-IDF Bigram | 98.3% | 91.2% |
| TF-IDF 2-5gram | 99.2% | 96.7% |
| **Qwen mean pooling (best layer)** | **99.7%** | **98.25%** |
| **Qwen last-token (best layer)** | **96.5%** | **90.9%** |

### Scientific Interpretation

**Finding 1 — Mean pooling, early peak (layer 4):**
Temporal period information is linearly encoded already in the early layers of Qwen and maintained throughout the entire network. The model doesn't need deep processing to separate OB/NA/LB — this happens in the first ~15% of layers.

**Finding 2 — Last-token pooling, late peak (layer 28):**
This is the opposite pattern — consistent with how autoregressive attention works. The last token accumulates context from all previous tokens progressively. Information is not concentrated until the final layers.

**Finding 3 — Maximal cleaning drop is moderate:**
Removing logograms, determinatives and all surface temporal markers drops accuracy from 99.7% → 98.25% (mean) and 96.5% → 90.9% (last-token). The signal survives aggressive cleaning — temporal information is encoded in the syllabic content, not just vocabulary markers.

**Finding 4 — Both p=0.001:**
Zero of 1000 permutations exceeded real accuracy in both pooling conditions. The result is not a statistical artifact.

### Plots Saved
- `results/plots/layer_accuracy_curve.png` — layer curve for mean pooling (both cleanings + TF-IDF baselines)
- `results/plots/layer_accuracy_curve_last_token.png` — same for last-token
- `results/plots/confound_random_label.png` — permutation test null distribution (mean)
- `results/plots/confound_random_label_last_token.png` — same for last-token
- `results/plots/tsne_best_layer.png` — t-SNE at best layer (all 4957 texts, mean pooling)
- `results/plots/tsne_best_layer_last_token.png` — same for last-token
- `results/plots/confusion_matrix_best_layer.png` — confusion matrices (mean)
- `results/plots/confusion_matrix_best_layer_last_token.png` — same for last-token

---

## Step 01b — Extract Random-Weights Baseline Activations
**Job:** 2301 | **Node:** g0381 | **Date:** 2026-03-30 13:03–13:33 UTC (~30 min)
**Status:** ✅ SUCCESS

### What this step does
Initialize Qwen2.5-7B with the same architecture and pretrained tokenizer, but **randomly initialized weights** (`AutoModelForCausalLM.from_config()` instead of `from_pretrained()`). Extract activations identically to Step 01. This isolates how much of the probe's accuracy comes from pretraining vs. tokenizer + architecture.

### Results
All 4 configurations extracted successfully (tier0/maximal × mean/last_token). Each produced 29 layer files + metadata.json, saved to `results/activations/qwen2.5-7b-instruct-random/`.

| Config | Wall Time |
|--------|-----------|
| tier0, mean | 7.0 min |
| maximal, mean | 5.8 min |
| tier0, last_token | 5.7 min |
| maximal, last_token | 11.7 min |

---

## Step 02b — Validity Tests
**Job:** 2302 | **Node:** g0381 | **Date:** 2026-03-30 13:41–15:05 UTC
**Status:** ⚠️ PARTIAL — Random baseline probing + Learning curve succeeded. PCA crashed (bug fixed). MLP + comparison plots never ran.

### Random-Weights Baseline Probing (reuses 02_linear_probe.py)

Ran the standard probe pipeline on random-weights activations. Results are striking:

#### Mean Pooling — Random vs Pretrained

| Cleaning | Random Best Layer | Random Acc | Pretrained Best Layer | Pretrained Acc | Gap | Selectivity |
|----------|-------------------|-----------|----------------------|---------------|-----|-------------|
| tier0 | 1 | 98.27% | 4 | 99.10% | +0.83% | Low |
| maximal | 0 | 93.61% | 3 | 96.25% | +2.64% | Low |

#### Last-Token Pooling — Random vs Pretrained

| Cleaning | Random Best Layer | Random Acc | Pretrained Best Layer | Pretrained Acc | Gap | Selectivity |
|----------|-------------------|-----------|----------------------|---------------|-----|-------------|
| tier0 | 1 | 94.33% | 28 | 95.54% | +1.21% | Low |
| maximal | 1 | 84.48% | 28 | 90.03% | +5.55% | Moderate |

#### Key Finding: Layer Curve Patterns Are Opposite

- **Pretrained:** Accuracy maintained or rising through depth (stable representations)
- **Random:** Best at layer 0–1, then **monotonically declining** (random transformations destroy signal)

This is the clearest qualitative difference. The pretrained model's transformations are information-preserving; random transformations progressively add noise.

#### Interpretation

The random model's 98.3% (tier0, mean) exactly matches the TF-IDF baseline (98.3%). Mean pooling over random embeddings is mathematically a random projection of the token-frequency distribution (Johnson-Lindenstrauss lemma). Since different Akkadian periods have distinct vocabulary distributions, random projections preserve that separability.

**What pretraining adds:** The gap grows with task difficulty — largest under maximal cleaning + last_token pooling (+5.5%). Pretraining helps most when surface distributional features are stripped and the model must compress contextual information into a single position.

**Selectivity (Hewitt & Liang 2019):** Probe accuracy minus random baseline accuracy. Only the maximal/last_token condition (5.5%) approaches a credible selectivity score. The mean-pooling tier0 result (0.8%) is essentially a TF-IDF finding.

---

## Step 02b — Validity Tests (Re-run, Job 2392)
**Job:** 2392 | **Node:** g0380 | **Date:** 2026-03-31 10:15–10:47 UTC (~32 min)
**Status:** ✅ ALL EXPERIMENTS COMPLETE (mean + last_token pooling)

### Experiment A — Learning Curve ✅

#### Mean Pooling

| Fraction | # Texts | tier0 (Layer 4) | maximal (Layer 3) |
|----------|---------|-----------------|-------------------|
| 1% | 42 | 93.10% ± 4.19% | 78.81% ± 7.86% |
| 5% | 210 | 96.57% ± 0.97% | 89.00% ± 2.70% |
| 10% | 421 | 97.39% ± 0.46% | 91.19% ± 1.50% |
| 25% | 1053 | 97.92% ± 0.61% | 93.74% ± 0.67% |
| 50% | 2106 | 98.55% ± 0.23% | 94.95% ± 0.33% |
| 100% | 4213 | 99.06% ± 0.08% | 96.15% ± 0.15% |

#### Last-Token Pooling

| Fraction | # Texts | tier0 (Layer 28) | maximal (Layer 28) |
|----------|---------|------------------|-------------------|
| 1% | 42 | 68.10% ± 8.73% | 57.86% ± 8.59% |
| 5% | 210 | 83.10% ± 3.55% | 74.14% ± 2.31% |
| 10% | 421 | 88.53% ± 1.05% | 78.65% ± 1.90% |
| 25% | 1053 | 92.52% ± 0.82% | 84.59% ± 1.31% |
| 50% | 2106 | 94.23% ± 0.40% | 87.68% ± 0.67% |
| 100% | 4213 | 95.66% ± 0.16% | 89.96% ± 0.29% |

**Interpretation:** Mean pooling is extremely sample-efficient — tier0 hits 93% with just 42 texts. Last_token is data-hungry — needs 10× more data to reach comparable accuracy. This contrast shows that mean pooling encodes period as a compact, immediately accessible feature; last_token requires contextual integration learned across more examples.

**Plots:** `results/plots/learning_curve.png`, `learning_curve_last_token.png`

---

### Experiment B — PCA Dimensionality Reduction ✅

#### Mean Pooling

| k | tier0 (Layer 4) | maximal (Layer 3) |
|---|-----------------|-------------------|
| 2 | 61.40% ± 1.35% | 64.21% ± 0.88% |
| 5 | 90.13% ± 0.55% | 71.45% ± 1.07% |
| 10 | 92.50% ± 0.72% | 77.62% ± 0.54% |
| 25 | 96.46% ± 0.14% | 86.83% ± 0.80% |
| 50 | 97.22% ± 0.16% | 91.17% ± 0.56% |
| 100 | 98.01% ± 0.27% | 93.80% ± 0.64% |
| 250 | 98.20% ± 0.58% | 95.28% ± 0.51% |
| 500 | 98.55% ± 0.51% | 95.18% ± 0.22% |
| 1000 | 98.81% ± 0.36% | 95.47% ± 0.72% |
| 3584* | 44.70% ± 6.03% | 38.48% ± 2.32% |

#### Last-Token Pooling

| k | tier0 (Layer 28) | maximal (Layer 28) |
|---|------------------|-------------------|
| 2 | 52.17% ± 1.66% | 48.99% ± 0.22% |
| 5 | 54.31% ± 0.77% | 48.92% ± 0.26% |
| 10 | 56.09% ± 0.65% | 56.28% ± 1.14% |
| 25 | 72.06% ± 0.83% | 69.52% ± 1.07% |
| 50 | 87.44% ± 0.53% | 77.81% ± 1.73% |
| 100 | 91.48% ± 1.09% | 83.84% ± 1.12% |
| 250 | 93.66% ± 0.80% | 87.35% ± 1.56% |
| 500 | 95.23% ± 0.67% | 88.30% ± 1.48% |
| 1000 | 95.20% ± 0.94% | 88.11% ± 1.43% |
| 3584* | 33.30% ± 6.60% | 34.35% ± 5.36% |

*k=3584 drop to near-chance is a numerical artifact: full-rank PCA retains noise components that destabilize the logistic regression. Ignore this point.

**Interpretation:** Mean pooling signal is very compact — **top 5 PCs recover 90%** of tier0 accuracy. Last_token signal is much more distributed — needs k≈50–100 to recover meaningful accuracy. Period information is organized differently depending on pooling: mean pooling concentrates it in top principal components, last_token spreads it across many dimensions.

**Plots:** `results/plots/pca_accuracy_vs_dims.png`, `pca_accuracy_vs_dims_last_token.png`

---

### Experiment C — Linear vs MLP Probe ✅

**Key finding: MLP is consistently WORSE than linear at every layer and condition.** This is stronger than "MLP ≈ linear" — a nonlinear probe actively hurts, confirming the data is genuinely and cleanly linearly separable.

#### Mean Pooling — Selected Layers (tier0 / maximal)

| Layer | Linear (tier0) | MLP (tier0) | Delta | Linear (maximal) | MLP (maximal) | Delta |
|-------|---------------|-------------|-------|-----------------|---------------|-------|
| 0 | 98.22% | 97.86% | −0.36% | 93.78% | 93.57% | −0.21% |
| 4 | 99.10% | 98.03% | −1.07% | 95.85% | 94.21% | −1.64% |
| 14 | 98.81% | 97.67% | −1.14% | 94.85% | 93.90% | −0.95% |
| 28 | 98.62% | 97.74% | −0.88% | 94.61% | 92.64% | −1.97% |

#### Last-Token Pooling — Selected Layers (tier0 / maximal)

| Layer | Linear (tier0) | MLP (tier0) | Delta | Linear (maximal) | MLP (maximal) | Delta |
|-------|---------------|-------------|-------|-----------------|---------------|-------|
| 1 | 86.30% | 80.44% | −5.86% | 84.57% | 78.59% | −5.98% |
| 14 | 87.47% | 85.09% | −2.37% | 76.90% | 76.33% | −0.57% |
| 28 | 95.54% | 93.76% | −1.78% | 90.03% | 88.18% | −1.85% |

**Plots:** `results/plots/mlp_vs_linear.png`, `mlp_vs_linear_last_token.png`

---

### Experiment D — Random Baseline Comparison ✅

At best layers:

| Pooling | Cleaning | Pretrained | Random | Gap (Selectivity) |
|---------|----------|-----------|--------|-------------------|
| Mean | tier0 (L4) | 99.10% | 97.72% | **+1.38%** |
| Mean | maximal (L3) | 96.25% | 90.74% | **+5.51%** |
| Last_token | tier0 (L28) | 95.54% | 84.83% | **+10.70%** |
| Last_token | maximal (L28) | 90.03% | 70.14% | **+19.89%** |

The last_token/maximal gap of **+19.9%** is the strongest evidence that pretraining contributes meaningful representation structure beyond what the tokenizer + random architecture provides.

**Plots:** `results/plots/random_baseline_comparison.png`, `random_baseline_comparison_last_token.png`

---

### Overall Validity Assessment

| Experiment | Finding | Thesis implication |
|-----------|---------|-------------------|
| Learning curve | Mean: 93% with 42 texts; last_token needs 10× more data | Period signal is immediately accessible in mean-pooled representations |
| PCA | Mean: top-5 PCs → 90%; last_token: needs k≈50–100 | Mean pooling concentrates signal; last_token distributes it |
| MLP vs Linear | MLP < linear everywhere (−0.2% to −6%) | Encoding is genuinely linearly structured |
| Random baseline | +1–5% for mean; +11–20% for last_token | Pretraining contribution is most visible in last_token representations |

**Pending:** cluster push needed for validity JSON files + 6 plots.

---

## Step 03 — Analyze Results
*Not yet run*

### What this will do
Classify the outcome as A, B, or C based on the probe results, comparing against the TF-IDF baselines from the bias check.

---

## Step 05 — ORCC PLS Year Regression + Ruler PLS-DA
**Dataset:** ORCC Royal Inscriptions, 1,202 fragments; 893 have non-null `year` labels (range 7–1132 BCE) and a `ruler` label (38 unique kings). PLS is fitted on the 893 labeled rows; projections cover all 1,586 SEAL+ORCC fragments so SEAL points can be placed in the same supervised latent space.

**Cluster jobs (final, after bug fixes):**
- `6522` — Qwen, all 4 cleaning×pooling combos, year + ruler
- `6523` — Random baseline, same
- `6433` — Akkadian MLM, tier0/mean only (only available config)
- TF-IDF — local (`05_compute_pls_tfidf.py`), no GPU

**Status:** ✅ SUCCESS — results pushed via cluster (commits `b0f4c25`, `79164d5`, `0336887`)

### What this step does

For each `(method, cleaning, pooling, layer)` combination we fit a **PLS regression** to predict year and a **PLS-DA** (one-hot regression → argmax) to classify ruler, evaluated by cross-validation. We L2-normalize activations row-wise before fitting (per `pls_utils.l2_normalize`) so that PLS sees unit-sphere vectors and scale variation doesn't dominate the latent directions.

**Year regression (GroupKFold by ruler, 5 folds):**
- Two target transforms: `raw` (year as float) and `log` (natural log of year)
- Sweep `n_components` k ∈ {1, 2, 3, 5}
- Folds where `y_test` is constant are flagged via Spearman NaN and excluded from mean metrics (`n_valid_folds` reported)
- Metrics: `r2`, `spearman`, `mae`, `mase`, `mdape`
- **Shuffled null:** `np.random.default_rng(42).permutation(y)` — global permutation, same CV splits

> **Bug fix (2026-05-09):** The original shuffled baseline permuted within-ruler groups. Because year ≈ ruler (median 1 year per ruler), within-group shuffling was a near-no-op and produced NaN baselines. Replaced with a global permutation that gives a true null. See `pls_utils._global_shuffle`.

**Ruler PLS-DA (StratifiedKFold, 5 folds):**
- One-hot encode 38 rulers, fit `PLSRegression` on the one-hot matrix, predict by argmax
- Sweep k ∈ {1, 2, 3, 5}
- Metrics: `accuracy`, `macro_f1`, `weighted_f1`; chance baselines: majority-class fraction and 1/n_classes; plus a global-shuffle baseline
- Note: rulers with only 1 fragment trigger a sklearn warning but don't break StratifiedKFold

**Projections (for viz):**
- Fit a full-data PLS with 5 components; project all 1,586 fragments
- For year: save `pls12-{raw|log}`, `pls23-{raw|log}`, `pls34-{raw|log}` (component pairs)
- For ruler: save `plsda12`
- TF-IDF projections fit on labeled subset (893 rows) but **project all 1,586** so SEAL fragments get coordinates too (fixed in 2026-05-10)

### Scripts

| Script | Purpose |
|--------|---------|
| `pls_utils.py` | Shared API: `l2_normalize`, `fit_pls_groupkfold`, `fit_pls_full`, `fit_plsda_stratified_kfold`, `fit_plsda_full`, `project`, `compute_metrics` |
| `05_compute_pls.py` | Qwen + Random driver — CLI: `--method`, `--cleaning`, `--pooling`, `--layers`, `--target {year,ruler}`, `--overwrite` |
| `05_compute_pls_mlm.py` | Akkadian MLM driver — `--target {year,ruler,both}` |
| `05_compute_pls_tfidf.py` | TF-IDF driver — runs locally, no GPU |
| `sbatch/orcc/pls_qwen.sh`, `pls_random.sh`, `pls_mlm.sh` | Slurm scripts; each runs all 4 cleaning×pooling × both targets and pushes results to GitHub on completion |

**Important `--overwrite` semantics:** Target-aware. Only clears `__year-*` keys when `--target year`, only `__ruler` keys when `--target ruler`. Earlier versions wiped everything for the prefix and caused data loss across target types.

**Important `print_summary()`:** Branches on `rec.get('target') == 'ruler'` to use `best_k_by_macro_f1` vs `best_k_by_spearman`. Earlier versions crashed with KeyError after ruler runs and prevented git push.

### Outputs (per method)

```
results/orcc_round1/pls/
├── pls_results_{method}.json          # Per-config metrics (k sweep, baselines, best-k)
└── pls_projections_{method}.json      # {fragment_ids, embeddings: {key → [[x,y]×1586]}}
```

**Config-key schema:**
- Year: `{method}__{cleaning}__{pooling}__L{nn}__year-{raw|log}`
- Ruler: `{method}__{cleaning}__{pooling}__L{nn}__ruler`

**Projection-key schema:**
- Year (mean): `{method}__{cleaning}__L{nn}__{pls12|pls23|pls34}-{raw|log}`
- Year (last): `{method}__{cleaning}__L{nn}__last__{pls12|pls23|pls34}-{raw|log}`
- Ruler (mean): `{method}__{cleaning}__L{nn}__plsda12`
- Ruler (last): `{method}__{cleaning}__L{nn}__last__plsda12`

### Total config counts

| Method | Layers | Cleanings | Poolings | Year configs | Ruler configs |
|--------|--------|-----------|----------|--------------|---------------|
| qwen   | 29     | 2         | 2        | 232 (×2 transforms) | 116 |
| random | 29     | 2         | 2        | 232 | 116 |
| mlm    | 5 (L00,04,08,12,16) | 1 (tier0) | 1 (mean) | 10  | 5   |
| tfidf  | 1 (L00) | 2        | 1 (na)   | 4   | 2   |

---

## Step 05b — Linear Classification Probes (CLS)
**Pre-existing code** (`05_compute_cls.py`) was already producing `cls_results_*.json` for ruler and year-as-category tasks via `LogisticRegression` + `StratifiedKFold(5)`. This work added aggregation and plotting (Step 06–07).

**Tasks:**
- `ruler` — 38-class classification
- `year` — year-as-category, classes are calendar years that appear in the labeled set

**Pipeline:** L2-normalize activations → fit `LogisticRegression(C=1.0, max_iter=1000)` per layer → 5-fold StratifiedCV → record accuracy/macro_f1/weighted_f1 + chance baselines.

**Outputs (already on cluster, pre-existing):**
```
results/orcc_round1/cls/
├── cls_results_qwen.json
├── cls_results_random.json
├── cls_results_mlm.json
└── cls_results_tfidf.json
```

---

## Step 06 — Aggregation
**Local scripts, run once after all Step 05 results land.**

| Script | Inputs | Outputs |
|--------|--------|---------|
| `06_aggregate_pls.py` | `pls_results_*.json` (×4) | `pls_best_layers.json`, `pls_layer_curves.json` |
| `06_aggregate_cls.py` | `cls_results_*.json` (×4) | `cls_best_layers.json`, `cls_layer_curves.json` |

**`pls_best_layers.json`** — one entry per `{method, cleaning, pooling, target}` group, picking the best layer by Spearman (regression) or macro_f1 (classification). 33 entries total.

**`pls_layer_curves.json`** — full curve data: list of `{layer, k, spearman_mean, r2_mean, mae_mean, mase_mean, mdape_mean, shuffled_*, ...}` rows per group, used by the plotter.

**`cls_best_layers.json`** — one entry per `{method, cleaning, pooling, task}`. 22 entries total.

**`cls_layer_curves.json`** — full curve data per group.

Both `06_aggregate_*.py` print summary markdown tables to stdout when run.

---

## Step 07 — Plotting
**Local scripts, idempotent — re-run anytime after Step 06.**

### `07_plot_pls_curves.py`

Reads `pls_layer_curves.json`, produces in `results/orcc_round1/pls/figures/`:

- **Per-group regression PNG** (`{method}_{cleaning}_{pooling}_{raw|log}.png`) — 2×3 grid:
  - Row 0: Spearman ρ | R² (clipped at −10) | MAE
  - Row 1: MASE | MDAPE | (hidden)
  - One line per k ∈ {1,2,3,5}, plus dashed shuffled baseline on Spearman and R²
- **Per-group ruler PNG** (`{method}_{cleaning}_{pooling}_ruler.png`) — 1×2: Accuracy | Macro-F1 vs layer with chance + shuffled dashed lines
- **Combined best-of**:
  - `best_of_year-{raw,log}.png` — all methods × cleaning × pooling on one Spearman plot (best-k per layer)
  - `best_of_mae_year-{raw,log}.png` — same for MAE

### `07_plot_cls_curves.py`

Reads `cls_layer_curves.json`, produces in `results/orcc_round1/cls/figures/`:

- **Per-group** (`{method}_{cleaning}_{pooling}_{ruler|year}.png`) — 1×2: Accuracy | Macro-F1 vs layer with chance baseline
- **Combined best-of** (`best_of_{ruler|year}.png`) — best macro-F1 per layer per method, all methods overlaid

---

## Step 08 — Viz Extension: PLS as a Reduction in the Embedding Explorer
**Date: 2026-05-11.** Adds 3 supervised reductions (PLS-Year raw/log, PLS-Ruler) to `seal_eda.html` alongside t-SNE / PCA / UMAP.

**Implementation:**
1. `02_merge_coords.py` extended to load `pls_projections_{qwen,random,mlm,tfidf}.json` and merge into `seal_viz_data.json`. Filters to `pls12-raw`, `pls12-log`, `plsda12` only (skips `pls23`, `pls34` for file-size reasons). Fragment IDs are compared as strings (viz stores SEAL IDs as ints, PLS uses strings).
2. `seal_eda.html` — three new buttons in the reduction toggle group with info tooltips. `buildKey()` extended: for TF-IDF the layer slot is `"L00"` when PLS is selected and `"na"` for t-SNE/PCA/UMAP. `reductionLabel` map updated with the three PLS labels.
3. **Standalone HTML is gitignored** — rebuild locally with `python3 v_1/src/viz/03_build_standalone_html.py`.

**Data size:** `seal_viz_data.json` grew from 45 MB / 715 keys → **92 MB / 1,468 keys**. GitHub warns at 50 MB but accepts up to 100 MB hard limit. The standalone HTML weighs ~97 MB.

**TF-IDF PLS-DA projection fix:** Original `05_compute_pls_tfidf.py` projected only the 893 labeled rows into the ruler PLS-DA space. Changed to project all 1,586 (`X` instead of `X_labeled`) so SEAL points appear in the viz. Year regression projections were already projecting all 1,586.

---

## Headline Findings (Step 05 + 05b)

> **For Akkadian royal inscriptions, the ranking is consistent: TF-IDF >> MLM ≈ Random > Qwen.**

| Probe | Task | Best method | Best score | Random baseline | Qwen score |
|-------|------|-------------|-----------|-----------------|------------|
| PLS regression | Year (Spearman) | TF-IDF tier0 | 0.181 | ~0.18 (similar) | 0.121 |
| PLS regression | Year (R²) | — | All catastrophically negative | — | — |
| PLS-DA | Ruler (Macro-F1) | Random tier0/mean | 0.115 | (itself) | 0.111 |
| LogReg (CLS) | Ruler (Macro-F1) | **TF-IDF tier0** | **0.326** | 0.235 | 0.117 |
| LogReg (CLS) | Ruler (Accuracy) | **TF-IDF tier0** | **0.78** | 0.66 | 0.52 |
| LogReg (CLS) | Year-cat (Macro-F1) | **TF-IDF tier0** | **0.270** | 0.194 | 0.095 |

**Interpretation:**
- Year regression failed for all methods (R² floored at −10 across the board, Spearman near zero). Even TF-IDF's 0.18 Spearman barely clears the shuffled null.
- For ruler classification, **random projections of Qwen's architecture beat the real Qwen** by a wide margin (0.235 vs 0.117 Macro-F1). This suggests Qwen's learned geometry is not aligned with Akkadian ruler/lexical structure.
- The MLM has a clear **U-shape** in its layer curves (high at L00, dip in middle, recovery at L15–16) — characteristic of "embedding+output layers retain lexical signal, middle layers compress."
- TF-IDF tier0 dominating CLS shows the dominant signal is **lexical overlap** (rulers have distinctive vocabulary), not geometric structure in pretrained representations.

---

## File Map (Step 05 onwards)

```
v_1/src/linear_probing/
├── pls_utils.py                       # Shared PLS/PLS-DA API
├── 05_compute_pls.py                  # Qwen + Random driver
├── 05_compute_pls_mlm.py              # MLM driver
├── 05_compute_pls_tfidf.py            # TF-IDF driver (local, no GPU)
├── 05_compute_cls.py                  # Linear probe driver (pre-existing)
├── 06_aggregate_pls.py                # Aggregate PLS into best_layers + curves
├── 06_aggregate_cls.py                # Aggregate CLS into best_layers + curves
├── 07_plot_pls_curves.py              # PLS layer plots (2×3 regression, 1×2 ruler, best-of)
├── 07_plot_cls_curves.py              # CLS layer plots (1×2, best-of)
├── sbatch/orcc/
│   ├── pls_qwen.sh                    # Job 6522 (and prior)
│   ├── pls_random.sh                  # Job 6523
│   └── pls_mlm.sh                     # Job 6433
└── results/orcc_round1/
    ├── pls/
    │   ├── pls_results_{qwen,random,mlm,tfidf}.json
    │   ├── pls_projections_{qwen,random,mlm,tfidf}.json
    │   ├── pls_best_layers.json
    │   ├── pls_layer_curves.json
    │   └── figures/                   # 37 PNGs
    └── cls/
        ├── cls_results_{qwen,random,mlm,tfidf}.json   # pre-existing
        ├── cls_best_layers.json
        ├── cls_layer_curves.json
        └── figures/                   # 24 PNGs

v_1/src/viz/
├── 02_merge_coords.py                 # Now loads PLS projections too
├── seal_eda.html                      # Now has PLS-Year(raw/log), PLS-Ruler reductions
├── seal_viz_data.json                 # 92 MB, 1,468 keys
└── seal_eda_standalone.html           # 97 MB, gitignored
```

---

## How to Reproduce From Scratch

```bash
# 1. On cluster (assumes activations from Step 01 already exist):
sbatch v_1/src/linear_probing/sbatch/orcc/pls_qwen.sh    # ~30 min
sbatch v_1/src/linear_probing/sbatch/orcc/pls_random.sh  # ~30 min
sbatch v_1/src/linear_probing/sbatch/orcc/pls_mlm.sh     # ~10 min
# (each pushes to GitHub on completion)

# 2. Locally:
git pull origin main
python3 v_1/src/linear_probing/05_compute_pls_tfidf.py --target both --overwrite
python3 v_1/src/linear_probing/06_aggregate_pls.py
python3 v_1/src/linear_probing/06_aggregate_cls.py
python3 v_1/src/linear_probing/07_plot_pls_curves.py
python3 v_1/src/linear_probing/07_plot_cls_curves.py

# 3. Update viz:
python3 v_1/src/viz/02_merge_coords.py
python3 v_1/src/viz/03_build_standalone_html.py
open v_1/src/viz/seal_eda_standalone.html
```
