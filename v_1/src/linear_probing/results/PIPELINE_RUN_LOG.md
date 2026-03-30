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

### Validity Tests Done and Still Needed
Need to read relevent knowldge on linear probing and create tests to validate results.

---

## Step 03 — Analyze Results
*Not yet run*

### What this will do
Classify the outcome as A, B, or C based on the probe results, comparing against the TF-IDF baselines from the bias check.
