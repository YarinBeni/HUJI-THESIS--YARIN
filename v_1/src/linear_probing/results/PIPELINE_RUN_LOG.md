# Linear Probing Pipeline — Run Log

**Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, hidden dim 3584)
**Cluster:** Schmidt Sciences HPC, H100 80GB
**Data:** 4,957 Akkadian letters — OB=1,497 | NA=2,435 | LB=1,025

---

## Step 00 — Tokenization Check
**Job:** 2030 | **Node:** g0374 | **Date:** 2026-03-26 10:19–10:20 UTC
**Status:** ✅ SUCCESS

### Overall Stats
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

### Observations
- **0 unknown / 0 byte-fallback tokens** — Qwen handles Akkadian Unicode characters without errors
- **Tokens are byte-level fragments** (e.g. `'Å¡'`, `'á¹£'`) — model treats Akkadian as raw bytes, no linguistic knowledge of the language. Expected for an OOD script.
- **Some texts will be truncated** at the 512-token max-length limit in `01_extract.sh` (max text is 3,873 tokens)
- **Minor period differences in token length** (OB=276, NA=256, LB=285) — small enough not to be alarming as a confound

---

## Step 00b — Quick EDA (Final Layer Embeddings)
**Job:** 2031 | **Node:** g0374 | **Date:** 2026-03-26 10:24–10:32 UTC (7.7 min)
**Status:** ✅ SUCCESS

### What it did
Loaded the full Qwen2.5-7B-Instruct model (7B weights, ~15GB, took 5:48 to load), ran all 4,957 texts through it in batches, extracted the **final layer** mean-pooled embeddings, then ran PCA and t-SNE (2D) to visualize period clustering.

### Results
| Metric | Value |
|--------|-------|
| Embeddings shape | (4957, 3584) |
| Batches processed | 620/620 |
| UMAP | Skipped (not installed) |
| Plot saved | `results/plots/quick_eda_final_layer.png` |
| Embeddings saved | `results/activations/qwen2.5-7b-instruct/final_layer_only.npz` |

### Observations
- Model loaded and ran successfully on H100 with no OOM errors
- All 4,957 texts processed without errors
- Plot shows final-layer PCA + t-SNE colored by OB/NA/LB — pull to desktop to inspect clustering
- UMAP not installed in thesis env (minor, PCA + t-SNE sufficient)

---

## Step 01 — Extract Activations
*Not yet run*

---

## Step 02 — Linear Probe
*Not yet run*

---

## Step 03 — Analyze Results
*Not yet run*
