# SEAL Corpus Embedding Explorer — Documentation

> **See also:** [../linear_probing/results/PIPELINE_RUN_LOG.md](../linear_probing/results/PIPELINE_RUN_LOG.md) Step 08 (PLS reductions exposed in the GUI) · [../../PROGRESS.md](../../PROGRESS.md) (project status)

> **Status:** v4 complete (2026-05-11). Data: 1,586 fragments × **1,468 embedding keys**, ~92 MB.
> **GUI file:** `seal_eda.html` — open via local server (below) or use the self-contained `seal_eda_standalone.html`
> **Data file:** `seal_viz_data.json` (~92 MB, committed to repo with GitHub size warning)
>
> **v4 additions (2026-05-11):** PLS-Year(raw), PLS-Year(log), PLS-Ruler reductions added — supervised projections from the ORCC PLS pipeline (see `v_1/src/linear_probing/results/PIPELINE_RUN_LOG.md` Step 08).

---

## What Is This?

An interactive scatter-plot tool for visually exploring how different embedding methods represent cuneiform tablet fragments from the SEAL/DLL/LBPL corpora (Akkadian letters, 384 fragments) and the ORCC corpus (Royal Inscriptions, 1,202 fragments). Each point is one fragment; you choose how to embed it, which layer to look at, and how to color the points.

The main research question it helps answer: **do fragments cluster by period, genre, or provenance — and does that clustering depend on the embedding method?**

---

## How to Run

### Option A — Standalone (no server needed)
```bash
open v_1/src/viz/seal_eda_standalone.html
```
All data is inlined. Double-click works. Rebuilt via `python3 v_1/src/viz/03_build_standalone_html.py`.

### Option B — Local server (uses `seal_viz_data.json` from disk)
```bash
cd v_1/src/viz
python3 -m http.server 8000
# then open: http://localhost:8000/seal_eda.html
```

---

## GUI Controls

### Method
| Option | Description |
|--------|-------------|
| **TF-IDF** | Bag-of-words TF-IDF on character n-grams (2,5). Baseline; no neural network. SEAL only. |
| **Qwen** | Hidden-state embeddings from Qwen2.5-7B-Instruct (pretrained LLM, 7B params, 29 layers). |
| **Random Qwen** | Same architecture as Qwen but with random weights. Isolates architecture-only structure. |
| **Yarin MLM** | Embeddings from a custom 36.7M-parameter Akkadian MLM trained from scratch. 16 layers, sign-level tokenizer. SEAL + ORCC, mean pooling only. |

### Pooling
| Button | Description |
|--------|-------------|
| **Mean** | Average all token hidden states (weighted by attention mask). |
| **Last token** | Hidden state of the last non-padding token. Available for Qwen and Random only. |

TF-IDF and Yarin MLM are always mean-pooled (Last token button is disabled for them).

### Layer (slider)
- **TF-IDF**: disabled
- **Qwen / Random**: layers 0–28
- **Yarin MLM**: layers 0, 4, 8, 12, 16

### Cleaning
| Button | Internal name | What it does |
|--------|--------------|--------------|
| **Raw Text** | `tier0` | Strips ORACC `@v` markup, non-breaking spaces, subscript-x (U+2093). |
| **Cleaned** | `maximal` | Raw Text + 11 aggressive filters: strips digits, truncates to 30 tokens, removes case endings, logograms, determinatives; keeps only syllabic tokens; normalizes long vowels; lowercases. |

Yarin MLM only has Raw Text. Cleaned embeddings for MLM would require a cluster re-run.

### Reduction
| Button | Description |
|--------|-------------|
| **t-SNE** | Non-linear; preserves local neighborhoods. Good for clusters. Axes have no meaning. perplexity=30, max_iter=1000, random_state=42. |
| **PCA** | Linear; preserves global variance. |
| **UMAP** | Non-linear; preserves both local and global structure. Available for Qwen, Random (mean and last-token), and Yarin MLM. n_neighbors=15, min_dist=0.1, random_state=42. |
| **PLS-Year** | **Supervised.** PLS regression components 1&2 trained to predict `year` (raw scale) from L2-normalized activations, GroupKFold by ruler. Axes encode predicted-date direction. Fitted on 893 labeled ORCC rows; all 1,586 fragments projected. |
| **PLS-Year(log)** | Same as PLS-Year with target = `log(year)`. |
| **PLS-Ruler** | **Supervised.** PLS-DA components 1&2 trained to classify 38 rulers via one-hot regression + argmax, StratifiedKFold. |

UMAP is not available for TF-IDF (sparse vectors; tsne+pca only).
PLS reductions are available for all methods but TF-IDF has only one "layer" (L00).

### Show Datasets (corpus filter)
Toggle checkboxes for SEAL, DLL, LBPL, ORCC. All start active. Hiding a corpus removes those points from the plot without reloading data.

### Normalize
Signed log transform on both axes: `sign(x) × log(1 + |x|)`. Useful when one outlier compresses the main cluster.

### Color by
Options: `domain`, `period`, `genre`, `sub_genre`, `provenance`, `ruler (ORCC)`, `year (ORCC)`.

Year coloring uses a continuous Viridis colorscale (reversed: older = darker). Fragments without a year are shown in grey.

---

## Data: `seal_viz_data.json`

Schema:
```json
{
  "fragments": [
    {
      "fragment_id": "12345",
      "corpus": "seal",
      "period": "Old Babylonian",
      "genre": "letter",
      "sub_genre": "royal letter",
      "provenance": "Nippur",
      "sub_provenance": "...",
      "domain": "administrative",
      "word_count": 47,
      "text_snippet": "a-na be-li-ia...",
      "year": null,
      "ruler": null
    },
    {
      "fragment_id": "Q003333",
      "corpus": "orcc",
      "period": "Neo-Assyrian",
      "genre": "Royal Inscription",
      "domain": "ORCC",
      "ruler": "Esarhaddon",
      "year": 673,
      ...
    }
  ],
  "embeddings": {
    "tfidf__tier0__na__tsne":         [[x,y], ...],   // 1586 pairs, both SEAL and ORCC
    "qwen__tier0__L15__tsne":         [[x,y], ...],   // mean pooling
    "qwen__tier0__L15__last__tsne":   [[x,y], ...],   // last-token pooling
    "qwen__tier0__L15__umap":         [[x,y], ...],   // mean + UMAP
    "qwen__tier0__L15__last__umap":   [[x,y], ...],   // last-token + UMAP
    ...
  }
}
```

Key naming: `{method}__{cleaning}__{layer}__{reduction}` (mean pooling)
            `{method}__{cleaning}__{layer}__last__{reduction}` (last-token pooling)

For PLS reductions on TF-IDF, the layer slot is `L00` (not `na`) — `buildKey()` in `seal_eda.html` handles this special case.

Null-padding: SEAL-only methods (TF-IDF, MLM) have `[null, null]` at ORCC positions. The HTML skips null entries when rendering.

**Key counts by method (v4):**

| Method | Unsupervised (tsne/pca/umap) | PLS (pls12-raw/log + plsda12) | Total |
|--------|---:|---:|---:|
| tfidf  | 4    | 6   | 10 |
| mlm    | 15   | 51  | 66 |
| qwen   | 348+348  | 348 | 1,044 |
| random | 348+348  | 348 | 1,044 |
| **Total** | | | **~1,468** |

**Known gaps (not yet computed):**
- MLM UMAP: activations exist on cluster; coord re-run needed with `--include-umap`
- TF-IDF for ORCC: sklearn job, no GPU needed
- MLM for ORCC: needs GPU job (Akkadian MLM on 1,202 ORCC fragments)

---

## Pipeline: How the Data Was Built

All scripts run from repo root.

### Step 0 — Inspect raw data
```
src/corpus/01_inspect_seal_data.py
```

### Step A — Build corpus parquets
```bash
python v_1/src/corpus/02_build_seal_corpus.py          # → seal_corpus.parquet (384 frags)
python v_1/src/corpus/03_build_orcc_corpus.py          # → orcc_corpus.parquet (1202 frags)
```

### Step B — TF-IDF coords (local, SEAL only)
```bash
python v_1/src/viz/01_compute_tfidf_coords.py
```
> **IMPORTANT:** Do not re-run after Step F — it overwrites `seal_viz_data.json`.

### Step C — Qwen + Random mean-pooled (cluster, SEAL)
```
sbatch v_1/src/linear_probing/sbatch/seal/extract_qwen_tier0.sh      # jobs 2994–2997
sbatch v_1/src/linear_probing/sbatch/seal/compute_2d_coords.sh       # job 3029
```
Outputs `seal_round4/seal_qwen_coords.json` (232 keys: tsne+pca, mean pooling).

### Step D — Yarin MLM (cluster, SEAL)
```
sbatch v_1/src/linear_probing/sbatch/seal/train_mlm.sh               # job 2998
sbatch v_1/src/linear_probing/sbatch/seal/extract_mlm_embeddings.sh  # job 3028
```
Outputs `seal_round4/seal_mlm_coords.json` (10 keys: tsne+pca, tier0 only).

### Step E — Last-token + UMAP for SEAL (cluster)
```
sbatch v_1/src/linear_probing/sbatch/seal/extract_qwen_tier0_last.sh     # job 4887
sbatch v_1/src/linear_probing/sbatch/seal/extract_qwen_maximal_last.sh   # job 4888
sbatch v_1/src/linear_probing/sbatch/seal/extract_random_tier0_last.sh   # job 4889
sbatch v_1/src/linear_probing/sbatch/seal/extract_random_maximal_last.sh # job 4890
sbatch v_1/src/linear_probing/sbatch/seal/compute_umap_coords_last.sh    # job 4906
```
Outputs `seal_round4/seal_qwen_coords_last.json` (348 keys: tsne+pca+umap, last-token).

### Step F — ORCC extraction (cluster)
```
sbatch v_1/src/linear_probing/sbatch/orcc/extract_qwen_tier0.sh      # job 4900
sbatch v_1/src/linear_probing/sbatch/orcc/extract_qwen_maximal.sh    # job 4901
sbatch v_1/src/linear_probing/sbatch/orcc/extract_random_tier0.sh    # job 4902
sbatch v_1/src/linear_probing/sbatch/orcc/extract_random_maximal.sh  # job 4903
sbatch v_1/src/linear_probing/sbatch/orcc/compute_2d_umap_coords.sh  # job 4908
```
Outputs `orcc_round1/orcc_qwen_coords_mean.json` + `orcc_qwen_coords_last.json` (348 keys each).

### Step G — Merge all coords (local)
```bash
python v_1/src/viz/02_merge_coords.py
```
Merges all coord JSONs + ORCC parquet metadata + the 4 `pls_projections_*.json` files into `seal_viz_data.json` (1,468 keys, 92 MB, 1,586 fragments).

The PLS step filters to `pls12-raw`, `pls12-log`, `plsda12` keys only (skips `pls23`/`pls34` for size). Fragment-ID validation normalizes both sides to strings since the viz stores SEAL IDs as ints.

### Step H — Build standalone HTML (local)
```bash
python v_1/src/viz/03_build_standalone_html.py
```
Inlines `seal_viz_data.json` into `seal_eda_standalone.html` (46 MB). This file is gitignored (too large).

---

## Files in This Directory

| File | Description |
|------|-------------|
| `seal_eda.html` | The GUI — open via server or use standalone |
| `seal_eda_standalone.html` | Self-contained HTML — gitignored, rebuild locally |
| `seal_viz_data.json` | All coords + metadata (44 MB, 710 keys, 1,586 fragments) |
| `01_compute_tfidf_coords.py` | Builds TF-IDF embeddings + initializes seal_viz_data.json |
| `02_merge_coords.py` | Merges all cluster coord JSONs + ORCC metadata |
| `03_build_standalone_html.py` | Inlines seal_viz_data.json into standalone HTML |

---

## Related Files

| File | Description |
|------|-------------|
| `data/evaluation/corpora/seal_corpus.parquet` | 384 SEAL/DLL/LBPL fragments |
| `data/evaluation/corpora/orcc_corpus.parquet` | 1,202 ORCC Royal Inscription fragments |
| `data/raw/chungrong/orcc_round1/royal_inscriptions.csv` | Raw ORCC source (1,202 frags) |
| `src/linear_probing/results/seal__embed/seal_qwen_coords.json` | Qwen+Random SEAL mean (232 keys) |
| `src/linear_probing/results/seal__embed/seal_mlm_coords.json` | MLM SEAL (10 keys) |
| `src/linear_probing/results/seal__embed/seal_qwen_coords_last.json` | Qwen+Random SEAL last-token (348 keys) |
| `src/linear_probing/results/orcc__embed/orcc_qwen_coords_mean.json` | ORCC mean-pooled (348 keys) |
| `src/linear_probing/results/orcc__embed/orcc_qwen_coords_last.json` | ORCC last-token (348 keys) |
| `src/linear_probing/sbatch/seal/` | All SEAL sbatch scripts |
| `src/linear_probing/sbatch/orcc/` | All ORCC sbatch scripts |
| `src/corpus/02_build_seal_corpus.py` | SEAL corpus builder |
| `src/corpus/03_build_orcc_corpus.py` | ORCC corpus builder |
