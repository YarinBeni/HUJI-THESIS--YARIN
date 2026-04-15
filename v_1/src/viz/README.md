# SEAL Corpus Embedding Explorer — Documentation

> **Status:** Complete (2026-04-15). All embedding methods live. Data: 384 fragments × 246 embedding keys.
> **GUI file:** `seal_eda.html` (open via local server — see below)
> **Data file:** `seal_viz_data.json` (4 MB, committed to repo)

---

## What Is This?

An interactive scatter-plot tool for visually exploring how different embedding methods represent 384 cuneiform tablet fragments from the SEAL/DLL/LBPL corpora. Each point is one fragment; you choose how to embed it, which layer to look at, and how to color the points.

The main research question it helps answer: **do fragments cluster by period, genre, or provenance — and does that clustering depend on the embedding method?**

---

## How to Run Locally

From the repo root:

```bash
cd v_1/src/viz
python3 -m http.server 8000
# then open: http://localhost:8000/seal_eda.html
```

A local server is required because the browser blocks loading `.json` from the filesystem directly (CORS). You cannot just double-click `seal_eda.html`.

---

## GUI Controls

### Method
Which embedding model produced the coordinates:

| Option | Description |
|--------|-------------|
| **TF-IDF** | Bag-of-words TF-IDF on character n-grams (2,5). No neural network. Baseline. |
| **Qwen** | Hidden-state embeddings from Qwen2.5-7B-Instruct (pretrained LLM, 7B params, 29 layers). Mean-pooled over tokens. |
| **Random Qwen** | Same architecture as Qwen but with random weights. Isolates what structure comes from the model's pre-training vs. the architecture alone. |
| **Yarin MLM** | Embeddings from a custom 36.7M-parameter Akkadian MLM trained from scratch on 40k+ cuneiform fragments. 16 layers, sign-level tokenizer (14,797 signs). Mean-pooled. |

### Layer (slider)
Which transformer layer's hidden states to use:
- **TF-IDF**: no layer concept — slider is disabled
- **Qwen / Random**: layers 0–28 (0 = after embedding, 28 = last layer before head)
- **Yarin MLM**: layers 0, 4, 8, 12, 16 (slider snaps to nearest valid step)

### Cleaning
Which text preprocessing was applied before embedding:

| Button | Internal name | What it does |
|--------|--------------|--------------|
| **Raw Text ⓘ** | `tier0` | Strips ORACC `@v` markup, non-breaking spaces, and subscript-x (U+2093). Otherwise the text is as found in the database — transliterated cuneiform with logograms, determinatives, and case endings intact. Example: `a-na dUTU LUGAL qi2-bi-ma` → `a-na dUTU LUGAL qi-bi-ma` |
| **Cleaned ⓘ** | `maximal` | Applies Raw Text + 11 aggressive linguistic filters: strips all digits, truncates to 30 tokens, removes case endings (`-am/-im/-um/-šum`), removes `w`/`y`, removes logograms (ALL-CAPS words like `LUGAL`, `UTU`), strips determinatives (`d-`, `giš-`, `uru-`, `lu2-`), keeps only syllabic (lowercase Akkadian) tokens, normalizes long vowels (`ā→a`, `ī→i`, `ū→u`, `ē→e`), strips subscript digits (`ba2→ba`), lowercases, strips plural `-meš`. Example: `d-UTU LUGAL qa2-bu-um` → `qabu` |

> **Note:** Yarin MLM only has Raw Text embeddings. Cleaned embeddings for MLM would require a cluster re-run.

### Reduction
Which 2D projection algorithm was used:

| Button | Description |
|--------|-------------|
| **t-SNE** | Non-linear; preserves local neighborhood structure. Good for seeing clusters. Axes have no absolute meaning. |
| **PCA** | Linear; preserves global variance. Axes represent principal components. |

Parameters: t-SNE uses perplexity=30, max_iter=1000, random_state=42. PCA uses 2 components, random_state=42.

### Normalize
Applies a signed log transform to both axes: `sign(x) × log(1 + |x|)`. Useful when one outlier fragment is very far from the rest and compresses the main cluster into a tiny dot. Does not change point ordering or group membership — only the visual scale.

### Color by
Which metadata column is used to color the points. Options: `domain`, `period`, `genre`, `sub_genre`, `provenance`.

### Hover tooltip
Hovering over any point shows: fragment_id, corpus (SEAL/DLL/LBPL), period, genre, sub_genre, provenance, word count, and a text snippet.

---

## Data: `seal_viz_data.json`

Schema:
```json
{
  "fragments": [
    {
      "fragment_id": 12345,
      "corpus": "seal",
      "period": "Old Babylonian",
      "genre": "letter",
      "sub_genre": "royal letter",
      "provenance": "Nippur",
      "sub_provenance": "...",
      "domain": "administrative",
      "word_count": 47,
      "text_snippet": "a-na be-li-ia..."
    },
    ...   // 384 entries, one per fragment
  ],
  "embeddings": {
    "tfidf__tier0__na__tsne":  [[x,y], [x,y], ...],  // 384 pairs
    "qwen__tier0__L00__tsne":  [[x,y], ...],
    "qwen__tier0__L00__pca":   [[x,y], ...],
    ...
    "mlm__tier0__L00__tsne":   [[x,y], ...],
    ...
    // 246 keys total
  }
}
```

Key naming convention: `{method}__{cleaning}__{layer}__{reduction}`
- layer is `na` for TF-IDF, `L00`–`L28` for Qwen/Random, `L00/L04/L08/L12/L16` for MLM

**Total keys by method:**

| Method | Keys | Layers | Cleanings | Reductions |
|--------|-----:|-------:|----------:|-----------:|
| tfidf  | 4    | 1 (na) | 2         | 2          |
| qwen   | 116  | 29     | 2         | 2          |
| random | 116  | 29     | 2         | 2          |
| mlm    | 10   | 5      | 1 (tier0) | 2          |
| **Total** | **246** | | | |

---

## Pipeline: How the Data Was Built

All scripts run from repo root. Full execution log: `PROGRESS.md`.

### Step 0 — Inspect raw data
```
src/corpus/01_inspect_seal_data.py
```
Profiles the three source CSVs (`seal.csv`, `dll.csv`, `lbpl.csv`), checks columns and fragment counts, writes `data/raw/chungrong/seal_round4/inspection_report.json` (MD5 hashes + data contract).

### Step A — Build corpus parquet
```
src/corpus/02_build_seal_corpus.py
```
Aggregates 40,484 word-level rows into 384 fragment-level rows. Computes `text_tier0` and `text_maximal` columns. Outputs `data/evaluation/corpora/seal_corpus.parquet` (384 rows × 15 columns).

### Step B — TF-IDF coords (local)
```
src/viz/01_compute_tfidf_coords.py
```
Fits TF-IDF (char_wb, ngram 2–5) on the 384 fragments, runs t-SNE + PCA, writes `seal_viz_data.json` with 4 TF-IDF keys + fragment metadata.

> **IMPORTANT:** Do not re-run this after Step E — it would overwrite `seal_viz_data.json` and erase all other embedding keys.

### Step C — Qwen + Random embeddings (cluster)
```
# Extract activations for 4 combinations (tier0/maximal × pretrained/random):
sbatch v_1/src/linear_probing/sbatch/seal/extract_qwen_embeddings.sh   # jobs 2994–2997

# Compute 2D coords (CPU):
sbatch v_1/src/linear_probing/sbatch/seal/compute_2d_coords.sh         # job 2999/3029
```
Outputs `results/seal_round4/seal_qwen_coords.json` (232 keys, 3.6 MB), nested under `{"embeddings": {...}}`.

### Step D — Yarin MLM training + embeddings (cluster)
```
# Train Akkadian MLM from scratch (H100, ~28 min):
sbatch v_1/src/linear_probing/sbatch/seal/train_mlm.sh                 # job 2998

# Extract MLM embeddings for 384 SEAL fragments (CPU, ~46s):
sbatch v_1/src/linear_probing/sbatch/seal/extract_mlm_embeddings.sh    # job 3028
```
The MLM (`v_1/src/archive/baseline_mlm/`) is a 16-layer transformer (d_model=384, d_ff=1536, 8 heads, 36.7M params) trained on Akkadian cuneiform at sign level. Checkpoint: `v_1/models/baseline_retrained/baseline_best.pt` (420 MB, val_loss=2.9777).

**Tokenization note:** SEAL corpus stores word-level transliterations (`GAB-RI`); the MLM tokenizer expects sign-level (`GAB RI`). Fix: `text.replace('-', ' ')` before tokenizing. This is implemented in `03_extract_seal_embeddings.py`.

The extraction script (`src/archive/baseline_mlm/03_extract_seal_embeddings.py`) outputs `results/seal_round4/seal_mlm_coords.json` (10 keys, 158 KB), flat structure.

### Step E — Merge all coords (local)
```
python v_1/src/viz/02_merge_coords.py
```
Merges TF-IDF base + Qwen/Random (unwrapping the nested `"embeddings"` key) + MLM (flat) into the final `seal_viz_data.json` (246 keys, 4 MB).

---

## Files in This Directory

| File | Description |
|------|-------------|
| `seal_eda.html` | The GUI — open via `python3 -m http.server 8000` |
| `seal_viz_data.json` | All embedding coords + fragment metadata (4 MB, 246 keys) |
| `01_compute_tfidf_coords.py` | Builds TF-IDF embeddings + initializes seal_viz_data.json |
| `02_merge_coords.py` | Merges Qwen/Random/MLM coords from cluster into seal_viz_data.json |

---

## Related Files (outside this directory)

| File | Description |
|------|-------------|
| `data/evaluation/corpora/seal_corpus.parquet` | Source data: 384 fragments × 15 columns |
| `src/linear_probing/results/seal_round4/seal_qwen_coords.json` | Qwen+Random 2D coords (232 keys) |
| `src/linear_probing/results/seal_round4/seal_mlm_coords.json` | MLM 2D coords (10 keys) |
| `src/archive/baseline_mlm/03_extract_seal_embeddings.py` | MLM extraction script |
| `src/linear_probing/sbatch/seal/` | All sbatch scripts for SEAL cluster jobs |
| `PROGRESS.md` | Full execution history with job IDs and results |
| `justification/parallel_plans_final.md` | Detailed parallel execution plan (Plans A–E) |
