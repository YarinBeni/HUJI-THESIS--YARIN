# Viz Extension Plan — 5 New Features
**Date:** 2026-04-21  
**Status:** ✅ COMPLETE (2026-04-28) — all 5 features + 3 data gaps closed, 715 keys, 1586 fragments, ~45MB data

---

## Feature Summary

| # | Feature | Complexity | Data deps | Cluster? |
|---|---------|------------|-----------|----------|
| 1 | ORCC (Royal Inscriptions) through full pipeline | High | ✅ File at `v_1/data/raw/chungrong/orcc_round1/royal_inscriptions.csv` | ✅ GPU + CPU |
| 2 | Year-based continuous color scale (ORCC sub_period) | Low | Depends on #1 | ❌ |
| 3 | Dataset source filter dropdown (SEAL/DLL/LBPL/ORCC) | Low | HTML works immediately; full data after #1 | ❌ |
| 4 | UMAP dimension reduction | Medium | Needs .npz files on cluster | ⚠️ CPU cluster |
| 5 | Pooling method dropdown (mean vs last-token) | Medium | New GPU extraction jobs | ✅ GPU |

---

## ORCC Dataset Facts

| Property | Value |
|---------|-------|
| File | `v_1/data/raw/chungrong/orcc_round1/royal_inscriptions.csv` |
| Schema | Same 16 columns as seal.csv — `sub_period` is newly populated |
| Fragments | 1,202 unique (`fragment_id` like `Q003333`) |
| Words | ~245,673 rows |
| Language | All `akk` |
| Genre | `Royal Inscription` (some null) |
| Period | Neo-Assyrian (939), Neo-Babylonian (217), Middle Babylonian (29), ... |
| Domain | Ruler-based e.g. `ribo(Esarhaddon)`, `rinap4(Esarhaddon)`, ... (46 unique values) |
| Sub_period | 893/1202 fragments have a date, e.g. `"ca. 674-673"`, `"ca. 625–605"` |
| Year range | 7–1,132 BCE (taking min of all digits in sub_period string) |
| Corpus tag | Will be `"orcc"`, domain `"ORCC"` |

**Year extraction rule:** `year = min(all integers in sub_period string)` — "ca. 674-673" → 673, "ca. 704-703, 688-681" → 681. Handles both `-` and `–` (en-dash).

**Note on `domain`:** The raw `domain` values encode the ruler/project (e.g. `ribo(Esarhaddon)`). We keep the raw value as the `domain` field in the parquet, and in the fragment metadata add a `ruler` field that strips the prefix to just the king name. This gives a useful coloring axis.

---

## Schema Design

### Key naming — backward-compatible extension

**Existing format (keep as-is):** `{method}__{cleaning}__{layer}__{reduction}`  
e.g. `qwen__tier0__L15__tsne`

**New key rules:**

```
Last-token pooling:  {method}__{cleaning}__{layer}__last__{reduction}
UMAP (mean):         {method}__{cleaning}__{layer}__umap
UMAP (last-token):   {method}__{cleaning}__{layer}__last__umap
```

- TF-IDF has no pooling axis — no `__last` variant ever
- Existing 246 keys are untouched (implicitly mean-pooled)
- HTML `buildKey()` inserts `__last` before reduction when `pooling === "last"`

### Fragment metadata — new fields

```json
{
  "fragment_id": "Q003333",
  "corpus":      "orcc",
  "domain":      "ribo(Esarhaddon)",
  "ruler":       "Esarhaddon",       ← NEW (extracted from domain)
  "year":        673,                 ← NEW (null for SEAL/DLL/LBPL; null for ORCC without sub_period)
  "period":      "Neo-Assyrian",
  "genre":       "Royal Inscription",
  "sub_genre":   null,
  "provenance":  "Babylon",
  ...
}
```

SEAL/DLL/LBPL fragments keep `year: null` and `ruler: null`.

---

## File Change Map

### New files to CREATE

| File | Purpose |
|------|---------|
| `v_1/src/corpus/03_build_orcc_corpus.py` | Build ORCC parquet; parse year; extract ruler from domain |
| `v_1/src/linear_probing/sbatch/seal/extract_qwen_tier0_last.sh` | SEAL last-token Qwen tier0 |
| `v_1/src/linear_probing/sbatch/seal/extract_qwen_maximal_last.sh` | SEAL last-token Qwen maximal |
| `v_1/src/linear_probing/sbatch/seal/extract_random_tier0_last.sh` | SEAL last-token Random tier0 |
| `v_1/src/linear_probing/sbatch/seal/extract_random_maximal_last.sh` | SEAL last-token Random maximal |
| `v_1/src/linear_probing/sbatch/seal/compute_umap_coords_last.sh` | CPU: SEAL last-token t-SNE+PCA+UMAP |
| `v_1/src/linear_probing/sbatch/orcc/extract_qwen_tier0.sh` | ORCC Qwen tier0 (mean + last in one job) |
| `v_1/src/linear_probing/sbatch/orcc/extract_qwen_maximal.sh` | ORCC Qwen maximal |
| `v_1/src/linear_probing/sbatch/orcc/extract_random_tier0.sh` | ORCC Random tier0 |
| `v_1/src/linear_probing/sbatch/orcc/extract_random_maximal.sh` | ORCC Random maximal |
| `v_1/src/linear_probing/sbatch/orcc/compute_2d_umap_coords.sh` | CPU: ORCC all reductions |

### Existing files to MODIFY

| File | What changes |
|------|-------------|
| `v_1/src/linear_probing/03_extract_seal_activations.py` | Add `--pooling [mean\|last]`, add `--output-dir` override, remove hardcoded `assert len==384` |
| `v_1/src/linear_probing/04_compute_2d_coords.py` | Add `run_umap()`, add `--include-umap` flag, add `--input-dirs` + `--output-path` flags, generalize N_TEXTS |
| `v_1/src/viz/02_merge_coords.py` | Accept ORCC coord JSONs; merge ORCC fragment metadata; add `year` / `ruler` fields |
| `v_1/src/viz/seal_eda.html` | 4 new UI controls: dataset filter, UMAP button, pooling dropdown, continuous year coloring |

---

## Cluster Job Plan

**Key design:** Each ORCC GPU job runs BOTH mean and last-token pooling in a single forward pass (hidden states computed once). This halves GPU cost for ORCC.

### GPU Jobs — submit all 8 in parallel (~2 hrs each on H100)

| Job | Script | Dataset | Cleaning | Produces |
|-----|--------|---------|---------|---------|
| G1 | `seal/extract_qwen_tier0_last.sh` | SEAL (384) | tier0 | `seal_round4/activations/qwen_tier0_last/` |
| G2 | `seal/extract_qwen_maximal_last.sh` | SEAL | maximal | `seal_round4/activations/qwen_maximal_last/` |
| G3 | `seal/extract_random_tier0_last.sh` | SEAL | tier0 | `seal_round4/activations/random_tier0_last/` |
| G4 | `seal/extract_random_maximal_last.sh` | SEAL | maximal | `seal_round4/activations/random_maximal_last/` |
| G5 | `orcc/extract_qwen_tier0.sh` | ORCC (1202) | tier0 | `orcc_round1/activations/qwen_tier0_{mean,last}/` |
| G6 | `orcc/extract_qwen_maximal.sh` | ORCC | maximal | `orcc_round1/activations/qwen_maximal_{mean,last}/` |
| G7 | `orcc/extract_random_tier0.sh` | ORCC | tier0 | `orcc_round1/activations/random_tier0_{mean,last}/` |
| G8 | `orcc/extract_random_maximal.sh` | ORCC | maximal | `orcc_round1/activations/random_maximal_{mean,last}/` |

**ORCC note:** 1,202 texts is 3× SEAL. ORCC jobs may take up to 4 hrs — set `--time=04:00:00`.  
**SEAL last-token jobs:** 384 texts, same time as originals (2 hrs is fine).

### CPU Jobs — submit all in parallel after GPU jobs done (~1 hr each)

| Job | Script | Input | Produces |
|-----|--------|-------|---------|
| C1 | `seal/compute_umap_coords_last.sh` | SEAL last-token activation dirs (×4) | `seal_round4/seal_qwen_coords_last.json` |
| C2 | `orcc/compute_2d_umap_coords.sh` | ORCC mean+last activation dirs (×8) | `orcc_round1/orcc_qwen_coords_mean.json` + `orcc_round1/orcc_qwen_coords_last.json` |
| C3 *(conditional)* | `seal/compute_umap_existing_mean.sh` | Existing SEAL mean .npz dirs | `seal_round4/seal_qwen_coords_umap.json` (adds `__umap` keys to existing mean) |

> C3 only if existing .npz activation files still exist on cluster. Check with `ls ~/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/seal_round4/activations/qwen_tier0/layer_00.npz`.

---

## Dependency Graph

```
[RESOLVED: ORCC at v_1/data/raw/chungrong/orcc_round1/royal_inscriptions.csv]
         │
         ├──► [T1] HTML changes — start immediately, no data deps
         │
         └──► [T2] Script adaptation — start immediately, no data deps
                    ├── Adapt 03_extract_seal_activations.py (--pooling, --output-dir)
                    ├── Adapt 04_compute_2d_coords.py (UMAP, --input-dirs, --output-path)
                    ├── Create 03_build_orcc_corpus.py
                    ├── Create all 11 new sbatch scripts
                    └── Update 02_merge_coords.py

[T2 done] ──► [T3] Cluster orchestration
               ├── Build ORCC corpus locally (~2 min)
               ├── SSH + git pull
               ├── Check if SEAL mean .npz files exist on cluster
               ├── Submit G1–G8 in parallel (~2–4 hrs)
               ├── When GPU done → submit C1–C3 (~1 hr)
               └── rsync results back

[T1 done AND T3 done] ──► [T0-FINAL] Merge + build standalone + test
```

---

## Terminal Prompts

> Copy-paste these exactly. Stop conditions are explicit.

---

### TERMINAL 1 — HTML/JS Changes (start immediately)

```
You are working in /Users/yarin.b/git/lititure-review.

GOAL: Implement 4 new UI features in v_1/src/viz/seal_eda.html.
READ the file completely before editing. Do NOT touch any Python files.
Do NOT rebuild the standalone HTML (03_build_standalone_html.py).

────────────────────────────────────────────────────────────
FEATURE A — Dataset source filter

Add a new ctrl-group called "Show datasets" with toggle buttons, one per corpus:
  SEAL · DLL · LBPL · ORCC
All start active (selected). Clicking deselects that dataset (grey background).

State: state.visibleCorpora = new Set(["seal", "dll", "lbpl", "orcc"])

In render(): filter fragments before building traces:
  const visFragments = DATA.fragments.filter(f => state.visibleCorpora.has(f.corpus));
Use visFragments everywhere instead of DATA.fragments.
Important: the coords index must still match DATA.fragments — use the original index i
  for coords lookup even after filtering:
  visFragments.forEach(f => { const i = DATA.fragments.indexOf(f); xs.push(coords[i][0]); ... })
  Or: pre-compute fragIdx = fragments.map((f,i) => i), then filter with indices.

Wire: each button toggles its corpus in/out of state.visibleCorpora; active = selected.

────────────────────────────────────────────────────────────
FEATURE B — UMAP in Reduction toggle

Add a 3rd button "UMAP" to #tog-reduction (after PCA button).
Tooltip text:
  "<strong>UMAP</strong> — <em>non-linear</em> reduction that better preserves
  both local clusters AND global layout compared to t-SNE.
  <br><br>Settings: <code>n_neighbors=15, min_dist=0.1, random_state=42</code>.
  <br>Note: UMAP data is only available for new embeddings; older SEAL mean-pooled
  layers may show the 'unavailable' message."

state.reduction can now be "tsne" | "pca" | "umap" — no other logic change needed.

────────────────────────────────────────────────────────────
FEATURE C — Pooling dropdown

Add a new ctrl-group "Pooling" with a toggle group:
  button "Mean"       data-val="mean"  (default active)
  button "Last token" data-val="last"

State: state.pooling = "mean" | "last"

Update buildKey() to:
  function buildKey() {
    const { method, layer, cleaning, reduction, pooling } = state;
    const layerStr = (method === "tfidf") ? "na" : layerLabel(layer);
    if (method === "tfidf" || pooling === "mean") {
      return `${method}__${cleaning}__${layerStr}__${reduction}`;
    }
    return `${method}__${cleaning}__${layerStr}__last__${reduction}`;
  }

Disable the "Last token" button when method === "tfidf".
Update the unavailable-hint for missing last-token data:
  "Last-token embeddings are not yet computed for this combination. Switch to Mean pooling."

────────────────────────────────────────────────────────────
FEATURE D — Year-based continuous coloring

Add "year (ORCC)" option to #sel-colorby with value "year".

When state.colorBy === "year":
  Split DATA.fragments (filtered by visibleCorpora) into two groups:
    - withYear: fragments where f.year != null
    - noYear:   fragments where f.year == null

  Trace 1 (grey, rendered first):
    type: "scatter", mode: "markers"
    name: "no date", showlegend: true
    marker: { color: "#cccccc", size: 6, opacity: 0.5 }
    x, y from noYear fragments

  Trace 2 (colored by year):
    type: "scatter", mode: "markers"
    name: "dated", showlegend: false
    marker: {
      color: withYear.map(f => f.year),
      colorscale: "Viridis",
      reversescale: true,   // so older (larger BCE) = darker/purple, recent = yellow
      showscale: true,
      colorbar: { title: "Year (BCE)", thickness: 15, len: 0.6 },
      size: 7,
      opacity: 0.85,
      line: { width: 0.5, color: "rgba(0,0,0,0.2)" }
    }

  Hover template for year trace should include:
    "<b>year:</b> %{customdata.year} BCE<br>"

Make render() detect colorBy === "year" and branch to this logic instead of the
standard categorical trace-per-label loop.

────────────────────────────────────────────────────────────
ALSO:
- Update #subtitle to show fragment count dynamically:
    `${DATA.fragments.length} fragments · Akkadian cuneiform · Master's thesis EDA`
- Update #title-bar info tooltip to mention ORCC and Royal Inscriptions
- Add "ruler" to the sel-colorby options (label: "ruler (ORCC)")
- Wire state.pooling into the toggle system (same pattern as tog-cleaning)

STOP when: seal_eda.html is saved with all features. No Python files, no standalone build.
```

---

### TERMINAL 2 — Script Adaptation (start immediately)

```
You are working in /Users/yarin.b/git/lititure-review.

GOAL: Adapt existing Python scripts and create new ones. No cluster interaction.
READ each file before editing.

────────────────────────────────────────────────────────────
TASK A — Add --pooling to extraction script
  File: v_1/src/linear_probing/03_extract_seal_activations.py

  1. Read the file first. Note it imports mean_pool from utils.py.
  2. Add after the mean_pool import (or inline in the file if utils.py is local):
       def last_token_pool(hidden_states, attention_mask):
           # hidden_states: (batch, seq_len, hidden_dim)
           # attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
           # For each sequence, pick the hidden state at the last non-padding token
           seq_lens = attention_mask.sum(dim=1) - 1  # index of last real token
           batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
           return hidden_states[batch_idx, seq_lens]  # (batch, hidden_dim)
  3. Add argparse argument: --pooling with choices=["mean","last"], default="mean"
  4. Add argparse argument: --output-dir (optional, overrides the default SEAL_ACTS_DIR subdir)
  5. In the extraction loop, use pooling arg to choose between mean_pool and last_token_pool
  6. Remove the hardcoded line `assert len(df) == 384` (replace with print statement)
  7. Keep all other behavior identical

────────────────────────────────────────────────────────────
TASK B — Add UMAP + generalize coordinate script
  File: v_1/src/linear_probing/04_compute_2d_coords.py

  1. Read the file first. Note it has hardcoded ACTIVATION_CONFIGS and N_TEXTS=384.
  2. Add:
       def run_umap(X):
           import umap as umap_lib
           reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
           return reducer.fit_transform(X.astype(float))
  3. Add CLI args:
       --include-umap   (flag, default False)
       --input-dirs     (nargs='+', list of activation dir NAMES relative to SEAL_ACTS_DIR
                         — overrides default ACTIVATION_CONFIGS)
       --input-base     (optional, base directory containing the input dirs;
                         defaults to SEAL_ACTS_DIR)
       --output-path    (output JSON path; defaults to current OUTPUT_PATH)
       --method-tags    (nargs='+', list of "method__cleaning" tags, parallel to --input-dirs)
  4. When --include-umap: after running t-SNE and PCA for each layer,
     also run run_umap() and store key `{method}__{cleaning}__L{NN}__umap`
  5. Change N_TEXTS: set it dynamically from the first .npz file encountered (not hardcoded)
  6. When --input-dirs is given, build ACTIVATION_CONFIGS from those args + --method-tags

────────────────────────────────────────────────────────────
TASK C — Create ORCC corpus builder
  File to CREATE: v_1/src/corpus/03_build_orcc_corpus.py

  READ v_1/src/corpus/02_build_seal_corpus.py first — match the output parquet schema.
  
  Input: v_1/data/raw/chungrong/orcc_round1/royal_inscriptions.csv
  
  ORCC-specific logic:
  1. Load CSV; group by fragment_id to get fragment-level records
  2. Text aggregation: same as SEAL — join clean_value words, fallback to value when null
  3. Apply same text_tier0 and text_maximal cleaning as SEAL
     (READ 02_build_seal_corpus.py to copy the exact cleaning functions)
  4. Extract year: `min(all integers in sub_period string)` using re.findall(r'\d+', ...)
     If sub_period is null/no numbers → year = None
  5. Extract ruler: strip the prefix from domain — e.g. "ribo(Esarhaddon)" → "Esarhaddon",
     "rinap4(Esarhaddon)" → "Esarhaddon". Regex: re.search(r'\((.+?)\)', domain) → group(1).
     If no parentheses → ruler = domain value as-is (e.g. plain "ribo")
  6. Set corpus = "orcc", domain = "ORCC" (override the raw domain column in the parquet
     by renaming the raw value to a new column "ruler")
  7. Output columns: fragment_id, corpus, word_language, domain, ruler, period, genre,
     sub_genre, provenance, sub_provenance, word_count, text, text_tier0, text_maximal, year
  8. Output: v_1/data/evaluation/corpora/orcc_corpus.parquet
  9. Print summary: N fragments, N with year, year range, period distribution

────────────────────────────────────────────────────────────
TASK D — Create sbatch scripts

  Read v_1/src/linear_probing/sbatch/seal/extract_qwen_tier0.sh as template.
  Note the cluster project path: ~/projects/HUJI-THESIS--YARIN
  Conda env: thesis

  Create these files:

  A) SEAL last-token scripts (4 files):
     v_1/src/linear_probing/sbatch/seal/extract_qwen_tier0_last.sh
       Same as extract_qwen_tier0.sh but:
       - job-name: seal_qwen_tier0_last
       - python call adds: --pooling last --output-dir v_1/src/linear_probing/results/seal_round4/activations/qwen_tier0_last
       - log file: seal_qwen_tier0_last_%j.out

     v_1/src/linear_probing/sbatch/seal/extract_qwen_maximal_last.sh   (similar)
     v_1/src/linear_probing/sbatch/seal/extract_random_tier0_last.sh   (similar, uses 03b_extract_random_seal_activations.py)
     v_1/src/linear_probing/sbatch/seal/extract_random_maximal_last.sh (similar)

     NOTE: for random scripts, check if they call 03b_extract_random_seal_activations.py
           or 03_extract_seal_activations.py with a flag — read the existing random scripts first.

  B) SEAL CPU coord script:
     v_1/src/linear_probing/sbatch/seal/compute_umap_coords_last.sh
       #SBATCH --gres=gpu:0 (no GPU needed)
       #SBATCH --mem=32G, --time=01:30:00, --cpus-per-task=8
       Calls: python 04_compute_2d_coords.py \
         --include-umap \
         --input-dirs qwen_tier0_last qwen_maximal_last random_tier0_last random_maximal_last \
         --input-base v_1/src/linear_probing/results/seal_round4/activations \
         --method-tags qwen__tier0 qwen__maximal random__tier0 random__maximal \
         --output-path v_1/src/linear_probing/results/seal_round4/seal_qwen_coords_last.json
       The method-tags here signal that these are LAST-TOKEN — hmm, but the key format
       needs __last in it. So actually pass method-tags as:
         qwen__tier0__last qwen__maximal__last random__tier0__last random__maximal__last
       And the coord script should handle 3-part tags for the key: method__cleaning__pooling
       where pooling is inserted before the reduction. Actually — the coord script just
       needs to use the tag as a prefix, then append __L{NN}__{reduction}.
       So if tag = "qwen__tier0__last", key = "qwen__tier0__last__L00__tsne" — wait,
       that doesn't match the schema! The schema is "qwen__tier0__L15__last__tsne"
       (pooling BEFORE reduction but AFTER layer).
       
       So the coord script must insert the pooling AFTER the layer token, not use it
       as a tag prefix. Best approach:
         Add --pooling [mean|last] flag to 04_compute_2d_coords.py
         Key = f"{method}__{cleaning}__L{NN:02d}{pooling_infix}__{reduction}"
         where pooling_infix = "__last" if pooling=="last" else ""
       
       Update Task B above to add --pooling flag to the coord script.
       This simplifies the sbatch calls.

  C) ORCC scripts (create directory first: mkdir -p v_1/src/linear_probing/sbatch/orcc):
     v_1/src/linear_probing/sbatch/orcc/extract_qwen_tier0.sh
       #SBATCH --job-name=orcc_qwen_tier0
       #SBATCH --time=04:00:00  (3x more text than SEAL)
       Calls 03_extract_seal_activations.py TWICE in sequence:
         python ... --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
                    --text-col text_tier0 --pooling mean \
                    --output-dir v_1/src/linear_probing/results/orcc_round1/activations/qwen_tier0_mean
         python ... --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \
                    --text-col text_tier0 --pooling last \
                    --output-dir v_1/src/linear_probing/results/orcc_round1/activations/qwen_tier0_last

     v_1/src/linear_probing/sbatch/orcc/extract_qwen_maximal.sh   (similar, text_maximal)
     v_1/src/linear_probing/sbatch/orcc/extract_random_tier0.sh   (similar, random model)
     v_1/src/linear_probing/sbatch/orcc/extract_random_maximal.sh (similar)

     v_1/src/linear_probing/sbatch/orcc/compute_2d_umap_coords.sh
       #SBATCH --gres=gpu:0, --mem=64G (more RAM for 1202 texts), --time=02:00:00
       Two calls (mean then last), or pass all 8 dirs at once with appropriate method-tags:
         python 04_compute_2d_coords.py \
           --include-umap --pooling mean \
           --input-dirs qwen_tier0_mean qwen_maximal_mean random_tier0_mean random_maximal_mean \
           --input-base v_1/src/linear_probing/results/orcc_round1/activations \
           --method-tags qwen__tier0 qwen__maximal random__tier0 random__maximal \
           --output-path v_1/src/linear_probing/results/orcc_round1/orcc_qwen_coords_mean.json

         python 04_compute_2d_coords.py \
           --include-umap --pooling last \
           --input-dirs qwen_tier0_last qwen_maximal_last random_tier0_last random_maximal_last \
           --input-base v_1/src/linear_probing/results/orcc_round1/activations \
           --method-tags qwen__tier0 qwen__maximal random__tier0 random__maximal \
           --output-path v_1/src/linear_probing/results/orcc_round1/orcc_qwen_coords_last.json

────────────────────────────────────────────────────────────
TASK E — Update merge script
  File: v_1/src/viz/02_merge_coords.py

  1. Read the file first.
  2. Generalize to also:
     a) Load ORCC corpus parquet and add ORCC fragment metadata (with year and ruler fields)
        to the fragments list — after the existing SEAL fragments
     b) Load these new coord JSONs if they exist:
          seal_round4/seal_qwen_coords_last.json
          seal_round4/seal_qwen_coords_umap.json    (optional)
          orcc_round1/orcc_qwen_coords_mean.json
          orcc_round1/orcc_qwen_coords_last.json
     c) Patch existing SEAL/DLL/LBPL fragments to add year=null, ruler=null fields
        (so all fragments have the same schema)
  3. Validation: allow variable fragment count (total = SEAL count + ORCC count)
     Each embedding key must have exactly N_total entries
  4. Print a clear summary showing counts per corpus

STOP when: all 5 tasks are done and all files are saved. No cluster jobs. No test runs.
```

---

### TERMINAL 3 — Cluster Orchestration (start AFTER Terminal 2 is complete)

```
You are working in /Users/yarin.b/git/lititure-review.
PREREQUISITE: Terminal 2 must be fully done before starting this.

GOAL: Build ORCC corpus, submit cluster jobs, wait, retrieve results.

────────────────────────────────────────────────────────────
STEP 1 — Build ORCC corpus locally (run from repo root, ~2 min)
  source venv/bin/activate  (or however the local venv works)
  python v_1/src/corpus/03_build_orcc_corpus.py
  # Verify: v_1/data/evaluation/corpora/orcc_corpus.parquet exists with ~1202 rows

────────────────────────────────────────────────────────────
STEP 2 — Inspect TF-IDF script to check if it can handle ORCC
  READ v_1/src/viz/01_compute_tfidf_coords.py
  Check if it hardcodes seal_corpus.parquet or accepts a --corpus flag.
  If hardcoded: run it once for ORCC by adapting manually, or note it as a TODO.
  We need TF-IDF coords for ORCC too (to have a baseline in the viz).
  If the script is easily adaptable, run it for ORCC and save orcc_tfidf_coords.json.
  If not, skip for now — ORCC will still appear in the viz with Qwen/Random embeddings.

────────────────────────────────────────────────────────────
STEP 3 — Push code and sync to cluster
  git add -A && git commit -m "Add ORCC corpus builder, updated extraction/coord scripts, new sbatch scripts"
  git push
  ssh [cluster]  (use your usual SSH method)
  cd ~/projects/HUJI-THESIS--YARIN
  git pull
  conda activate thesis
  # Check umap-learn is installed:
  python -c "import umap; print('umap ok')" || conda install -c conda-forge umap-learn -y
  mkdir -p v_1/src/linear_probing/results/orcc_round1/activations
  mkdir -p v_1/src/linear_probing/logs

────────────────────────────────────────────────────────────
STEP 4 — Check if existing SEAL mean .npz files still exist
  ls v_1/src/linear_probing/results/seal_round4/activations/qwen_tier0/ 2>/dev/null | head -3
  # If layer_00.npz present: we can compute UMAP for old SEAL mean data (submit C3 later)
  # If absent: skip C3

────────────────────────────────────────────────────────────
STEP 5 — Submit 8 GPU jobs in parallel
  for script in \
    v_1/src/linear_probing/sbatch/seal/extract_qwen_tier0_last.sh \
    v_1/src/linear_probing/sbatch/seal/extract_qwen_maximal_last.sh \
    v_1/src/linear_probing/sbatch/seal/extract_random_tier0_last.sh \
    v_1/src/linear_probing/sbatch/seal/extract_random_maximal_last.sh \
    v_1/src/linear_probing/sbatch/orcc/extract_qwen_tier0.sh \
    v_1/src/linear_probing/sbatch/orcc/extract_qwen_maximal.sh \
    v_1/src/linear_probing/sbatch/orcc/extract_random_tier0.sh \
    v_1/src/linear_probing/sbatch/orcc/extract_random_maximal.sh; do
    JOB=$(sbatch --parsable $script)
    echo "Submitted $script → Job $JOB"
  done
  squeue -u $USER

────────────────────────────────────────────────────────────
STEP 6 — Wait for all 8 GPU jobs to complete
  Watch: squeue -u $USER   (check periodically, ORCC jobs ~4 hrs, SEAL ~2 hrs)
  All jobs done when they disappear from squeue.

────────────────────────────────────────────────────────────
STEP 7 — Submit CPU coord jobs (all in parallel)
  sbatch v_1/src/linear_probing/sbatch/seal/compute_umap_coords_last.sh
  sbatch v_1/src/linear_probing/sbatch/orcc/compute_2d_umap_coords.sh
  # If STEP 4 showed existing .npz files:
  #   sbatch v_1/src/linear_probing/sbatch/seal/compute_umap_existing_mean.sh

────────────────────────────────────────────────────────────
STEP 8 — Wait for CPU jobs, then rsync results back
  # All CPU jobs done (~1-2 hrs):
  rsync -avz ~/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/ \
    [LOCAL_MACHINE]:/Users/yarin.b/git/lititure-review/v_1/src/linear_probing/results/

STOP when: rsync is complete and you've confirmed results exist locally.
Then notify the orchestration terminal.
```

---

## This Terminal — Orchestration Steps

```
OPEN ORDER:
  t=0:   Open Terminal 1 (HTML) — no deps
  t=0:   Open Terminal 2 (Scripts) — no deps
  WAIT:  Terminal 2 done → Open Terminal 3 (Cluster)
  WAIT:  Terminal 1 done AND Terminal 3 done → proceed below

FINAL STEPS (in this terminal):
  1. python v_1/src/viz/02_merge_coords.py
  2. python v_1/src/viz/03_build_standalone_html.py
  3. cd v_1/src/viz && python3 -m http.server 8000
  4. Open http://localhost:8000/seal_eda.html
  5. Smoke-test: all 5 features working, ORCC points visible, year coloring works
```

---

## Open Risks

| Risk | Mitigation |
|------|-----------|
| SEAL mean .npz files deleted from cluster | Skip UMAP for old SEAL mean in this iteration; UMAP available for all new data |
| ORCC GPU jobs fail (4 hrs not enough) | Bump `--time` to 06:00:00 in ORCC sbatch scripts |
| `umap-learn` not in cluster conda env | `conda install -c conda-forge umap-learn` before submitting CPU jobs |
| Merge script fragment count mismatch | Validate total = len(seal_corpus) + len(orcc_corpus) |
| HTML colorBy="year" has no data initially (before cluster jobs) | Graceful fallback: if all years null, show grey points with message |
