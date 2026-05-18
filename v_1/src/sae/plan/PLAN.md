# Track C — SAE Analysis: Implementation Plan

> **See also:** [../../linear_probing/results/PIPELINE_RUN_LOG.md](../../linear_probing/results/PIPELINE_RUN_LOG.md) (linear probe results this builds on) · [../../../../PLAN_round2_qwen_diagnosis.md](../../../../PLAN_round2_qwen_diagnosis.md) Phase 2 (Qwen-3 SAE follow-up)

> **Status:** Planning phase (April 2026)
> **Can start now:** Steps 1–2 on existing letters corpus (4,957 texts)
> **Blocked for scaling:** Full dataset delivery from advisor (~40k fragments)

---

## Research Goal

Track B (linear probing) found that Qwen2.5-7B-Instruct linearly encodes
Akkadian historical period (OB/NA/LB) in its activations. Track C asks:
**what interpretable features compose that period representation?**

Sparse Autoencoders decompose dense activations into 131k sparse, interpretable
features. By running sparse probing on SAE features, we identify the specific
features that carry period information — bridging Track B's "there is a direction"
with Track C's "here is what that direction is made of."

### What This Produces (letters corpus)

1. **Sparse probe accuracy curve** — How many SAE features (out of 131k) are needed
   to classify period? If k=50 features reach 90%+, the period signal is concentrated
   in a tiny interpretable subset.

2. **Probe direction decomposition** — Track B found weight vector w (the "temporal
   direction"). We compute which SAE decoder columns align with w. This decomposes
   the temporal direction into named features.

3. **Per-period feature profiles** — Which features fire for OB but not NA/LB?

4. **Cross-layer comparison** — Do layers 7, 15, 23 use the same features for period,
   or does the model build period representation from different features at different
   depths?

5. **Automated feature interpretation** — For each top feature, differential bigram
   analysis between high-activation and low-activation texts reveals what linguistic
   patterns the feature responds to, and at what text positions. Connects directly
   to the bias check's finding that bigrams carry the period signal.

### Future Work (after advisor review of initial results)

- **Per-token SAE extraction (Option B):** Extract per-token activations at the most
  interesting SAE layer(s) only (not all layers). Run SAE per-token and identify
  *which specific tokens* activate each feature. This would upgrade Analysis E from
  "this feature correlates with texts containing bigram X" to "this feature fires on
  token Y at position Z." Requires GPU but only for 1-2 layers, not all 29.
  Decision point: run after reviewing initial results with advisors.

- **40k dataset:** Genre-controlled analysis (letters vs administrative vs literary),
  large-scale EDA across text types, more statistical power for feature interpretation.

---

## Pre-trained SAE

Arditi (2024) SAE for Qwen2.5-7B-Instruct:
- **131,072 features** (131k)
- **Available at layers 7, 15, 23 only**
- Same SAE used in Feldman et al. 2026
- We do NOT train an SAE. We load the pre-trained one.

### Why last_token pooling only

The SAE was trained on **per-token** residual stream activations. Last-token pooling
extracts the activation at one real token position — exactly what the SAE expects.
Mean pooling averages across all positions, producing a synthetic vector the SAE
never saw during training. Since we didn't train this SAE, we should feed it
the input distribution it was designed for.

Our best last_token probing results (L28: 95.5% tier0, 90.0% maximal) show strong
period signal in last-token activations. The SAE layers (7/15/23) won't match L28's
accuracy, but they should still contain period-relevant information.

---

## Folder Structure

```
v_1/src/sae/
├── plan/
│   └── PLAN.md                     # This file
├── utils.py                        # Constants, SAE loading, data I/O
├── 01_extract_sae_features.py      # Run SAE encoder on existing activations
├── 02_analyze.py                   # Sparse probe + feature analysis (merged)
├── sbatch/
│   ├── 01_extract.sh               # CPU job: SAE feature extraction
│   └── 02_analyze.sh               # CPU job: sparse probe + analysis + plots
└── results/
    ├── sae_features/               # Sparse feature activations per layer
    │   └── qwen2.5-7b-instruct/
    │       ├── tier0_last_token/
    │       │   ├── layer_07.npz
    │       │   ├── layer_15.npz
    │       │   └── layer_23.npz
    │       └── maximal_last_token/
    │           └── ...
    ├── analysis/                   # Probe results + feature rankings (JSON)
    └── plots/
```

---

## Step 0: Verify SAE Loading (First Action)

Run this on the cluster before writing any pipeline code:

```bash
#!/bin/bash
#SBATCH --job-name=sae_test
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=sae_test_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
~/miniconda3/envs/thesis/bin/pip install sae-lens

python -u -c "
import sae_lens
from sae_lens import SAE
print('SAELens version:', sae_lens.__version__)

# List available Qwen releases
releases = sae_lens.pretrained_saes.get_pretrained_saes_directory()
qwen_releases = [k for k in releases.keys() if 'qwen' in k.lower()]
print('Qwen releases:', qwen_releases)

# Try loading layer 7
sae, cfg, sparsity = SAE.from_pretrained(
    release='qwen2.5-7b-instruct-131k',   # adjust based on qwen_releases output
    sae_id='blocks.7.hook_resid_post',     # adjust if hook name differs
)
print('Loaded! W_enc shape:', sae.W_enc.shape)
print('W_dec shape:', sae.W_dec.shape)
"
```

If the release name is wrong, use the `qwen_releases` output to find the correct one.
If the SAE isn't in SAELens at all, load encoder/decoder weights directly from
HuggingFace (search for Arditi's repos).

**New cluster dependency:** `pip install sae-lens`

---

## Step 1: `01_extract_sae_features.py`

**Input:** Existing last_token `.npz` activations from `linear_probing/results/activations/`
**Output:** SAE sparse feature activations per layer
**GPU needed:** No (matrix multiplication on existing data)

### Algorithm

```
For each layer in [7, 15, 23]:
    Load activations: layer_{LL}.npz → shape (n_texts, 3584)
    Load SAE encoder for this layer
    z = ReLU(W_enc @ a + b_enc)     → shape (n_texts, 131072)  # sparse
    Save z as compressed .npz
    Save metadata.json (n_texts, n_features, mean_active_features, ...)
```

### Storage

Dense float16 `.npz` with numpy compression.
4957 × 131072 × 2 bytes ≈ 1.3 GB uncompressed per layer.
With sparsity + compression: ~50-150 MB per layer in practice.

### CLI

```bash
python -u v_1/src/sae/01_extract_sae_features.py \
    --model qwen2.5-7b-instruct \
    --cleaning tier0 \
    --pooling last_token
```

Args: `--model`, `--cleaning` (tier0|maximal), `--pooling` (last_token only for now),
`--layers` (default: 7,15,23)

**Runtime:** ~5-15 min for all 3 layers (CPU only)

---

## Step 2: `02_analyze.py`

**Input:** SAE features from step 1 + period labels
**Output:** Sparse probe results, feature rankings, comparison with linear probe, plots
**GPU needed:** No

### Analysis A: Sparse Probing

Following Feldman et al. §5.1:

1. Load SAE features for a given layer (n_texts × 131072)
2. **Filter:** Remove features active on <1% or >99% of texts (dead/universal)
3. **Z-score normalize** each surviving feature
4. **Mean-difference ranking (one-vs-rest):**
   For each period P and feature φ:
   `Δ_φ(P) = |E[Z_φ | period=P] - E[Z_φ | period≠P]|`
5. **Top-k selection:** Take top 128 features by max Δ across periods
6. **L1 logistic regression** on top-128 features, 5-fold CV for regularization
7. Report: accuracy, per-period F1, top features with coefficients

### Analysis B: Probe Direction Decomposition

The linear probe at layer L has weight vector w ∈ R^3584 (the "temporal direction").
The SAE has decoder columns W_dec[φ] ∈ R^3584 for each feature φ.

```
alignment_φ = cosine_similarity(W_dec[φ], w_probe)
```

Features with high |alignment| are what the temporal direction is "made of" in SAE
feature space. This is the key Track B → C bridge.

**Note:** The linear probe weights aren't saved in Track B's results JSON. This script
trains a quick logistic regression at the relevant layer (takes seconds from existing
activations) and extracts `clf.coef_` on the spot. No Track B modification needed.

### Analysis C: Per-Period Feature Profiles

For top-50 features from Analysis A:
- Violin/boxplot of activation values split by OB/NA/LB
- Identifies "OB-specific", "NA-specific", "LB-specific" features

### Analysis D: Cross-Layer Comparison

Compare top features across layers 7, 15, 23:
- Do the same features appear at multiple layers?
- Does the model use different features for period at different depths?
- Accuracy comparison: sparse probe accuracy at each SAE layer

### Analysis E: Automated Feature Interpretation (Bigram Differential)

For each top-k feature, computationally identify what linguistic patterns it responds
to by comparing bigram distributions between high-activation and low-activation texts.

**Algorithm:**
```
For each top feature φ (e.g. top 20 features):
    high_texts = top 100 texts by feature φ activation
    low_texts  = bottom 100 texts by feature φ activation

    # 1. Differential bigram frequencies
    high_bigram_freq = normalized bigram counts across high_texts
    low_bigram_freq  = normalized bigram counts across low_texts
    differential     = high_bigram_freq - low_bigram_freq
    → rank bigrams by |differential|
    → top-20 enriched bigrams = "this feature responds to these patterns"

    # 2. Bigram position profiles
    For each of the top-20 differential bigrams:
        Record position (token index in text) where bigram appears
        Build position distribution for high_texts vs low_texts
    → Shows whether feature is position-sensitive (e.g. greeting formulas
      always at positions 0-5 vs content patterns appearing mid-text)
```

**Why bigrams:** The bias check showed bigrams carry far more period signal than
unigrams (98.3% vs 84.8% on tier0). Bigram-level analysis is the right granularity.

**Why top/bottom 100 (not 10):** Bigram distributions need mass. With 4,957 texts,
top/bottom 100 gives enough data for meaningful frequency counts.

**Output per feature (4 plots):**

1. **Differential bigram bar chart** — horizontal bars, top-20 bigrams. Bars right =
   enriched in high-fire texts, bars left = enriched in low-fire texts. One plot per
   feature, immediately shows what the feature "looks for."

2. **Bigram position heatmap (high-fire)** — x-axis = token position (0..30),
   y-axis = top-10 differential bigrams, color = frequency. Shows where in the text
   these bigrams appear for high-activation texts.

3. **Bigram position heatmap (low-fire)** — same layout for low-activation texts.
   Comparing plots 2 and 3 reveals position-sensitivity.

4. **Summary table** (CSV/markdown) — for each feature: feature index, period
   association, top-5 enriched bigrams, top-5 depleted bigrams, dominant position range.
   This is the human-readable output for the thesis.

**Connection to bias check:** This directly reuses the bigram analysis from
`v_1/src/bias_check/`. The bias check found that certain bigram patterns distinguish
periods. Now we ask: do the SAE features align with those same bigram patterns?

### Plots (full list)

1. Accuracy bar: SAE sparse probe (k=128) vs full linear probe at same layer
2. Top-20 features per period (bar chart of mean-difference or coefficient)
3. Probe direction decomposition: top-20 SAE features by alignment with w_probe
4. Per-period violin plots for top features
5. Cross-layer feature overlap (heatmap or Venn-style)
6. Per-feature differential bigram bar charts (Analysis E, top 20 features)
7. Per-feature bigram position heatmaps (Analysis E, top 20 features)

### Output JSON

```json
{
  "layer": 7,
  "cleaning": "tier0",
  "pooling": "last_token",
  "n_features_total": 131072,
  "n_features_after_filter": 45231,
  "sparse_probe": {
    "k": 128,
    "accuracy": 0.89,
    "f1_macro": 0.87,
    "best_C": 0.1
  },
  "linear_probe_accuracy_same_layer": 0.91,
  "top_features": [
    {"feature_idx": 53435, "period": "OB", "mean_diff": 0.84,
     "coeff": 1.23, "probe_alignment": 0.45},
    ...
  ]
}
```

### CLI

```bash
python -u v_1/src/sae/02_analyze.py \
    --model qwen2.5-7b-instruct \
    --cleaning tier0 \
    --pooling last_token \
    --layer 7 \
    --top-k 128
```

**Runtime:** ~20-40 min per layer (CPU, sklearn)

---

## Step 3: Feature Interpretation (Human + Automated)

After running `02_analyze.py`, interpretation combines automated bigram analysis
(Analysis E) with manual reading:

1. **Start with Analysis E outputs:** The differential bigram charts and position
   heatmaps show, for each top feature, which bigrams are enriched/depleted and
   where they appear. This gives an immediate computational fingerprint per feature.

2. **Read the summary table:** `results/analysis/feature_interpretation.csv` lists
   each feature's top enriched/depleted bigrams and dominant positions. Scan this
   to form hypotheses (e.g., "feature #X enriches OB greeting bigrams at positions 0-5").

3. **Spot-check with actual texts:** For the most interesting features, read a few
   high-activation texts to confirm the bigram analysis and catch patterns that
   bigrams miss (e.g., syntactic structures, semantic content).

4. **Neuronpedia check (if available):** Look up feature indices at
   https://neuronpedia.org/. May not have Qwen SAE descriptions — verify availability.

5. **Linguistic interpretation:** With your Assyriology knowledge, map identified
   patterns to known diachronic changes in Akkadian (phonological shifts, grammatical
   markers, formulaic expressions, vocabulary changes between OB/NA/LB).

The pipeline is infrastructure. The interpretation is the thesis content.

---

## sbatch Scripts

### `sbatch/01_extract.sh`

```bash
#!/bin/bash
#SBATCH --job-name=sae_extract
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=v_1/src/sae/logs/extract_%j.out

echo "=== SAE Feature Extraction ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

mkdir -p v_1/src/sae/logs

echo "=== tier0, last_token ==="
python -u v_1/src/sae/01_extract_sae_features.py \
    --model qwen2.5-7b-instruct \
    --cleaning tier0 \
    --pooling last_token \
    || { echo "FAILED: tier0 last_token"; exit 1; }

echo "=== maximal, last_token ==="
python -u v_1/src/sae/01_extract_sae_features.py \
    --model qwen2.5-7b-instruct \
    --cleaning maximal \
    --pooling last_token \
    || { echo "FAILED: maximal last_token"; exit 1; }

echo "=== Done ==="
echo "End: $(date)"
```

### `sbatch/02_analyze.sh`

```bash
#!/bin/bash
#SBATCH --job-name=sae_analyze
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=v_1/src/sae/logs/analyze_%j.out

echo "=== SAE Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK

mkdir -p v_1/src/sae/logs

for LAYER in 7 15 23; do
  for CLEANING in tier0 maximal; do
    echo "=== Layer $LAYER, $CLEANING ==="
    python -u v_1/src/sae/02_analyze.py \
        --model qwen2.5-7b-instruct \
        --cleaning $CLEANING \
        --pooling last_token \
        --layer $LAYER \
        --top-k 128 \
        || { echo "FAILED: layer $LAYER $CLEANING"; exit 1; }
  done
done

echo "=== Done ==="
echo "End: $(date)"
```

---

## Execution Order

```
Step 0: Verify SAE loads on cluster (sae_test job)
        Install sae-lens if not present
        ↓
Step 1: Write utils.py + 01_extract_sae_features.py
        sbatch sbatch/01_extract.sh
        Input:  linear_probing/results/activations/ (already exists)
        Output: sae/results/sae_features/
        Time:   ~15 min (CPU)
        ↓
Step 2: Write 02_analyze.py
        sbatch --dependency=afterok:$STEP1_JOB sbatch/02_analyze.sh
        Input:  SAE features + period labels + linear probing activations
        Output: sae/results/analysis/ + sae/results/plots/
        Time:   ~2-3 hrs (CPU, 6 layer/cleaning combos)
        ↓
Step 3: Manual interpretation (human work, no code)
        Inspect top features, Neuronpedia lookup, linguistic analysis
        ↓
Future: Re-run on 40k dataset when received from advisor
        Genre-controlled analysis, large-scale EDA
```

---

## `utils.py` Design

```python
# Constants
SAE_LAYERS = [7, 15, 23]
SAE_N_FEATURES = 131_072
SAE_RELEASE = "qwen2.5-7b-instruct-131k"  # verify with Step 0

# Paths — same pattern as linear_probing
_THIS_DIR = Path(__file__).resolve().parent
DATA_PATH = _THIS_DIR / '../../data/evaluation/corpora/texts_for_evaluation.parquet'
RESULTS_DIR = _THIS_DIR / 'results'
LP_RESULTS_DIR = _THIS_DIR / '../linear_probing/results'  # to read existing activations

# Functions
load_sae(layer)                          # load pre-trained SAE for a layer
sae_features_dir(model_name, cleaning)   # path to saved features
load_sae_features(model_name, cleaning, layer)  # load feature matrix
```

Import `get_splits`, `PERIODS`, `PERIOD_COLORS`, `SEED`, `PERIOD_MAP` from
`linear_probing.utils` to guarantee identical splits and label conventions.

When the 40k dataset arrives, update `DATA_PATH` here and in `linear_probing/utils.py`
simultaneously. Env var mechanism can be added at that point across all pipelines.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| SAELens release name wrong | Step 0 prints available releases; adjust name |
| SAE not in SAELens at all | Load weights directly from HuggingFace |
| Sparse probe accuracy very low (<60%) | SAE may not decompose period signal well at these layers; report as finding |
| 131k × 5k matrix too large for memory | float16 + compression; process layers one at a time |
| Neuronpedia has no Qwen feature descriptions | Rely on manual text inspection for interpretation |
| Top features are uninterpretable | Still a valid thesis finding — "period signal is distributed, not concentrated" |

---

## What NOT to Do

- **Do NOT train an SAE** — use the pre-trained Arditi one
- **Do NOT use mean pooling** — SAE expects per-token activations; last_token is correct
- **Do NOT implement steering** — deferred, more complex, unclear value for classification task
- **Do NOT refactor dataset paths now** — defer until 40k dataset arrives
- **Do NOT implement Feldman's residualization / CATE** — doesn't apply to our classification task

---

## Status Tracker

- [ ] Step 0: Verify SAE loads on cluster
- [ ] Write `utils.py`
- [ ] Write `01_extract_sae_features.py`
- [ ] Write sbatch/01_extract.sh
- [ ] Run Step 1: extract SAE features (letters corpus)
- [ ] Write `02_analyze.py`
- [ ] Write sbatch/02_analyze.sh
- [ ] Run Step 2: sparse probe + analysis
- [ ] Step 3: Feature interpretation (automated bigram analysis + manual reading)
- [ ] **Review results with advisors** — decision point before proceeding
- [ ] (Future) Per-token SAE extraction at best layer(s) for token-level interpretability
- [ ] (Future) Re-run on 40k dataset
