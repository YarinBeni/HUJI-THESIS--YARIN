# Linear Probing Implementation Plan
Last Updated: 2026-03-25

## Goal
Replicate Gurnee & Tegmark's linear probing methodology on Akkadian letters data.
Test whether open-source LLMs encode temporal period (OB/NA/LB) linearly in their activations.

## Letters vs. SEAL: Scope and Relationship

This plan targets the **letters corpus** (4,957 texts, 3 periods) as a controlled pipeline test. Letters are the "easy case" — the bias check confirmed that even TF-IDF character n-grams separate them at 84–99%. This is intentional: if linear probing cannot find temporal signal on easy data, it will not work on harder data either.

The **SEAL literary texts** (~400+ texts from Chungrong, pending delivery) are the thesis-facing evaluation. Literary texts are copied across generations and may lack clear period signal — making them the genuinely interesting challenge. When SEAL data arrives, new TF-IDF baselines must be computed for it before running probes, so that the same TF-IDF vs. LLM comparison is available. The letters pipeline developed here will transfer directly to SEAL with minimal changes (swap data path, recompute baselines).

## TF-IDF Baselines (Letters)
From the bias check (TF-IDF + logistic regression, 5-fold CV on 4,957 letters):
- Unigrams: 84.8% (69.1% after maximal cleaning)
- Bigrams: 98.3% (91.2% after cleaning)
- 2-5 grams: 99.2% (96.7% after cleaning)

## Success Criteria

Success is NOT a single accuracy threshold. It is evaluated on three axes:

**(a) Minimum bar:** Probe accuracy at the best layer must beat the **maximally-cleaned unigram baseline (69.1%)** on matched evaluation splits. Below this, the LLM encodes less temporal information than single-character frequencies after aggressive denoising.

**(b) Strong evidence:** Probe accuracy competing with or exceeding the cleaned bigram baseline (91.2%) on matched splits, with the peak occurring at **mid-to-late layers** (not layers 0-3). This would suggest the LLM encodes temporal information beyond surface token statistics.

**(c) Statistical significance:** The best-layer result must pass a permutation test (1,000 label shuffles, p < 0.01) AND show a clear layer-peak shape (not flat across all layers). A flat curve at high accuracy would suggest the signal is trivially available at every layer (i.e., surface features propagated unchanged).

## Models
Start with one, expand later:
1. **Llama-3.1-8B-Instruct** — primary (has pre-trained SAE for Track C, closest to Gurnee & Tegmark's LLaMA-2)
2. Gemma-2-9B-IT — secondary (has SAE)
3. Qwen2.5-7B-Instruct — tertiary (has SAE, already verified on cluster)

## Data
- Source: `v_1/data/evaluation/corpora/texts_for_evaluation.parquet`
- 4,957 letters, 3 classes: OB (1,497), NA (2,435), LB (1,025)
- Later: SEAL literary texts (~400+ from Chungrong, when available)

## Evaluation Protocol

Use the **same 70/15/15 stratified split** as the bias check (seed=42) to enable direct comparison:
- Train (3,470): used for fitting the probe
- Validation (743): used for layer selection and hyperparameter tuning (regularization strength)
- Test (744): used **once** for final reporting after the best layer and hyperparams are locked

For the layer-accuracy curve (exploratory), use 5-fold CV on the train+val set (4,213 texts). The test set is held out and only touched for the final reported number.

Regularization grid for logistic regression: `C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}`, selected via validation set accuracy.

**Metrics:** accuracy, F1 macro, per-class precision/recall/F1, confusion matrix at best layer.

## Cleaning Ablation (Confound Control)

Instead of statistical length confounds, reuse the **existing 11 cleaning filters** from the bias check:

1. Run the full pipeline (Steps 0–2) on **tier0-cleaned text** (raw baseline)
2. Run the full pipeline again on **maximally-cleaned text** (all 11 filters stacked)

This tests ALL confounds at once — length, logograms, w/y phoneme, subscript digits, case endings, determinatives, etc.

**What to look for:**
- If probe accuracy drops after cleaning → the probe was partly relying on surface features
- If the best layer **shifts** between raw and cleaned (e.g., raw peaks at layer 4, cleaned peaks at layer 18) → different layers encode surface vs. deep features — a very interesting finding
- If probe accuracy holds after cleaning → the LLM captures something beyond TF-IDF features
- Optionally: greedy ablation at the best layer (stack filters one by one), directly comparable to the bias check ablation curves

## Random-Label Baseline (Confound Control)

For the best layer, shuffle labels 1,000 times, run the probe each time, record distribution of null accuracies. Verify real accuracy is statistically separated (p < 0.01). This proves the probe learns real signal, not noise from high-dimensional activations (4,096 dims vs. ~4,000 samples).

## Reproducibility

- Random seed: 42 (splits, sklearn, t-SNE, torch)
- Fold assignment: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Torch: `torch.manual_seed(42)`, `torch.use_deterministic_algorithms(True)` where possible
- Regularization grid: `C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}`
- Python/library versions: use cluster `thesis` conda env (Python 3.11, PyTorch 2.10, Transformers 5.3, sklearn)

## Infrastructure

**Everything runs on the Schmidt Sciences HPC cluster** via Slurm sbatch, NOT locally.
- Cluster repo path: `~/projects/HUJI-THESIS--YARIN`
- Conda env: `thesis` at `~/miniconda3/envs/thesis/`
- Partition: `voltagepark`
- GPUs: H100 80GB (1 GPU is enough for 8B models)
- Activations storage: cluster NFS (`~/projects/HUJI-THESIS--YARIN/v_1/src/linear_probing/results/`)
- Workflow: edit locally → `git push` → `git pull` on cluster → `sbatch`

**Storage estimate for activations:**
- Llama-3.1-8B: 32 layers × 4,957 texts × 4,096 dims × 4 bytes = ~2.5 GB per run
- With 2 runs (raw + cleaned) = ~5 GB
- Cluster has 500 TB NFS — no issue

---

## Step 0 — Tokenization Sanity Check
**Script:** `00_tokenization_check.py`
**Sbatch:** `sbatch/00_tokenization.sh` (1 GPU, 30 min, ~16GB VRAM)

Pick 10 texts (mix of OB/NA/LB, short/long). For each model's tokenizer:
- Print raw text → tokenized tokens → decoded tokens
- Count: tokens per text, unknown/fallback tokens, average token length
- Check: do different periods produce systematically different token counts or patterns?

**Output:** `results/tokenization_check.json`

**What we're looking for:**
- If Akkadian is tokenized into single-character or byte-level pieces → the model treats it as raw bytes
- If some Akkadian words survive as recognizable subwords → the model has partial knowledge
- If token counts differ wildly by period → flag for later analysis

**Decision gate:** Not a blocker. Proceed regardless, but calibrate expectations.

**Verify:** Open JSON → check token counts × avg token length ≈ original text char count. Decoded tokens should reconstruct the original text.

---

## Step 0.5 — Quick EDA on Final-Layer Embeddings
**Script:** `00b_quick_eda.py`
**Sbatch:** `sbatch/00b_quick_eda.sh` (1 GPU, 30 min)

Before committing to full 32-layer extraction:
1. Extract **final-layer** mean-pooled embeddings only (one forward pass, save only last hidden state)
2. Run UMAP and PCA, color by period
3. Check: is there any obvious clustering?

**Output:** `results/plots/quick_eda_final_layer.png`

**Verify:** Plot has 4,957 dots. Colors match class counts (1,497 blue OB, 2,435 purple NA, 1,025 red LB). Even no clustering is a valid result — mid-layers may differ.

---

## Step 1 — Extract Activations
**Script:** `01_extract_activations.py`
**Sbatch:** `sbatch/01_extract.sh` (1 GPU, 2-4 hours, 64GB RAM)

For each text in the 4,957 letters:
1. Apply tier0 cleaning (strip `@v` markup, encoding artifacts)
2. Tokenize with the model's tokenizer
3. Forward pass with `output_hidden_states=True` (HuggingFace returns all layers in one call)
4. Mean-pool across token positions at each layer → one vector (D,) per layer per text
5. Save per-text token counts in metadata
6. Save as `.npz`, one per layer

**Run twice:**
- `01_extract_activations.py --cleaning tier0` (raw baseline)
- `01_extract_activations.py --cleaning maximal` (all 11 filters stacked)

**Output:**
```
results/activations/llama-3.1-8b-instruct/
├── tier0/
│   ├── layer_00.npz ... layer_31.npz
│   └── metadata.json
└── maximal/
    ├── layer_00.npz ... layer_31.npz
    └── metadata.json
```

**Verify:**
```python
import numpy as np
X = np.load('layer_16.npz')['activations']
assert X.shape == (4957, 4096)
assert not np.any(np.isnan(X))
assert np.std(X) > 0.01  # not all zeros
assert not np.allclose(X[0], X[1])  # different texts differ
```
- All 32 files ~78 MB each
- Layer 0 activations should have different distribution than layer 31

---

## Step 2 — Linear Probe at Every Layer
**Script:** `02_linear_probe.py`
**Sbatch:** `sbatch/02_probe.sh` (CPU only, no GPU, 1 hour, 32GB RAM)

### 2a. Layer-accuracy curve (5-fold CV on train+val)

For each layer L, for each cleaning condition (tier0, maximal):
1. Load activations (train+val subset, ~4,213 texts)
2. `LogisticRegression(penalty='l2', C=best_C, max_iter=1000)`, 5-fold stratified CV
3. Record: accuracy, F1 macro, per-class precision/recall/F1

### 2b. Random-label baseline (at best layer only)

Shuffle labels 1,000 times, run probe, record null distribution.

### 2c. Final evaluation (held-out test set)

Lock best layer + best C from 2a, retrain on full train+val, evaluate on test (744 texts). Run permutation test (1,000 shuffles, p < 0.01).

**Output:**
- `results/probe_results_{model_name}.json` — all metrics
- `results/plots/layer_accuracy_curve.png` — curve with CI bands + TF-IDF baselines as horizontal lines
- `results/plots/confound_random_label.png` — null distribution
- `results/plots/tsne_by_layer.png` — t-SNE at early/best/late layers, colored by period
- `results/plots/confusion_matrix_best_layer.png`

**Verify:**
- Random-label baseline → ~33-40% accuracy (if 60%+, bug: labels not shuffled)
- Curve should be smooth across adjacent layers (no wild jumps)
- Test accuracy within ~3-5% of CV accuracy (if gap > 10%, overfitting)
- Confusion matrix: probe should NOT just predict NA for everything (majority = 49%)

---

## Step 3 — Interpret and Decide
**Script:** `03_analyze_results.py`
**Sbatch:** `sbatch/03_analyze.sh` (CPU only, 30 min)

Compare tier0 vs. maximal cleaning results. Produce summary.

### Outcome A: Probe beats cleaned baselines AND peaks at mid-layers AND passes controls
Suggestive of temporal representation beyond surface statistics, pending SEAL replication.
- Extract probe weight vector at best layer → candidate "time direction"
- Project activations onto this direction → 1D temporal score per text
- t-SNE: raw activations vs. projected → does projection improve separation?
- Move to Track C (SAE decomposition)
- Replicate on SEAL when available

### Outcome B: Probe works but peaks at early layers OR best-layer shifts to early after cleaning
Consistent with surface-only encoding.
- Compare raw vs. cleaned layer curves — which features mattered?
- Publishable: "LLMs encode surface-level temporal signal for OOD languages"
- Proceed to fine-tuning fallback

### Outcome C: Probe fails to beat cleaned unigram floor (69.1%)
Boundary finding: Gurnee & Tegmark's result doesn't extend to OOD languages.
- Fine-tuning mandatory
- Write up as negative result (meaningful contribution)

---

## File Structure
```
v_1/src/linear_probing/
├── plan/
│   └── PLAN.md                      (this file)
├── 00_tokenization_check.py
├── 00b_quick_eda.py
├── 01_extract_activations.py
├── 02_linear_probe.py
├── 03_analyze_results.py
├── utils.py                         (data loading, tier0 + maximal cleaning, splits, constants)
├── sbatch/
│   ├── 00_tokenization.sh
│   ├── 00b_quick_eda.sh
│   ├── 01_extract.sh
│   ├── 02_probe.sh
│   └── 03_analyze.sh
└── results/
    ├── tokenization_check.json
    ├── activations/
    │   └── llama-3.1-8b-instruct/
    │       ├── tier0/
    │       │   ├── layer_00.npz ... layer_31.npz
    │       │   └── metadata.json
    │       └── maximal/
    │           ├── layer_00.npz ... layer_31.npz
    │           └── metadata.json
    ├── probe_results_llama-3.1-8b-instruct.json
    └── plots/
        ├── quick_eda_final_layer.png
        ├── layer_accuracy_curve.png
        ├── confound_random_label.png
        ├── tsne_by_layer.png
        ├── confusion_matrix_best_layer.png
        └── cleaning_ablation_comparison.png
```

## Execution Order
All steps run on the Schmidt cluster via sbatch:
1. `sbatch sbatch/00_tokenization.sh` → verify tokenization
2. `sbatch sbatch/00b_quick_eda.sh` → quick EDA on final layer
3. `sbatch sbatch/01_extract.sh` → extract all-layer activations (tier0 + maximal)
4. `sbatch sbatch/02_probe.sh` → probe + confound controls + final eval
5. `sbatch sbatch/03_analyze.sh` → interpret, produce summary + time direction
