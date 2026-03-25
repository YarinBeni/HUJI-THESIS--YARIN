# Implementation Prompts for Linear Probing Pipeline

Each section below is a self-contained prompt for implementing one step of the pipeline.
Before implementing any step, read:
1. `v_1/src/linear_probing/plan/PLAN.md` — the full plan with success criteria and evaluation protocol
2. `v_1/src/cluster/README.md` — cluster setup, sbatch patterns, conda env
3. `v_1/src/cluster/test_model_load.py` — working example of HuggingFace model loading + hidden state extraction on the cluster
4. `v_1/src/bias_check/run_bias_check.sh` — working sbatch template with fail-fast pattern

---

## Shared: `utils.py`

**Implement first** — all scripts import from this.

**What to build:**
```python
# Constants
SEED = 42
DATA_PATH = Path('../../data/evaluation/corpora/texts_for_evaluation.parquet')
RESULTS_DIR = Path(__file__).parent / 'results'
PERIODS = ['OB', 'NA', 'LB']  # in this order for label encoding
PERIOD_COLORS = {'OB': '#1976D2', 'NA': '#7B1FA2', 'LB': '#E53935'}

# Data loading
def load_letters() -> pd.DataFrame:
    """Load the 4,957 letters with columns: text, period, fragment_id, etc."""

# Cleaning
def clean_tier0(text: str) -> str:
    """Strip @v markup, non-breaking space, subscript-x. Same as bias check."""

def clean_maximal(text: str) -> str:
    """Apply all 11 filters from the bias check greedily stacked."""

# Splits — MUST match bias check for direct comparison
def get_splits(df, seed=SEED):
    """70/15/15 stratified split. Returns train_idx, val_idx, test_idx."""
    # Use sklearn StratifiedShuffleSplit with the SAME seed as bias check
```

**Reference files for cleaning functions:**
- `v_1/src/bias_check/bias_analysis_finetune.ipynb` cell 16 — all 11 filter definitions
- `v_1/src/bias_check/bias_analysis.ipynb` — tier0 cleaning

**How to verify:** Load data, check `len(df) == 4957`, check period value counts match (OB=1497, NA=2435, LB=1025). Apply tier0 cleaning to a sample text containing `@v` — should be removed. Apply maximal cleaning — should produce lowercase syllabic-only text.

---

## Step 0: `00_tokenization_check.py`

**What it does:** Load tokenizer(s) for the target model(s), tokenize 10 sample Akkadian texts, print and save analysis.

**Input:** 10 texts from `texts_for_evaluation.parquet` — pick 3 OB, 4 NA, 3 LB, mix of short and long.

**Implementation:**
1. Load tokenizer via `AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")`
2. For each text:
   - Apply tier0 cleaning
   - Tokenize: `tokens = tokenizer(text, return_tensors='pt')`
   - Get token strings: `token_strs = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])`
   - Decode back: `decoded = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)`
   - Count total tokens, count byte-fallback tokens (tokens starting with `<0x` or `Ġ` or single bytes)
3. Compute summary: mean tokens per text by period (use ALL 4,957 texts for this), token count distributions
4. Save to `results/tokenization_check.json`

**CLI:** `python 00_tokenization_check.py --model meta-llama/Llama-3.1-8B-Instruct`

**Sbatch** (`sbatch/00_tokenization.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=tok_check
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tok_check_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
python v_1/src/linear_probing/00_tokenization_check.py \
    --model meta-llama/Llama-3.1-8B-Instruct
```

**Verify:** Open `tokenization_check.json`. Each text should have `token_count > 0`. Decoded text should approximately match the original (minor whitespace differences OK). Token counts across all 4,957 texts should have a reasonable distribution (median probably 50-200 tokens).

---

## Step 0.5: `00b_quick_eda.py`

**What it does:** One forward pass through the model, extract final-layer mean-pooled embeddings for all 4,957 texts, plot UMAP + PCA colored by period.

**Implementation:**
1. Load model with `output_hidden_states=True`, `torch_dtype=torch.bfloat16`, `device_map="auto"`
2. Process texts in batches (batch_size=8, pad to max length in batch):
   - Tokenize with `tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)`
   - Forward pass: `outputs = model(**inputs, output_hidden_states=True)`
   - Take `outputs.hidden_states[-1]` (last layer)
   - Mean-pool: for each text in batch, average over non-padding token positions using `attention_mask`
   - Move to CPU, convert to float32, collect
3. Stack into array (4957, hidden_dim)
4. Run PCA (n_components=2) and UMAP (n_components=2, n_neighbors=15, min_dist=0.1)
5. Plot 2 panels side by side, colored by period, with legend showing class counts
6. Save plot

**CLI:** `python 00b_quick_eda.py --model meta-llama/Llama-3.1-8B-Instruct`

**Sbatch** (`sbatch/00b_quick_eda.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=quick_eda
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/quick_eda_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
python v_1/src/linear_probing/00b_quick_eda.py \
    --model meta-llama/Llama-3.1-8B-Instruct
```

**Verify:** Plot has exactly 4,957 dots. No NaN warnings. Embedding array shape is (4957, 4096) for Llama-3.1-8B. Even a single blob is a valid result — it means the final layer doesn't separate periods visually, but probing at other layers may still work.

**Note:** Install umap-learn if not in the thesis env: `pip install umap-learn`. If umap fails to install, skip UMAP and do PCA + t-SNE instead.

---

## Step 1: `01_extract_activations.py`

**What it does:** Extract mean-pooled activations at ALL layers for all 4,957 texts. Save one `.npz` per layer.

**Implementation:**
1. Parse CLI args: `--model`, `--cleaning {tier0,maximal}`, `--batch-size 8`, `--max-length 512`
2. Load model with `output_hidden_states=True`, `torch_dtype=torch.bfloat16`, `device_map="auto"`
3. Load data, apply selected cleaning function
4. Process in batches:
   - Tokenize with padding + truncation
   - Forward pass → `outputs.hidden_states` is a tuple of (n_layers+1,) tensors, each (batch, seq_len, hidden_dim)
   - For each layer: mean-pool over non-padding positions using `attention_mask`
   - Append to per-layer lists
5. Stack each layer's results: (4957, hidden_dim)
6. Save:
   - `results/activations/{model_short_name}/{cleaning}/layer_{LL:02d}.npz` with key `'activations'`
   - `results/activations/{model_short_name}/{cleaning}/metadata.json` with:
     - `model_id`, `cleaning`, `n_texts`, `n_layers`, `hidden_dim`
     - `text_ids` (list of fragment_ids, order matches rows)
     - `period_labels` (list of period strings, order matches rows)
     - `token_counts` (list of ints — how many tokens each text produced)
     - `timestamp`

**CLI:**
```bash
python 01_extract_activations.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --cleaning tier0 \
    --batch-size 8 \
    --max-length 512
```

**Sbatch** (`sbatch/01_extract.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=extract_acts
#SBATCH --partition=voltagepark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extract_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN

echo "=== Extracting activations (tier0) ==="
python v_1/src/linear_probing/01_extract_activations.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --cleaning tier0 \
    --batch-size 8 \
    || { echo "FAILED: tier0 extraction"; exit 1; }

echo "=== Extracting activations (maximal cleaning) ==="
python v_1/src/linear_probing/01_extract_activations.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --cleaning maximal \
    --batch-size 8 \
    || { echo "FAILED: maximal extraction"; exit 1; }

echo "=== Done ==="
```

**Verify:**
```python
import numpy as np, json
# Check one layer file
X = np.load('.../tier0/layer_16.npz')['activations']
assert X.shape == (4957, 4096), f"Wrong shape: {X.shape}"
assert not np.any(np.isnan(X)), "NaN detected"
assert np.std(X) > 0.01, "Near-zero activations"
assert not np.allclose(X[0], X[1]), "Identical rows"

# Check all 32+1 layer files exist and are ~same size
import os
files = sorted(os.listdir('.../tier0/'))
npz_files = [f for f in files if f.endswith('.npz')]
assert len(npz_files) == 33, f"Expected 33 files (embedding + 32 layers), got {len(npz_files)}"

# Check metadata
with open('.../tier0/metadata.json') as f:
    meta = json.load(f)
assert meta['n_texts'] == 4957
assert meta['hidden_dim'] == 4096
assert len(meta['period_labels']) == 4957
assert len(meta['token_counts']) == 4957
```

**Important:** The script should print progress (e.g., `"Batch 50/620 done"`) and total wall time. If it takes longer than 3 hours, reduce batch size or add `torch.cuda.empty_cache()` between batches.

---

## Step 2: `02_linear_probe.py`

**What it does:** Train linear probes at every layer, plot layer-accuracy curve, run confound controls, produce final test-set evaluation.

**Implementation:**

### 2a. Layer-accuracy curve
```python
for cleaning in ['tier0', 'maximal']:
    for layer in range(n_layers + 1):  # include embedding layer
        X = load_layer_activations(model, cleaning, layer)
        X_tv = X[train_val_idx]  # train+val only
        y_tv = labels[train_val_idx]

        # 5-fold CV with hyperparameter selection
        best_C, best_acc, best_f1 = None, 0, 0
        for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(C=C, penalty='l2', max_iter=1000,
                                     random_state=42, multi_class='multinomial')
            acc = cross_val_score(clf, X_tv, y_tv, cv=skf, scoring='accuracy').mean()
            f1 = cross_val_score(clf, X_tv, y_tv, cv=skf, scoring='f1_macro').mean()
            if acc > best_acc:
                best_C, best_acc, best_f1 = C, acc, f1

        results[cleaning][layer] = {
            'accuracy': best_acc, 'f1_macro': best_f1, 'best_C': best_C
        }
```

### 2b. Random-label baseline (at best layer)
```python
best_layer = argmax(results['tier0'][layer]['accuracy'] for layer in range(n_layers))
X_best = load_layer_activations(model, 'tier0', best_layer)[train_val_idx]
null_accs = []
for i in range(1000):
    y_shuffled = np.random.RandomState(42 + i).permutation(y_tv)
    clf = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    acc = cross_val_score(clf, X_best, y_shuffled, cv=skf, scoring='accuracy').mean()
    null_accs.append(acc)
p_value = (np.array(null_accs) >= results['tier0'][best_layer]['accuracy']).mean()
```

### 2c. Final test-set evaluation
```python
# Lock best layer + best C, retrain on full train+val
clf = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
clf.fit(X_tv, y_tv)
y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)
per_class = classification_report(y_test, y_pred, target_names=PERIODS, output_dict=True)
```

### 2d. Plots
1. **Layer-accuracy curve:** x=layer, y=accuracy. Two lines (tier0 blue, maximal green). Horizontal dashed lines for TF-IDF baselines (69.1%, 84.8%, 91.2%, 99.2%). Error bands from CV std.
2. **Random-label null:** histogram of 1,000 null accuracies + red line for real accuracy.
3. **t-SNE at 3 layers:** early (layer 2), best, late (layer 31). 3×2 grid (top row=tier0, bottom=maximal), colored by period.
4. **Confusion matrix:** at best layer, on test set.

**CLI:** `python 02_linear_probe.py --model llama-3.1-8b-instruct --n-permutations 1000`

**Sbatch** (`sbatch/02_probe.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=lin_probe
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
python v_1/src/linear_probing/02_linear_probe.py \
    --model llama-3.1-8b-instruct \
    --n-permutations 1000
```

Note: NO GPU needed. This is pure sklearn on numpy arrays.

**Verify:**
- Random-label accuracy should be 33-40%. If higher, labels are not shuffled correctly.
- Layer curve should be smooth (no ±30% jumps between adjacent layers).
- Test accuracy within ~5% of CV accuracy at best layer.
- Confusion matrix: not all predictions in one class.
- Two cleaning conditions should produce two different curves (if identical, cleaning had no effect OR activations were loaded wrong).

---

## Step 3: `03_analyze_results.py`

**What it does:** Read Step 2 outputs, classify outcome (A/B/C), produce summary and time direction if applicable.

**Implementation:**
1. Load `probe_results_{model}.json`
2. Determine best layer for tier0 and maximal cleaning
3. Compare:
   - Best accuracy vs. cleaned unigram floor (69.1%) → minimum bar
   - Best accuracy vs. cleaned bigram (91.2%) → strong evidence
   - Best layer position: early (0-3) vs. mid (4-24) vs. late (25+)
   - Layer shift between tier0 and maximal cleaning
   - Permutation p-value
4. Classify outcome A/B/C per criteria in PLAN.md
5. If Outcome A:
   - Retrain probe at best layer on all train+val data
   - Save probe weight vector as `results/time_direction_{model}.npy` shape (hidden_dim,)
   - Project all activations onto this direction: `scores = X @ direction` → shape (4957,)
   - Plot t-SNE of raw activations vs. projected (1D scores as x-axis, noise as y-axis)
6. Save summary JSON with all metrics, outcome, and next steps

**CLI:** `python 03_analyze_results.py --model llama-3.1-8b-instruct`

**Sbatch** (`sbatch/03_analyze.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analyze_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
python v_1/src/linear_probing/03_analyze_results.py \
    --model llama-3.1-8b-instruct
```

**Verify:**
- Outcome classification is consistent with the numbers.
- If Outcome A: time direction vector norm > 0, projected scores show separation between periods (plot OB/NA/LB score histograms — should be shifted).
- Summary JSON contains all fields: best_layer, test_acc, test_f1, p_value, outcome, per_class metrics, cleaning comparison.
