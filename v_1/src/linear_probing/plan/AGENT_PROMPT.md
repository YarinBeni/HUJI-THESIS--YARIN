# Agent Prompt: Implement Linear Probing Pipeline

## Your Task

Implement the linear probing pipeline for testing whether LLMs encode Akkadian temporal period (OB/NA/LB) linearly in their activations. You are writing Python scripts and Slurm sbatch files that will run on an HPC cluster (Schmidt Sciences, H100 GPUs).

**Do NOT run any scripts locally.** All execution happens on the cluster via sbatch. Your job is to write correct, complete code that can be git-pushed and then run on the cluster.

## Files You MUST Read First (in order)

1. **`v_1/src/linear_probing/plan/PLAN.md`** — the full implementation plan. Contains success criteria, evaluation protocol, confound controls, file structure, and what each step does. This is your primary spec.

2. **`v_1/src/linear_probing/plan/IMPLEMENTATION_PROMPTS.md`** — detailed implementation instructions for each script. Contains pseudocode, CLI args, sbatch templates, and verification checks. Follow these closely.

3. **`v_1/src/cluster/README.md`** — cluster setup. Key facts:
   - Partition: `voltagepark`
   - Conda env: `thesis` at `~/miniconda3/envs/thesis/`
   - Repo on cluster: `~/projects/HUJI-THESIS--YARIN`
   - Python 3.11, PyTorch 2.10, Transformers 5.3, sklearn
   - 1 GPU is enough for 8B models

4. **`v_1/src/cluster/test_model_load.py`** — working example of loading a HuggingFace model and extracting hidden states on the cluster. Use the same pattern: `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True)`.

5. **`v_1/src/bias_check/run_bias_check.sh`** — working sbatch template with fail-fast pattern (`|| { echo "FAILED"; exit 1; }`). Follow this style for all sbatch files.

6. **`v_1/src/bias_check/bias_analysis_finetune.ipynb`** — cell 7 has `clean_tier0()`, cell 16 has all 11 cleaning filter definitions. Copy these into `utils.py`.

7. **`v_1/data/evaluation/corpora/`** — the data directory. The parquet file `texts_for_evaluation.parquet` has the 4,957 letters.

## What to Implement

Implement these files in order. Each script must be self-contained (imports, argparse, main guard).

### 1. `utils.py`
Shared utilities. All other scripts import from this.
- `SEED = 42`
- `load_letters()` → loads the parquet, returns DataFrame with `text`, `period`, `fragment_id` columns
- `clean_tier0(text)` → copy from bias check notebook cell 7
- `clean_maximal(text)` → apply all 11 filters from bias check notebook cell 16, stacked in order
- `get_splits(df)` → 70/15/15 stratified split using `StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)` applied twice (first to get test, then to split remainder into train/val). Returns `(train_idx, val_idx, test_idx)`.
- `TFIDF_BASELINES` dict with the accuracy numbers from the bias check
- `PERIOD_COLORS` dict for plotting
- Model short name helper: `"meta-llama/Llama-3.1-8B-Instruct"` → `"llama-3.1-8b-instruct"`

### 2. `00_tokenization_check.py`
- Takes `--model` arg (HuggingFace model ID)
- Loads tokenizer only (no model needed, no GPU needed — but sbatch gives GPU anyway because tokenizer download sometimes needs it)
- Tokenizes ALL 4,957 texts (not just 10) to get full statistics
- Prints 10 sample texts with their tokens (3 OB, 4 NA, 3 LB)
- Saves summary stats to `results/tokenization_check.json`: per-text token counts, per-period mean/median/std token counts, sample tokenizations

### 3. `00b_quick_eda.py`
- Takes `--model` arg
- Loads model, extracts final-layer mean-pooled embeddings for all 4,957 texts
- **Critical: use `attention_mask` when mean-pooling** — do NOT average over padding tokens
- Runs PCA (2D) and t-SNE (2D, perplexity=40). Skip UMAP if not installed.
- Saves plot to `results/plots/quick_eda_final_layer.png`
- Also saves the embeddings array to `results/activations/{model}/final_layer_only.npz` (useful for quick iteration without re-extracting)

### 4. `01_extract_activations.py`
- Takes `--model`, `--cleaning {tier0,maximal}`, `--batch-size 8`, `--max-length 512`
- Loads model with `output_hidden_states=True`
- Processes all 4,957 texts in batches
- **Critical: use `attention_mask` for mean-pooling** — sum hidden states where mask=1, divide by mask sum
- Saves one `.npz` per layer (including layer 0 = embedding layer) with key `'activations'`
- Saves `metadata.json` with text IDs, labels, token counts, model info
- Prints progress every 50 batches and total wall time at the end

### 5. `02_linear_probe.py`
- Takes `--model` (short name matching the activations directory), `--n-permutations 1000`
- Loads activations from `results/activations/{model}/tier0/` and `results/activations/{model}/maximal/`
- Loads metadata to get labels and splits
- Runs the full evaluation protocol from PLAN.md:
  - 5-fold CV layer-accuracy curve for both cleaning conditions
  - Hyperparameter search over C values at each layer
  - Random-label baseline at best layer (1,000 permutations)
  - Final test-set evaluation at locked best layer + best C
  - Full metrics: accuracy, F1 macro, per-class precision/recall/F1, confusion matrix
- Produces all plots listed in PLAN.md
- Saves everything to `results/probe_results_{model}.json`

### 6. `03_analyze_results.py`
- Takes `--model` (short name)
- Reads `probe_results_{model}.json`
- Classifies outcome as A, B, or C per PLAN.md criteria
- If Outcome A: extracts time direction vector, projects activations, saves plots
- Produces `results/summary_{model}.json` with final verdict

### 7. Sbatch files in `sbatch/`
Create one `.sh` file per step. Follow the pattern from `run_bias_check.sh`:
- Set `--partition=voltagepark`
- Use `source ~/miniconda3/etc/profile.d/conda.sh && conda activate thesis`
- `cd ~/projects/HUJI-THESIS--YARIN`
- Use fail-fast: `|| { echo "FAILED: ..."; exit 1; }`
- Create `logs/` directory in the sbatch if it doesn't exist: `mkdir -p v_1/src/linear_probing/logs`
- Output goes to `v_1/src/linear_probing/logs/`

GPU allocation:
- `00_tokenization.sh` → 1 GPU, 32GB RAM, 30 min
- `00b_quick_eda.sh` → 1 GPU, 64GB RAM, 1 hour
- `01_extract.sh` → 1 GPU, 64GB RAM, 4 hours (runs tier0 then maximal sequentially)
- `02_probe.sh` → NO GPU, 16 CPUs, 32GB RAM, 2 hours
- `03_analyze.sh` → NO GPU, 8 CPUs, 16GB RAM, 30 min

## Key Technical Details

### Mean pooling with attention mask
```python
# CORRECT — only average over real tokens, not padding
def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
    summed = (hidden_states * mask).sum(dim=1)   # (batch, hidden_dim)
    counts = mask.sum(dim=1).clamp(min=1)        # (batch, 1)
    return summed / counts
```

### Accessing all hidden states in one forward pass
```python
outputs = model(**inputs, output_hidden_states=True)
# outputs.hidden_states is a tuple of (n_layers + 1) tensors
# outputs.hidden_states[0] = embedding layer output
# outputs.hidden_states[1] = layer 1 output
# outputs.hidden_states[-1] = final layer output
# Each tensor shape: (batch_size, seq_length, hidden_dim)
```

### Maximal cleaning — all 11 filters stacked
Copy the filter definitions from the bias check notebook. The maximal cleaning applies ALL of them in sequence:
```python
def clean_maximal(text):
    t = clean_tier0(text)
    for name, fn in ALL_FILTERS.items():
        t = fn(t)
    return t
```

### Data split consistency
The bias check used `StratifiedShuffleSplit` with `random_state=42`. You MUST use the same split so that the TF-IDF baselines are directly comparable. If the bias check used a different split method, check `v_1/src/bias_check/config.py` for `SPLIT_SEED`, `TRAIN_RATIO`, etc. and match exactly.

### Llama-3.1-8B-Instruct specifics
- HuggingFace ID: `meta-llama/Llama-3.1-8B-Instruct`
- May require HuggingFace token for gated model access. Check if it's already cached on the cluster at `~/.cache/huggingface/`. If not, the user will need to run `huggingface-cli login` on the cluster first.
- 32 transformer layers + 1 embedding layer = 33 hidden states
- Hidden dim: 4096
- If the model doesn't load (gated access), fall back to `Qwen/Qwen2.5-7B-Instruct` which is already verified working on the cluster (28 layers, hidden dim 3584). Adjust layer counts and hidden dim expectations accordingly.

## File Structure to Create
```
v_1/src/linear_probing/
├── utils.py
├── 00_tokenization_check.py
├── 00b_quick_eda.py
├── 01_extract_activations.py
├── 02_linear_probe.py
├── 03_analyze_results.py
├── sbatch/
│   ├── 00_tokenization.sh
│   ├── 00b_quick_eda.sh
│   ├── 01_extract.sh
│   ├── 02_probe.sh
│   └── 03_analyze.sh
├── logs/                            (create empty, sbatch output goes here)
└── results/                         (created by scripts at runtime)
```

## Verification Checklist

After implementation, verify each script by reading through it and checking:

- [ ] `utils.py`: `load_letters()` returns 4,957 rows; `clean_tier0` removes `@v`; `clean_maximal` produces lowercase syllabic-only; splits sum to 4,957
- [ ] `00_tokenization_check.py`: handles `--model` arg; saves JSON; tokenizes all 4,957 texts for stats
- [ ] `00b_quick_eda.py`: uses `attention_mask` in mean pooling; saves plot with 4,957 dots; handles model loading correctly
- [ ] `01_extract_activations.py`: uses `attention_mask`; saves 33 `.npz` files (layers 0-32); `metadata.json` has all fields; handles `--cleaning` flag; prints progress
- [ ] `02_linear_probe.py`: loads from correct paths; uses 70/15/15 split; CV on train+val only; test set touched once; permutation test shuffles correctly; all plots generated
- [ ] `03_analyze_results.py`: outcome classification matches PLAN.md criteria; time direction saved if Outcome A
- [ ] All sbatch files: correct partition, conda activation, cd to repo, fail-fast, correct resource allocation (GPU vs CPU)
- [ ] No hardcoded absolute paths in Python scripts — use relative paths from the script location or CLI args
- [ ] All scripts have `if __name__ == '__main__':` guard
