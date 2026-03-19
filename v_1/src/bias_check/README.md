# Bias Check Pipeline

Pre-Track-A validation: verifies that the test set carries no exploitable surface-level temporal signal before running LLM evaluation.

**Research question**: Can a simple classifier distinguish Old Babylonian / Neo-Assyrian / Late Babylonian texts from transliteration alone? If yes, the benchmark is biased.

**Target**: ~33% accuracy (chance level). Statistically assessed via permutation testing.

---

## Quick Start

```bash
# From repo root. All commands run the scripts directly from the project root.

# Step 1: Featurize (requires texts_for_evaluation.parquet from evaluation pipeline)
python v_1/src/bias_check/01_featurize.py

# Step 2: Train all 8 model variants
python v_1/src/bias_check/02_train.py

# Step 3: Evaluate (computes metrics + permutation tests)
python v_1/src/bias_check/03_evaluate.py

# Step 4: Generate plots
python v_1/src/bias_check/04_plot.py

# Step 5: Generate markdown report
python v_1/src/bias_check/05_report.py
```

**Local debug mode** (10% of data, fast CPU run):
```bash
python v_1/src/bias_check/01_featurize.py --debug
python v_1/src/bias_check/02_train.py --debug
python v_1/src/bias_check/03_evaluate.py --n-permutations 50
python v_1/src/bias_check/04_plot.py
python v_1/src/bias_check/05_report.py
```

**Cluster** (Schmidt Sciences HPC, H100):
```bash
# On cluster web terminal:
cd ~/projects/lititure-review && git pull
mkdir -p logs
sbatch v_1/src/bias_check/run_bias_check.sh

# Monitor:
squeue -u $USER
tail -f logs/bias_check_*.out
```

---

## Pipeline Flow

```
v_1/data/evaluation/corpora/texts_for_evaluation.parquet
        │   (4,957 texts, 3 classes)
        ▼
[01_featurize.py]   TF-IDF char n-grams (2–5) → stratified 70/15/15 splits
        │
        ▼
v_1/data/evaluation/bias_check/features/
├── train.npz  val.npz  test.npz   ← sparse TF-IDF matrices
├── y_train.npy  y_val.npy  y_test.npy
└── vectorizer.pkl
        │
        ▼
[02_train.py]   8 model variants (MLP + Attention+MLP), early stopping
        │
        ▼
v_1/data/evaluation/bias_check/models/
├── mlp_1layer.pt  mlp_2layer.pt  ...   ← best checkpoints
└── training_history.json
        │
        ▼
[03_evaluate.py]   test metrics + permutation test (1000 perms, LR proxy)
        │
        ▼
v_1/data/evaluation/bias_check/metrics/
└── all_metrics.json
        │
        ├─→ [04_plot.py] → plots/
        └─→ [05_report.py] → bias_check_report.md
```

---

## Model Variants

| Name | Type | n_attn_blocks | n_mlp_layers | ~Params |
|------|------|:---:|:---:|:---:|
| mlp_1layer | MLP only | 0 | 1 | 2.6M |
| mlp_2layer | MLP only | 0 | 2 | 2.7M |
| mlp_3layer | MLP only | 0 | 3 | 2.8M |
| mlp_5layer | MLP only | 0 | 5 | 2.9M |
| attn1_mlp3 | Attn+MLP | 1 | 3 | 0.3M |
| attn2_mlp3 | Attn+MLP | 2 | 3 | 0.4M |
| attn3_mlp3 | Attn+MLP | 3 | 3 | 0.5M |
| attn5_mlp3 | Attn+MLP | 5 | 3 | 0.7M |

MLP sweep isolates depth effect; Attention sweep (fixed 3-layer MLP head) isolates attention effect.

---

## Featurization Design

**TF-IDF char n-grams (not learned embeddings)**:

1. **Small dataset**: ~3,500 training samples → learned embeddings overfit. TF-IDF has zero learnable parameters.
2. **Interpretability**: TF-IDF failure is unambiguous. Learned-embedding failure is not.
3. **Correct signal level**: `char_wb` n-grams (2–5) capture orthographic/morphological patterns exactly as flagged by Chungrong (sign choices, spelling conventions, morphological suffixes, determinatives).
4. **Academic standard**: Ojala & Garriga (JMLR 2010) established permutation testing with simple classifiers as the gold standard for dataset bias checks.

---

## Verdict Criteria

| p-value | Verdict | Meaning |
|---------|---------|---------|
| < 0.01  | ❌ FAIL | Statistically significant bias — halt Track A |
| 0.01–0.05 | ⚠️ WARN | Marginal — investigate before proceeding |
| ≥ 0.05  | ✅ PASS | No significant bias — proceed to Track A |

Overall verdict is the worst case across all 8 models.

---

## Output Files

```
v_1/data/evaluation/bias_check/
├── features/
│   ├── train.npz / val.npz / test.npz
│   ├── y_train.npy / y_val.npy / y_test.npy
│   └── vectorizer.pkl
├── models/                          ← best checkpoint per variant (.pt)
├── metrics/
│   ├── all_metrics.json             ← test metrics + permutation p-values
│   └── training_history.json        ← per-epoch train/val curves
├── plots/
│   ├── accuracy_vs_complexity.png
│   ├── f1_per_class.png
│   ├── permutation_test.png
│   ├── confusion_<name>.png  (×8)
│   └── training_curves.png
└── bias_check_report.md             ← final verdict + full results
```

---

## Configuration

All hyperparameters, paths, and model definitions are in `config.py`:
- `TFIDF_KWARGS` — featurization settings
- `MODEL_VARIANTS` — 8 model definitions
- `HIDDEN_DIM`, `ATTN_DIM`, `DROPOUT`, `BATCH_SIZE`, `LEARNING_RATE` — training
- `N_PERMUTATIONS`, `PVALUE_FAIL`, `PVALUE_WARN` — permutation test thresholds

---

## Dependencies

Requires packages from `v_1/requirements.txt`:
`torch`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`
