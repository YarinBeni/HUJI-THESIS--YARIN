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

## Analysis Notebooks

The scripts above answer *whether* bias exists. Two notebooks answer *why* — they are the deeper diagnostic layer and contain the most interpretable results.

### `bias_analysis.ipynb` — Eval set (4,957 letters)

13-section investigation into what drives the 99% classifier accuracy:

| Section | What it does |
|---------|-------------|
| 2. Metadata Leaks | Shows `corpus_source` is a perfect proxy for period (archibab→OB, oracc→NA, lbl→LB). The #1 root cause. |
| 3. Text Length | Length distributions per period — weak signal (~60% LR accuracy alone). |
| 4. Top Discriminative N-grams | Per-period mean TF-IDF + log-odds ratio — which char n-grams are most characteristic. |
| 5. Diacritic Frequencies | Frequencies of ā,ī,ū,ē,š,ṣ,ṭ,ḫ across periods (~54% alone — near chance). |
| 6. Number Suffix Conventions | Homophone subscript digits (sign2, sign3) vary by corpus tradition (~60% alone). |
| 7. Vocabulary Overlap | Period-exclusive vs shared word-level tokens — exclusive rate alone is ~50% (chance). |
| 8. Logogram / Uppercase Ratio | Sumerian logograms (UPPERCASE) rate per period (~65% alone). |
| 9. Summary | Ranked table: all hand-crafted features combined → 77.9% — still below TF-IDF's 99%. Signal is distributed. |
| 10. Cleaning Ablation | **11-step greedy cleaning** showing accuracy drop per step (see below). |
| 11. t-SNE | 2D visualisation of TF-IDF clusters — three clearly separated period clouds even at unigram level. |
| 12. Entity Bias | Deity & place name mention rates per period (e.g. d-EN-ZU: 16.4% OB / 0.2% NA). Entity features alone → 72.4%. |
| 13. PLS-DA + VIP scores | Partial Least Squares Discriminant Analysis — quantifies each n-gram's contribution to classification. Top unigram VIPs: `g` (2.20), `w` (2.14), `m` (1.97). Cross-dataset comparison vs finetune set in §13d. |

**Cleaning ablation results** (greedy, 11 steps applied on top of tier0):

| N-gram range | Baseline | After all 11 steps | Drop |
|---|---|---|---|
| 2-5 grams | 99.2% | 96.8% | −2.4% |
| Bigrams | 98.3% | 91.2% | −7.2% |
| Unigrams | 84.8% | 69.3% | −15.4% |

Accuracy does not drop to chance because syllabic structure differs across periods at a linguistic level — you can remove markup conventions but not Akkadian phonology. The 11-step cleaning pipeline becomes `clean_maximal` in the linear probing pipeline (`v_1/src/linear_probing/utils.py`).

**Three-tier signal hierarchy** (summary of all sections):
1. **Writing conventions** — determinatives, subscript digits, logogram rates. Removable by cleaning.
2. **Phonology & morphology** — syllable patterns, case endings, consonant distributions. Partially removable.
3. **Content/pragmatics** — deity names, place names, formulaic expressions. Not removable without destroying meaning.

### `bias_analysis_finetune.ipynb` — Finetune set (10,435 fragments)

Repeats the core analysis on the finetune dataset (more fragments, shorter texts): TF-IDF + LR with 5-fold CV, t-SNE, PLS-DA with VIP scores. The cross-dataset VIP comparison (§13d in the eval notebook) identifies which features are robust across both datasets vs. corpus-specific artifacts.

---

---

## Multi-task SEAL Mode (Phase C)

The script `06_bias_check_cv.py` runs TF-IDF + LR bias checks across the 6 SEAL/DLL/LBPL
metadata tasks. Unlike the letters pipeline (NN variants, fixed split), this uses logistic
regression only with adaptive-k stratified CV — appropriate for the small 246–384 fragment
datasets.

```bash
# Debug (domain/tier0, 100 perms, no plots):
python3 v_1/src/bias_check/06_bias_check_cv.py --debug

# Full run (all 6 tasks × tier0 + maximal, 1000 perms, plots):
python3 v_1/src/bias_check/06_bias_check_cv.py --plots

# Single task/cleaning:
python3 v_1/src/bias_check/06_bias_check_cv.py --tasks domain --cleanings tier0
```

**Data source**: `seal_tasks.py` registry → `data/evaluation/corpora/seal_corpus.parquet`

**Output**: `data/evaluation/bias_check/seal_round4/<task>/<cleaning>/`
- `task_summary.json` — N, classes, singletons dropped, k
- `metrics.json` — CV scores, C grid, per-class F1, permutation test
- `report.md` — human-readable summary + confusion matrix
- `plots/` — confusion.png, perm_null.png, per_class_f1.png

**Phase C results** (ran 2026-04-07, all 12 combinations):

| Task | tier0 F1 | maximal F1 | p (both) |
|------|----------:|----------:|------:|
| domain | 0.952 | 0.889 | 0.001 |
| period | 0.608 | 0.464 | 0.001 |
| genre | 0.361 | 0.269 | 0.001 |
| sub_genre | 0.286 | 0.267 | 0.001 |
| provenance | 0.171 | 0.128 | 0.001 |
| sub_provenance | 0.171 | 0.128 | 0.001 |

All FAIL (p=0.001). Signal is genuine — diachronic and domain structure is detectable from
text surface form for all 6 tasks. This is a **positive result**: the signal exists and is
real, validating the linear probing pipeline.

---

## Dependencies

Requires packages from `v_1/requirements.txt`:
`torch`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`
