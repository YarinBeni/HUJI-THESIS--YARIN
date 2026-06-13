# Pillar 0 — Shared harness: labels, views, ordinal-eval metrics

> **Agent brief.** You are implementing the foundation every other Round 4 pillar
> imports. Read `README.md` in this folder first (cluster + data layout). This pillar
> is **CPU-only and fast** — run it locally and on the cluster both. Do not start P2/P3/P4
> until this lands and Yarin has seen the smoke-test output.

## Goal

Three small, well-tested modules under `v_1/src/chronorank/` that turn the existing
ORCC data + frozen activations into the ingredients the ordinal model and its evaluation
need:

1. **`labels.py`** — ruler→interval table and pairwise/ordinal constraint generation.
2. **`transforms.py`** — the augmentation/masking views as **runtime, composable callables**
   (PyTorch-transform style), used for consistency training & robustness eval.
3. **`eval_ordinal.py`** — the ordinal/calibration metrics that extend the existing PLS scoreboard.

## Dependencies

None upstream. You are the root. Downstream: P2, P3, P4, P6 import you; P1 uses `eval_ordinal.py`.

## What to read (repo)

- `round4/README.md` §3 (paths, labeling facts — **the genre
  caveat and the "year = BCE, ruler=interval" facts are load-bearing**).
- `v_1/src/linear_probing/utils.py` — SEED, 70/15/15 split ratios, `PERIOD_MAP`, the 11-filter
  `clean_maximal`. **Reuse these constants; do not redefine them.**
- `v_1/src/linear_probing/pls_utils.py` — `compute_metrics`, `l2_normalize`, `fit_pls_groupkfold`
  (GroupKFold *by ruler* — your splits must do the same to prevent leakage).
- `v_1/src/geodesic/utils.py` — `find_acts_dir`, `load_layer`, `load_year_labels` (the loaders
  you'll reuse everywhere). Look at how `pairwise_order_acc` is already defined — match its
  semantics so numbers are comparable across pillars.
- ORCC masked text columns already exist (`text_tier0_masked`, `text_maximal_masked`) — inspect
  what masking they apply before you re-implement name masking in `views.py`. Also
  `v_1/src/linear_probing/name_masking.py`.

## What to read (papers — only these for P0)

- **CORAL / CORN** (Cao et al.; Shi et al.) — rank-consistent ordinal regression; motivates why
  interval/ordinal targets beat multiclass. You only need the target/loss framing.
- **Calibration for regression** (interval coverage / NLL) — any standard reference; you're
  implementing PICP (prediction-interval coverage probability) and Gaussian NLL.

## What to build

### `v_1/src/chronorank/labels.py`
```python
def ruler_intervals(orcc_df) -> dict[str, tuple[int,int]]:
    """ruler -> (year_min, year_max) in BCE. Single-year rulers => (y, y).
    Esarhaddon is the notable multi-year span. Drop rows with NA year (e.g. 'ribo')."""

def make_interval_targets(orcc_df) -> np.ndarray:
    """(N,2) array of [low, high] BCE bounds per fragment, from its ruler interval.
    For single-year rulers, optionally widen by a small ± pad (arg, default 0)."""

def make_pairs(years, rulers, *, margin_years=50, max_pairs=None, balance_by_ruler=True,
               seed=42) -> np.ndarray:
    """Generate (i, j) ordered pairs where year_i and year_j differ by >= margin_years
    AND the two rulers' intervals do NOT overlap (no ambiguous pairs). Balance sampling
    by ruler so Ashurbanipal (n=268) doesn't dominate the 41-ruler tail. Return (P,2)
    int array with convention: text i is EARLIER (larger BCE) than text j? Document the
    direction explicitly and keep it consistent with eval_ordinal.pairwise_order_acc."""

def groupkfold_by_ruler(rulers, n_splits=5, seed=42):
    """Thin wrapper so every pillar gets identical ruler-grouped folds. Reuse the
    GroupKFold logic already in pls_utils.fit_pls_groupkfold."""
```
Decide and **document in a module docstring** the sign convention for "earlier" once
(BCE: earlier = larger number) and reuse it everywhere — this is the #1 source of silent bugs.

### `v_1/src/chronorank/transforms.py`  ← RUNTIME transforms, not pre-saved (Yarin's call)
**Design rule (see README "Engineering principle"):** these are **callables applied on the fly
in the data loader's `__getitem__`**, exactly like image transforms in torchvision — the corpus
is loaded once and each view is computed per-sample at batch time. Do **not** materialize N copies
of the text as parquet columns or in-memory arrays. This keeps memory flat and makes adding a new
view a one-line change.
```python
class TextTransform:                # base: __call__(text:str)->str, composable
    ...
class NameMask(TextTransform): ...     # reuse name_masking.py logic; mask ruler/divine/place/official
class FormulaRemove(TextTransform): ...# strip royal opening/closing formulae
class Crop(TextTransform):             # def __init__(self, n_words=32)
    ...
class Normalize(TextTransform): ...    # diacritics/sign normalization
class Compose:                         # Compose([NameMask(), Crop(32)]) -> single callable
    ...
VIEWS = {"orig": Compose([]), "name_masked": NameMask(), "formula_removed": FormulaRemove(),
         "crop32": Crop(32), "normalized": Normalize()}
```
**Reuse, don't reinvent:** `name_masking.py` already implements the masking regex, and ORCC ships
`text_tier0_masked`/`text_maximal_masked` columns. Your `NameMask` callable should wrap that same
logic so a runtime call reproduces those columns exactly (verify on a few rows). Only `FormulaRemove`
is genuinely new. The pre-saved masked columns remain useful as a cache/sanity-check, but the
*training path* calls the transform at load time.

### `v_1/src/chronorank/eval_ordinal.py`
```python
def pairwise_order_accuracy(pred_score, years, margin_years=50) -> float:
    """Match geodesic.utils.pairwise_order_acc semantics exactly (call it if convenient)."""
def interval_coverage(mu, sigma, low, high, level=0.8) -> float:
    """PICP: fraction of fragments whose ruler interval is covered by the level% predictive
    interval. Report at 0.8 and 0.9."""
def gaussian_nll(mu, sigma, year_point) -> float: ...
def regression_block(y_true, y_pred, y_train) -> dict:
    """Wrap pls_utils.compute_metrics so spearman/mae/mase are identical to the scoreboard."""
def full_report(pred) -> dict:
    """One dict combining: spearman, mae, mase, pairwise_order_acc, picp@80, picp@90, nll.
    This is the canonical Round-4 metric block all pillars print."""
```

## Cluster / sbatch

P0 is CPU-only and tiny. Provide **one** smoke-test sbatch that builds the label table and
runs `full_report` on the existing Thalesian L11 PLS predictions, to prove the metrics line
up with the known Spearman ≈ 0.41.

```bash
# v_1/src/chronorank/sbatch/P0_smoke.sbatch  (CPU only — no --gres)
#!/bin/bash
#SBATCH --job-name=cr_P0_smoke
#SBATCH --partition=voltagepark
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=v_1/src/chronorank/logs/%j.out
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/projects/HUJI-THESIS--YARIN
git pull --rebase origin main || echo "WARN pull failed"
python -u v_1/src/chronorank/eval_ordinal.py --smoke   # fit PLS on Thalesian L11, print full_report
```
Tell Yarin to paste: `cd ~/projects/HUJI-THESIS--YARIN && git pull && sbatch v_1/src/chronorank/sbatch/P0_smoke.sbatch`.
Also run the same `--smoke` locally first and paste the output into your handoff.

## Report back / success criterion

**PASS** when: (a) `make_pairs` produces a balanced, non-ambiguous pair set whose size and
per-ruler histogram you print; (b) `full_report` on Thalesian L11 reproduces the known
Spearman ≈ 0.41 (±0.01) from the existing scoreboard — this proves your metric wrapper is
consistent with the established pipeline; (c) all five transforms run **as runtime callables**
on 3 sample ORCC texts (paste before/after), and `NameMask` reproduces the existing
`text_maximal_masked` column on those rows. Hand Yarin the smoke `.out` and the pair-count histogram.
