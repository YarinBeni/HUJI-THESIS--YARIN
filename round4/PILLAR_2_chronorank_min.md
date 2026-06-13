# Pillar 2 — Minimal ChronoRank (pairwise rank + interval loss)

> **Agent brief.** This is the core of Thrust B and the lowest-risk model experiment in the
> whole round. Read `README.md` first, and **P0 must be merged** (you import `labels.py`,
> `eval_ordinal.py`). CPU-only — frozen embeddings are already on disk; you train a tiny head.
> Run a local sanity check, then hand Yarin one small sbatch.

## Goal

Replace "frozen embedding → PLS regression" with "frozen embedding → small head trained with the
**correct ordinal objective**". Test whether the right objective alone (no bigger model, no
finetuning) improves or matches the Thalesian PLS baseline while giving calibrated uncertainty.

This is the plan's **Stage 1**: `L = L_pairwise_rank + L_interval`.

## Dependencies

**P0 (required):** `chronorank/labels.py` (intervals, pairs, ruler-grouped folds),
`chronorank/eval_ordinal.py` (`full_report`). Frozen activations already on disk (§3). No GPU.

## What to read (repo)

- `README.md` §3 + §4 (loaders, the maximal-balanced harness, what "success" means).
- `v_1/src/geodesic/utils.py` — `find_acts_dir('thalesian_akk300m','maximal','mean')`,
  `load_layer`, `load_year_labels`. **This is how you get z_i.** Best layer ≈ L11.
- `v_1/src/linear_probing/pls_utils.py` — the PLS baseline you must beat/match (`fit_pls_groupkfold`)
  and `l2_normalize` (apply the same normalization to your inputs for a fair comparison).
- `v_1/src/linear_probing/round2_phase0/run_mc_probes.py` — the MC-balanced draws machinery; reuse
  `draws_matrix.npy` + `corpus_fragment_order.json` so your scores are on the same balanced subsets
  as every other method in the scoreboard.

## What to read (papers)

- **CORAL / CORN** — rank-consistent ordinal regression (the interval/ordinal target framing).
- **Learning-to-rank, pairwise (RankNet logistic pairwise loss)** — the `L_pair` form below.
- **Rank-N-Contrast (RnC)** — *skim*; full RnC is P3-adjacent. For P2 you only need the pairwise margin idea.

## What to build

### `v_1/src/chronorank/model.py`
```python
class ChronoHead(nn.Module):
    """frozen z (d,) -> (s, mu, log_sigma). Variants via arg: 'linear' | 'mlp1' | 'sparse_linear'.
    s = chronological score (monotone in time); mu = predicted year (BCE); sigma from log_sigma.
    Keep it TINY — this is the whole point (small trainable part over frozen features)."""

def loss_pairwise(s_i, s_j, margin):  # logistic pairwise: log(1+exp(-(s_j - s_i - margin)))
def loss_interval(mu, log_sigma, low, high):  # -log P(low <= year <= high | mu, sigma), Gaussian CDF
```

### `v_1/src/chronorank/train.py`
- Load Thalesian L11 maximal/mean embeddings via `find_acts_dir`/`load_layer`; `l2_normalize`.
- Build interval targets + balanced non-ambiguous pairs from `labels.py`.
- Train `ChronoHead` with `L = L_pairwise_rank + λ·L_interval` under **GroupKFold-by-ruler**
  (no ruler appears in both train and val — this is the anti-leak guard).
- Evaluate with `eval_ordinal.full_report` on the **same MC-balanced draws** PLS uses.
- CLI: `--method thalesian_akk300m --layer 11 --cleaning maximal --pool mean --head linear|mlp1 --lambda-interval 1.0`.

### Comparison table (the deliverable)
Run the head on the matrix the plan specifies and print one table:
```
PLS baseline (existing) | linear ChronoRank | ChronoRank + interval | mlp1 ChronoRank
  × backbones: Thalesian L11 | Qwen3-8B L16 | Qwen3-1.7B L9 | TF-IDF | random
```
Metrics per cell: spearman, mae, pairwise_order_acc, picp@80, picp@90.

## Cluster / sbatch

CPU-only and fast. One sbatch:
```bash
# v_1/src/chronorank/sbatch/P2_chronorank.sbatch  (no --gres)
#SBATCH --cpus-per-task=32 --mem=64G --time=02:00:00
# ... standard preamble (README §2) ...
for HEAD in linear mlp1; do
  for M in thalesian_akk300m qwen3_8b qwen3_1b7 random; do
    python -u v_1/src/chronorank/train.py --method $M --cleaning maximal --pool mean \
        --head $HEAD --lambda-interval 1.0 --out v_1/src/chronorank/results
  done
done
python -u v_1/src/chronorank/train.py --tfidf --head linear --out v_1/src/chronorank/results
# then commit results/*.json + push (FT3 pattern)
```
Run `train.py` locally on Thalesian L11 first (it's seconds on CPU) and paste the table into handoff.
Give Yarin: `cd ~/projects/HUJI-THESIS--YARIN && git pull && sbatch v_1/src/chronorank/sbatch/P2_chronorank.sbatch`.

## Report back / success criterion

**PASS** when the comparison table is filled and you can state plainly: *does the ordinal
objective match or beat PLS on Thalesian L11 under maximal-balanced, and does it now give
non-trivial interval coverage?* Per README §4, **matching PLS Spearman (~0.41) while adding
calibrated coverage is already a PASS** — a modest Spearman gain is a bonus, not the bar.
Report the numbers honestly even if flat. This unblocks P3, P5, P6.
