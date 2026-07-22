# P10 — reduce-then-kernel (the advisor's "help extract the signal" idea)

**Question (advisor):** the chronology manifold may be easier to read after a low-dim
embedding. So reduce the activations *first* (PCA / PLS / UMAP → 3D), optionally
normalize, and *then* run the P9 geodesic/RBF kernels and the P8 supervision dial —
does pre-reducing help?

## What it does

Inside every CV fold (fit on train, apply to test — **no leakage**):
1. **reduce** to `dims`=3 with one of `raw` / `pca` / `pls` / `umap`
   (PLS supervised → uses train years; PCA/UMAP unsupervised; `raw` = no reduction),
2. optionally **normalize** the reduced coords: `none` / `zscore` / `l2` (fit on train),
3. run the same estimators as P9/P8 on the reduced features:
   - **gkpls** (geodesic kernel PLS), **rbfkpls** (Euclidean RBF kernel PLS),
     **krr_geo** (kernel ridge on the geodesic Gram) — reused from `p9_gkpls/gkpls.py`,
   - the **supervision dial** (λ from 1=pure geometry to 0=pure supervision) — reused
     from `p8_lambda_probe/lambda_probe.py`.

Balanced-MC exactly as P8/P9: 200 balanced draws × GroupKFold-by-ruler, mean±std over
draws, at the **P9-best layer** per method. Target = **year** (Spearman). `raw/none`
reproduces the P9/P8 numbers — the anchor every reduce+norm cell is compared against.

`t-SNE` has no train→test transform, so it is **visualization only** (in
`plot_reductions.py`), never in the probe.

## Files

- `reduce_kernels.py` — reducers + per-fold reduce/normalize + the reduced balanced-MC
  driver (reuses the P8/P9 `eval_fold`s).
- `run_acts.py` — cluster runner: per method, sweep {raw,pca,pls,umap}×{none,zscore,l2}
  at the P9-best layer, `maximal`+`engtier0` cleanings.
- `run_tfidf_local.py` — TF-IDF floor (char 2-5 → SVD-512), CPU, runnable locally.
- `plot_reductions.py` — 3D scatter of PCA/PLS/UMAP/t-SNE × {year,ruler} × {raw,z-score}
  (exploratory whole-data viz; motivates the probe).
- `aggregate_p10.py` — per method×cleaning table of gkpls per reducer×norm, Δ vs raw.

## Run

```bash
# activations (cluster; installs umap-learn in-job)
sbatch v_1/src/stress_tests/p10_reduce_kernels/sbatch/P10_run.sbatch      # array 0-8
sbatch v_1/src/stress_tests/p10_reduce_kernels/sbatch/P10_plots.sbatch    # figures
# then
python v_1/src/stress_tests/p10_reduce_kernels/aggregate_p10.py
# TF-IDF floor (local, no GPU):
python v_1/src/stress_tests/p10_reduce_kernels/run_tfidf_local.py
```

## Reading it

`RESULTS_p10.md` shows, per arm, gkpls Spearman for every reducer×norm with **Δ vs the
raw anchor**. Reduction *helps* only if some reduce+norm beats `raw/none` by more than
the draw-to-draw std. Caveats: reduction is fit per-fold (correct), but with the
year≈ruler-identity structure (see `p9`/`akkadian`) a gain in Spearman can still be
better *ruler separation* rather than a genuine timeline — cross-check against the
leave-one-ruler-out result in `world_models/akkadian`. The dial's `pred@λ=0` is the
supervised readout; `align1@λ=1` the unsupervised-geometry one.
