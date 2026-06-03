# Fig-1 follow-ups (advisor review, 03.06.26)

Two analyses requested when we reviewed Fig 1 (the supervised year-signal panel).
Both run on the **same 4 best-from-Fig-1A models**: `mlm`, `tfidf`,
`thalesian_cunei400m`, `qwen3_32b`.

Their best **balanced** layers (from `T1_year_pls.csv`, year=raw, by Spearman):

| model               | layer | cleaning | pool | balanced Sp |
|---------------------|-------|----------|------|-------------|
| mlm                 | L01   | tier0    | mean | 0.424       |
| tfidf               | L00   | tier0    | na   | 0.407       |
| thalesian_cunei400m | L12   | tier0    | mean | 0.411       |
| qwen3_32b           | L09   | tier0    | mean | 0.399       |

---

## Task 4 — PLS components tradeoff (`pls_ksweep/`)

**Question (Barak):** PLS beat Ridge in Fig-1A. Ridge uses *all* activation
columns; PLS uses only `k` directions. How many components actually help, and
where does adding more stop improving (or start hurting)?

**Design:** balanced regime (200 MC draws), x-axis = `k` ∈
`{1,2,3,5,8,16,32,64,128}`, y-axis = year Spearman (mean ± SD over draws), one
curve per model at its best layer above. Ridge drawn as a horizontal reference
(from `T2_year_ridge.csv`).

**Why this k ceiling:** balanced draws = 8 rulers × 21 frags = 168, and 5-fold
CV trains on ~134 of them. PLS components are capped at ~min(n_train, n_dim), so
the real ceiling is ~130 fragments — *not* the activation dimension. 64 is
already ~half the ceiling; 128 ≈ the wall. To push k into the hundreds we'd need
the imbalanced regime (1,193 frags) — deliberately not done here.

- Cluster: `pls_ksweep/sbatch/pls_ksweep.sbatch` (CPU; reuses `run_mc_probes.py`
  with the new `--pls-k` flag, isolated output dir + method-tag so it never
  collides with the existing `mc_balanced` results).
- Local: `pls_ksweep/aggregate_and_plot.py` (reads per-draw JSONs → curve).

## Task 3 — error-overlap / intersection (`error_overlap/`)

**Question (Gabi):** do the 4 models get the *same* fragments right and wrong
(→ they all read one shared surface signal, supports "dating is shallow"), or
different ones (→ something model-specific is happening)?

**Design + a flag to confirm:** uses the **imbalanced** full corpus with
GroupKFold-by-ruler, because that is the only regime where *every* fragment
receives exactly one out-of-fold prediction (balanced only ever samples the 8
well-attested rulers, so most fragments would never be predicted). This is an
error-analysis diagnostic, not a headline metric, so the imbalanced regime is
fine here — but flag for Yarin to veto.

- Cluster: `error_overlap/sbatch/error_overlap.sbatch` →
  `dump_oof_predictions.py` writes one `{fragment_id, ruler, year_true,
  year_pred}` record per fragment per model.
- Local: `error_overlap/analyze_overlap.py` (per-fragment abs-error, error
  correlation across models, "within ±100 yr" agreement, Venn) **plus a
  per-metadata-label breakdown**: one fraction-correct heatmap (groups × models)
  for each label, so you can see whether the shared errors cluster on any label
  (e.g. all models fail on the same period). Labels: `ruler`, `period`,
  `provenance`, `domain`, `sub_genre` (object type — cylinder/brick/wall-slab).
  `corpus`/`word_language`/`genre` are single-valued and `sub_provenance` is
  all-null in ORCC, so they are excluded. Tune with `--min-n` (min fragments per
  group) and `--top-k` (max groups).

---

### Cluster conventions
conda env `thesis`, repo at `~/projects/HUJI-THESIS--YARIN`, balanced draws at
`v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/`.
Activation tensors live only on the cluster (`orcc__embed/activations/`), so both
cluster scripts are CPU probing jobs — no GPU, no new extraction.
