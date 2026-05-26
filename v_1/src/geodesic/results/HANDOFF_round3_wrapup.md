# Handoff — finish & wrap up Round 3 (ORCC dating section)

You are taking over the **Round 3** results section of Yarin's Akkadian-cuneiform
interpretability thesis. The science is done; your job is to **fill the remaining empty cells**
so every model has results in every core readout, **run three pre-agreed confound controls**, and
**regenerate the result artifacts** so the section can be written up. Do not re-run anything that
already has results.

## Hard constraint (do not violate)
**You cannot run Python directly on the HPC cluster — only `sbatch` scripts that call Python.**
Local laptop Python is fine (TF-IDF, metadata baselines, anything CPU-trivial on the parquet).
Anything that needs model **activations** must go through an `sbatch` job. Yarin runs the jobs
himself in his cluster terminal and reports job IDs back — you generate the scripts and the exact
commands. (If a `cluster-job-runner` agent is available, use it for sbatch generation.)

## Where everything is
- Repo root: `/Users/yarin.b/git/lititure-review` (cluster path: `~/projects/HUJI-THESIS--YARIN`).
- Per-experiment tables (regenerate after any new result): `v_1/src/geodesic/results/tables/`
  via `python v_1/src/linear_probing/build_experiment_tables.py`.
- Balanced-vs-imbalanced scoreboard: `python v_1/src/linear_probing/build_balance_scoreboard.py`.
- Narrative docs to update at the end: `v_1/src/geodesic/results/RESULTS_BY_TEST.md`,
  `EXPERIMENTS_SUMMARY.md`, `orcc_round3_REPORT.md`.
- Balanced-MC runner: `v_1/src/linear_probing/round2_phase0/run_mc_probes.py`
  (joblib `--n-jobs`, `--draws-range`, self-healing JSON resume).
- Balanced-MC sbatch fan-out: `v_1/src/linear_probing/sbatch/orcc/{mc_chunk,mc_aggregate,submit_mc_fanout,submit_mc_backfill}.sh`.
- Full-set probes (PLS/Ridge): live under `v_1/src/linear_probing/results/orcc__probe_pls/`
  and `orcc__probe_cls_numeric/`; the producing scripts are in `v_1/src/linear_probing/`
  (find the PLS and cls_numeric probe entrypoints; they read activations from
  `v_1/src/linear_probing/results/orcc__embed/activations/`).
- Name masking (canonical): `v_1/src/linear_probing/name_masking.py`
  (masks `m-`/`f-` + theophoric `d-<god>-<predicate>`; keeps bare gods). Corpus has
  `text_{tier0,maximal}_masked` columns already.
- Corpus: `v_1/data/evaluation/corpora/orcc_corpus.parquet`
  (cols: ruler, period, provenance, sub_provenance, genre, year, word_count, text_tier0,
  text_maximal, + *_masked). **genre is useless (1 value); no archive/dynasty columns.**

## TASK A — fill the missing result cells
Confirm against the CSVs after each addition. Exact gaps (from `tables/`):

**A1 — Year regression, Ridge full-set (`T2_year_ridge.csv`, regime=fullset).**
Present only for qwen3_1b7/8b/32b. **Missing:** `thalesian_cunei400m`, `thalesian_akk300m`,
`mlm`, `qwen`, `tfidf`, `random`. Run the cls_numeric (Ridge) probe on the **full set** for
these — activations already extracted, so this is a probe re-run (sbatch if it reads neural
activations; `tfidf`/`random` may be local). Output to `orcc__probe_cls_numeric/`.

**A2 — Year regression, Ridge balanced-MC (`T2_year_ridge.csv`, regime=balanced).**
Present for mlm/tfidf/qwen/qwen3_*. **Missing:** `thalesian_cunei400m`, `thalesian_akk300m`,
`random`. Use `submit_mc_backfill.sh` pattern with `PROBES=<model>_cls_numeric` (see its header).

**A3 — Year regression, PLS (`T1_year_pls.csv`).**
**Missing fullset:** `tfidf` (PLS-year was never computed for tfidf — only ruler). 
**Missing balanced:** `random` (its MC `results` came back empty — known activation-path issue;
3 fix options recorded in `project_round3_geodesic.md`: symlink, re-extract ~30min cluster, or
drop). Decide with Yarin; if dropping, mark `random` explicitly "N/A — control" in the tables.

**A4 — Ruler classification balanced-MC (`T3_ruler_classification.csv`).**
**Missing mc:** `random`. Same root cause as A3; resolve together.

After A1–A4: rerun both builder scripts, confirm no remaining blank core cells (or explicit N/A).

## TASK B — three confound controls (pre-agreed with Yarin)
**B1 — Metadata-only year baseline (the floor).** Predict year from metadata alone
(ruler + provenance + period; **NOT** genre — 1 value) with the same Ridge GroupKFold-by-ruler
and the same balanced MC draws. This is the floor every embedding must beat. Local, CPU-trivial
(one-hot the metadata columns of the parquet, reuse `pls_utils.fit_ridge_year_groupkfold` and the
balanced draws in `results/orcc_round2_phase0/balanced_subset/`). New CSV `T8_metadata_baseline.csv` + MD.

**B2 — Leave-one-provenance-out (confound generalization).** Re-run the LORO machinery but group
by **provenance** instead of ruler, on the geodesic best configs (qwen L1, thalesian L6/L7).
Tests whether the temporal manifold generalizes across find-sites, not just rulers. Add rows to
`T5_loro.csv` (or a sibling `T5b_lopo.csv`) with a `group=provenance` column. (Genre unusable.)

**B3 — Same-ruler nearest-neighbor audit (ruler-only, per Yarin's scope).** For the best geodesic
config, among fragments of the **same ruler**, check whether nearest neighbors in the manifold are
still closer in year than chance — i.e. is there sub-ruler temporal ordering, or is the signal
entirely between-ruler? Report neighbor purity within-ruler vs a within-ruler shuffled null. New
CSV `T9_sameruler_nn.csv` + MD. (Do **not** do archive/genre variants — Yarin scoped this to ruler.)

## Gotchas (learned this round — do not relearn the hard way)
- **GroupKFold by ruler** on the imbalanced full set produces degenerate folds (a held-out ruler
  often spans one date → Spearman NaN, `n_valid_folds` < 5). That's why mlm/qwen/random full-set
  PLS-year is ~0/negative. Don't "fix" it — report it; the balanced number is the real one.
- **Balanced ruler Macro-F1 is NOT comparable to imbalanced** (8 classes vs 11–41). Never frame
  balancing as "helping" classification.
- **OMP_NUM_THREADS=1** in every sbatch (joblib threading backend oversubscribes otherwise).
- **Only `mc_aggregate.sh` commits** (it's the sole committer to avoid git races); chunk jobs
  write draws and do not commit. The runner is self-healing (validates draw JSON on resume; skips
  corrupt files) — killed jobs leave truncated JSONs, that's expected.
- **`git pull --rebase origin main`** before pushing — the cluster commits results remotely.
- Name masking keeps **bare gods** intentionally (a deity invoked is a period signal, not a name).
- Phase-D `archive` coloring is degenerate (single-valued) — ignore those 3 PNGs.

## Success criteria (the section is "wrapped up" when)
1. `T1`/`T2`/`T3` have a result (or explicit, justified `N/A`) for **every** model in both regimes.
2. B1/B2/B3 controls produce committed CSVs + MD and a one-line verdict each.
3. Both builder scripts rerun clean; `RESULTS_BY_TEST.md` + `EXPERIMENTS_SUMMARY.md` +
   `orcc_round3_REPORT.md` updated to cite the now-complete tables (kill any remaining
   "REPORT-only / no JSON" caveats).
4. A short `## Verdict: PASS|FAIL` appended to `orcc_round3_REPORT.md` stating whether the
   thesis claims (scale doesn't buy dating under balancing; dating is orthographic drift not name
   lookup; the only neural win is the geometric manifold) survive with the gaps filled.
5. Everything committed; ask Yarin before `git push`.

## Do NOT
- Re-run experiments that already have results (waste of cluster time).
- Run Python on the cluster login node (sbatch only).
- Add archive/genre control variants (genre is 1-value; archive doesn't exist; NN audit is
  ruler-only by Yarin's decision).
- Push without asking.
