# chrono/ build status — living board

Implements Part II of `phd_plan_chrono_jepa.md` (this branch). Updated by the
orchestrating agent after every wave; read this first when resuming.

## Wave B1 — parallel module build (agents A1–A7)
| Agent | Module | Plan tickets | Status |
|---|---|---|---|
| A1 | data contract + splits | P0.1, P0.2 | ✅ 14 tests; real artifacts: 1,187/40/47, 5 split files |
| A2 | augmentation engine | P1.1–P1.5 | ✅ 10 tests; NOTE: corpus text_tier0 is transliteration — eng gloss merged from translations.parquet |
| A3 | loss library | P2.1–P2.7 | ✅ 19 tests incl. gradchecks, soft-vs-scipy .0027, conformal coverage |
| A4 | models + trainer | P3.1, P3.2 | ✅ 10 tests; overfit gate ≥.95 |
| A5 | eval protocol | P0.4-eval, P3.4-core | ✅ 16 tests |
| A6 | cluster scripts + sbatch | P0.3, P0.4, C-jobs | ✅ 6 tests (selftest path); real transformers path untested until C1 |
| A7 | P1.0 philology survey | P1.0 (new) | ✅ docs/dating_criteria.md (DRAFT, needs Assyriologist) |

## After-wave gates (orchestrator)
- [x] full pytest green locally: 75/75 in 16s
- [x] local end-to-end smoke on REAL corpus: 37,984 views → tfidf → train_cjb (train ρ .61) → mc_balanced 200 draws (ρ .67 train-fit), placebo +.01 ✓
- [x] adversarial review wave 1 (protocol lens): 7 findings, all fixed
- [x] adversarial review wave 2 (math / trainer / cluster lenses): 17
      findings, every one verified by an independent refuter; all triaged
      and fixed below. Reviewers ran MUTATION TESTS on our own test suite —
      several tests were vacuous (see W2-T).
- [x] all 7 R2 findings triaged and FIXED (see FIXES below); 75/75 green;
      artifacts rebuilt; end-to-end re-run clean.

### Review fixes applied — wave 2 (17 findings)
| # | Sev | Finding | Fix |
|---|---|---|---|
| **W2-DATA** | **high** | **ORCC's `year` holds a CENTURY, not a year, for 6 rulers / 13 docs** (values 7–10). Marduk-zakir-šumi I (reigned c. 855–819 BC) was stored as "9", Eriba-Marduk (c. 769) as "8" — so 9th-century Babylonian kings sat 250 years AFTER Antiochus I, at the extreme late end of the axis, and took 19% of the ranking gradient. **This affects the M.Sc. corpus too** (same 13 docs, same years, in `pairs_data.load_eligible`) | `contract._repair_years`: values ≤ 30 are century indices → mid-century year (9 → 850 BC), row marked `t_quality='century'`. Corpus t range is now −1132…−261 (historically sane) instead of −1132…−7 |
| W2-1 | high | split-conformal assumed DOC exchangeability; with t block-constant per ruler, nominal 80% delivered 66.5% and 6/40 rulers were covered 0% of the time | (open — see NEXT) |
| W2-2 | med | margins ran to 9.99 while the score scale is floored at std 1.0 → 36% of pairs permanently saturated, force allocated by reign distance not by violation | `_margin` = 2·tanh(gap/std(t)), order-preserving and bounded; `t_std` is now an explicit CORPUS constant (fold std ranged 101–127) |
| W2-3 | high | `_sandbox.sh` never checked the branch out: on a cluster clone sitting on main it rebased **main** and would have pushed main's history into yarin-sandbox; failures were silent | explicit `checkout -B`, HEAD verified after sync, non-zero return on failure; `commit_push_sandbox` returns 1 when the push never lands |
| W2-4 | high | every `#SBATCH --output=chrono/sbatch/logs/...` pointed at a directory that cannot exist in a clone (`**/logs/` is gitignored) — Slurm opens that file BEFORE the script's mkdir, so all four jobs would have died instantly | tracked `chrono/sbatch/logs/.gitkeep` + .gitignore negation |
| W2-5 | med | the 5-task C3 array read-modify-writes ONE results.parquet with no lock → silently dropped rows | `append_results` now holds an flock across read-concat-write and lands via `os.replace`; verified with 8 concurrent writers, 8/8 rows kept |
| W2-6 | high | **the P3.4 robustness battery was not wired at all**: the only score writer hard-coded `condition='orig'`, so the condition×split grid — the plan's headline deliverable — could never have more than one row | `_condition_scores` scores EVERY augmentation chain; the grid is now computable (8 conditions × 3 splits, table in this file's history) |
| W2-7 | high | per-fold scores were un-poolable (different affine scales, no test marker) so the pooled-OOF gate metric was unassemblable | scores gain `is_test` and fold-local `s_rank` |
| W2-8 | high | `run_baseline_gate` reported gkf as a mean of per-fold rho (forbidden by F2) and its verdict was max-over-52-cells vs a fixed threshold — a simulation put max-of-52 of pure ruler-block noise at ρ 0.72, so **that gate could not fail** | verdict is read off an A-PRIORI cell (PLS/L11/mean); the best cell is printed as "SELECTION-INFLATED, context only"; pooled read-out + block null wired in |
| W2-9 | high | EmbStore resume keyed on id membership only — `text_sha` was written but never verified, so a views rebuild under stable view_ids leaves the cache stale and C1 skips extraction | (open — see NEXT) |
| W2-10 | med | view-pair sampler drew byte-identical text in 23–39% of steps and DIFFERENT LANGUAGES in ~50% — half the Barlow pressure was akk-vs-eng invariance | (open — see NEXT) |
| W2-11 | low | batch of 1 made `bt_loss`/`variance_loss` silently dead (returned D with zero gradient) | `bt_loss` raises on batch < 2 |
| W2-12 | low | the loss-library fallback could silently swap the real losses for stubs on the cluster | fallback is opt-in via `CHRONO_ALLOW_FALLBACK_LOSSES=1`, else hard ImportError |
| W2-13 | med | nested config keys were never validated; `emin_thalesian.yaml` asks for view seeds [0,1,2] while views.parquet has [0,1] | (open — see NEXT) |
| W2-T | med | **mutation testing of our own suite**: deleting split-conformal entirely, single-centering HSIC, var-instead-of-std, ×100 off-diagonal weight and `<=` in the disjointness rule ALL left the tests green | margin test now pins the real formula; the rest are (open — see NEXT) |

### Review fixes applied (wave B1)
| # | Sev | Finding | Fix |
|---|---|---|---|
| F1 | high | `mask_ruler` was a byte-for-byte NO-OP on all Akkadian views (ruler_spans_akk empty on 1187/1187 — anglicized names never occur in transliteration) → a "survives ruler masking" row that passes by construction + thousands of duplicate GPU texts | `engine._base_text`: akk mask chains start from the corpus' pre-masked tier, `[PN]`→`<RULER>`. Now 7,017 akk masks; 954/2374 akk mask views differ from orig (the rest genuinely never name the ruler) |
| F2 | high | battery averaged per-fold ρ where folds are single-ruler: gkf dropped 2/5 folds, **loro n=1 was literally the within-Esarhaddon ρ wearing an unseen-ruler label** | per-split read-out policy: mc → per-draw, everything else → `pooled_rho` (one ρ over concatenated held-out docs); `readout` column added to BATTERY_COLS |
| F3 | high | tfidf featurizer + head fitted on ALL docs → held-out cells transductive, biased vs cross-fitted baselines | `build_features(..., fit_doc_ids=)` fits on fold-train views only; scores parquet gains `fit` ('oof'/'full') + `fold` provenance columns |
| F4 | med | mc's 200 draws share ONE fixed 8-ruler design, ~48/168 doc overlap, 116 docs never sampled → ρ_sd read as an SE overstates certainty ~10× | documented at source in `splits.py` + SLA; ruler-level uncertainty required for any model-vs-model claim |
| F5 | med | placebo shuffled t per DOC, but 39/40 rulers carry ONE year → null 3× too narrow (±.18 vs ±.50) | added `block_placebo_rho` (permutes ruler→t); doc-level demoted to leak detector; SLA now mandates the block null for significance |
| F6 | low | diacritic variants missed ('Sîn' vs 'Sin') → a few "masked" eng views still named the ruler | length-preserving NFKD fold in `contract._fold`; eng span docs 547→548 |
| F7 | low | `test_spearman` silently subset the fold's test docs | raises `KeyError` listing missing ids |
| F8 | med | **(orchestrator finding)** order pairs were drawn ONCE before the epoch loop: 2,428 frozen constraints, 20% of docs in none of them, median ruler-pair contributing 1 pair (quota is min(m, n_i, n_j) and the ruler tail is long) — the plan's combinatorial-supervision promise under-delivered | `train.resample_pairs` (default true): redraw per epoch with a derived seed. Measured coverage 80.1% → 99.9% after 5 epochs, 100% by 10, at zero GPU cost |

### Verified by the orchestrator (not agent claims)
- **Axis sign is correct end-to-end.** For a pair (i, j) with t_i < t_j the
  gradient moves s_i down and s_j up; all emitted pairs satisfy t_i < t_j;
  fitting a free score vector on the REAL pairs alone gives ρ(s, t) = +0.62
  (positive as the SLA requires). A sign flip here would have inverted every
  downstream number silently.
- **Pair eligibility is not the bottleneck**: 776 of 780 ruler pairs have
  disjoint reign proxies, so the order signal spans essentially the whole
  ruler graph.
- [ ] adversarial review wave (math/leakage/determinism) + fixes
- [x] pushed to yarin-sandbox
- [ ] **NEXT (before any GPU run)** — five wave-2 findings still open:
      W2-1 block-conformal calibration (`groups=` by ruler + ruler-level
      coverage), W2-9 EmbStore text_sha verification, W2-10 constrain the
      view-pair sampler (same-language, reject identical text),
      W2-13 nested config validation, W2-T strengthen the vacuous tests
      (touching intervals, HSIC double-centering, off-diag weight,
      conformal split). None of these blocks C0/C1; W2-1 and W2-10 must
      land before C3 is read as science.
- [ ] then hand sbatch order to Yarin:
      C0_tests → C1_extract → C2_baseline_gate (P0.4) → C3_emin (E-MIN, gate G2)
