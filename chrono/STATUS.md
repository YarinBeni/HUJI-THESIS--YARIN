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
- [ ] KNOWN ISSUES for review wave: (i) gkf_ruler folds 0/1 are single-ruler (mega-rulers fill a fold; per-fold ρ undefined) — needs snake-assignment or pooled-OOF read-out; (ii) LORO within-ruler ρ undefined by construction — battery needs pooled-across-folds read-out for gkf/loro/held-out; (iii) trainer tfidf fitted on ALL view texts = transductive for fold comparisons (fine for smoke, leakage for E-MIN vs per-fold baselines)
- [ ] adversarial review wave (math/leakage/determinism) + fixes
- [ ] push to yarin-sandbox; hand sbatch order to Yarin:
      C0_tests → C1_extract → C2_baseline_gate (P0.4) → C3_emin (E-MIN, gate G2)
