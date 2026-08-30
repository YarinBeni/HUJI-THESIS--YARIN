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
- [x] adversarial review wave: 1 of 4 lenses returned before the session
      agent limit hit (R2:protocol, 7 evidence-backed findings); the other
      three lenses (math / trainer / cluster) NEVER RAN — re-run them before
      trusting chrono/losses, the trainer internals or the sbatch suite.
- [x] all 7 R2 findings triaged and FIXED (see FIXES below); 75/75 green;
      artifacts rebuilt; end-to-end re-run clean.

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
- [ ] adversarial review wave (math/leakage/determinism) + fixes
- [x] pushed to yarin-sandbox
- [ ] NEXT: re-run the 3 missing review lenses (math/trainer/cluster), then hand sbatch order to Yarin:
      C0_tests → C1_extract → C2_baseline_gate (P0.4) → C3_emin (E-MIN, gate G2)
