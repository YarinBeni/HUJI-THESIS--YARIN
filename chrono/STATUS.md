# chrono/ build status — living board

Implements Part II of `phd_plan_chrono_jepa.md` (this branch). Updated by the
orchestrating agent after every wave; read this first when resuming.

## Wave B1 — parallel module build (agents A1–A7)
| Agent | Module | Plan tickets | Status |
|---|---|---|---|
| A1 | data contract + splits | P0.1, P0.2 | ⏳ launched |
| A2 | augmentation engine | P1.1–P1.5 | ⏳ launched |
| A3 | loss library | P2.1–P2.7 | ⏳ launched |
| A4 | models + trainer | P3.1, P3.2 | ⏳ launched |
| A5 | eval protocol | P0.4-eval, P3.4-core | ⏳ launched |
| A6 | cluster scripts + sbatch | P0.3, P0.4, C-jobs | ⏳ launched |
| A7 | P1.0 philology survey | P1.0 (new) | ⏳ launched |

## After-wave gates (orchestrator)
- [ ] full pytest green locally (CPU)
- [ ] local end-to-end smoke: build corpus→splits→views→tfidf features→train_cjb 30 epochs→battery
- [ ] adversarial review wave (math/leakage/determinism) + fixes
- [ ] push to yarin-sandbox; hand sbatch order to Yarin:
      C0_tests → C1_extract → C2_baseline_gate (P0.4) → C3_emin (E-MIN, gate G2)
