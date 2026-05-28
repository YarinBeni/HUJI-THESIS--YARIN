# Round 3 — per-experiment tables

Each test has a `.md` (what it is, data/split, how to read the CSV) and a `.csv` (every config x
every metric, straight from the result JSONs). Regenerate with
`python v_1/src/linear_probing/build_experiment_tables.py`.

| Test | MD | CSV(s) |
|---|---|---|
| 1 Year regression — PLS | T1_year_pls.md | T1_year_pls.csv |
| 2 Year regression — Ridge | T2_year_ridge.md | T2_year_ridge.csv |
| 3 Ruler classification — CLS (logistic) | T3_ruler_classification.md | T3_ruler_classification.csv |
| 3b Ruler classification — PLS-DA | T3b_ruler_plsda.md | T3b_ruler_plsda.csv |
| 4 Geodesic / Isomap manifold | T4_geodesic.md | T4_geodesic.csv |
| 5 LORO leave-one-ruler-out | T5_loro.md | T5_loro.csv, T5_loro_per_ruler.csv |
| 6 Phase D visualization | T6_phase_d.md | T6_phase_d.csv |
| 7 TF-IDF name-masking control | T7_name_masking.md | T7_name_masking.csv |
| 9 Direct elicitation (kp0/kp1/kp2) | T9_elicitation.md | T9_elicitation.csv |
| 10 Prompted reprobe (pv0-pv3) | T10_prompt_reprobe.md | T10_prompt_reprobe.csv |

See also `../RESULTS_BY_TEST.md` (narrative, best-config tables) and
`../EXPERIMENTS_SUMMARY.md` (advisor-facing, embedded plots).
