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
| W2-1 | high | split-conformal assumed DOC exchangeability; with t block-constant per ruler, nominal 80% delivered 66.5% and 6/40 rulers were covered 0% | `fit(..., groups=)` splits BY RULER and banks ONE residual per calibration ruler (block conformal); `coverage_by_group` reports per-ruler coverage; `n_effective` counts BLOCKS (20), not documents (594). Real corpus: ruler-coverage .54 → **.88**, rulers at 0% 12 → 3. Honest price: mean interval 98 → 527 yr |
| W2-2 | med | margins ran to 9.99 while the score scale is floored at std 1.0 → 36% of pairs permanently saturated, force allocated by reign distance not by violation | `_margin` = 2·tanh(gap/std(t)), order-preserving and bounded; `t_std` is now an explicit CORPUS constant (fold std ranged 101–127) |
| W2-3 | high | `_sandbox.sh` never checked the branch out: on a cluster clone sitting on main it rebased **main** and would have pushed main's history into yarin-sandbox; failures were silent | explicit `checkout -B`, HEAD verified after sync, non-zero return on failure; `commit_push_sandbox` returns 1 when the push never lands |
| W2-4 | high | every `#SBATCH --output=chrono/sbatch/logs/...` pointed at a directory that cannot exist in a clone (`**/logs/` is gitignored) — Slurm opens that file BEFORE the script's mkdir, so all four jobs would have died instantly | tracked `chrono/sbatch/logs/.gitkeep` + .gitignore negation |
| W2-5 | med | the 5-task C3 array read-modify-writes ONE results.parquet with no lock → silently dropped rows | `append_results` now holds an flock across read-concat-write and lands via `os.replace`; verified with 8 concurrent writers, 8/8 rows kept |
| W2-6 | high | **the P3.4 robustness battery was not wired at all**: the only score writer hard-coded `condition='orig'`, so the condition×split grid — the plan's headline deliverable — could never have more than one row | `_condition_scores` scores EVERY augmentation chain; the grid is now computable (8 conditions × 3 splits, table in this file's history) |
| W2-7 | high | per-fold scores were un-poolable (different affine scales, no test marker) so the pooled-OOF gate metric was unassemblable | scores gain `is_test` and fold-local `s_rank` |
| W2-8 | high | `run_baseline_gate` reported gkf as a mean of per-fold rho (forbidden by F2) and its verdict was max-over-52-cells vs a fixed threshold — a simulation put max-of-52 of pure ruler-block noise at ρ 0.72, so **that gate could not fail** | verdict is read off an A-PRIORI cell (PLS/L11/mean); the best cell is printed as "SELECTION-INFLATED, context only"; pooled read-out + block null wired in |
| W2-9 | high | EmbStore resume keyed on id membership only — `text_sha` written but never verified, so a views rebuild under stable view_ids left the cache stale and C1 skipped extraction | `has(..., texts=)` compares sha, new `stale()`, `get(..., texts=)` raises; the C1 resume check now passes chunk texts |
| W2-10 | med | view-pair sampler drew byte-identical text in 23–39% of steps and DIFFERENT LANGUAGES in ~50% — half the Barlow pressure was akk-vs-eng invariance | `_pair_rows`: one language per step, identical-text draws rejected and resampled; realised rate now reported as `identical_view_rate` in every run. **23–39% → 1.4%.** Root cause found on the way: `mask_ruler` alone is a no-op for the 54% of glosses that never name the ruler, so both configs' branch-A menus now use chains that always alter text |
| W2-11 | low | batch of 1 made `bt_loss`/`variance_loss` silently dead (returned D with zero gradient) | `bt_loss` raises on batch < 2 |
| W2-12 | low | the loss-library fallback could silently swap the real losses for stubs on the cluster | fallback is opt-in via `CHRONO_ALLOW_FALLBACK_LOSSES=1`, else hard ImportError |
| W2-13 | med | nested config keys never validated; `emin_thalesian.yaml` asked for view seeds [0,1,2] against a 2-seed artifact | trainer warns loudly for any requested chain/seed absent from views.parquet; the cluster config's seeds corrected to [0,1] |
| W2-T | med | **mutation testing of our own suite**: deleting split-conformal, single-centering HSIC, var-instead-of-std, ×100 off-diag weight and `<=` disjointness ALL passed green | four new mutation-killing tests (hand-computed HSIC tr(KHLH) reference, numeric Barlow pin, std hinge value, touching-interval fixture) + the block-conformal test. **Re-audited: all five mutations now CAUGHT** |
| C1-1 | high | `--layers 0-12` and the a-priori verdict cell L11 assumed a 12-block encoder; Thalesian/AKK_300m has 8 blocks + embeddings = 9 hidden states (0..8). The C1 smoke run failed loudly on the bounds check; had the range been valid-but-wrong the gate verdict would have fallen through to the selection-inflated best cell | layer grid is 0-8 everywhere (extract, gate, C1/C2 sbatch, INTERFACES); `APRIORI_LAYER = 8` (top encoder block); new test `test_apriori_cell_is_a_real_layer` |
| C1-2 | high | C1 (job 32500) died in 15s on the SAME layer bug that C1-1 fixed: **slurm copies the batch script at submit time**, so the in-job `sync_sandbox` updated the .py files but the frozen sbatch still passed `--layers 0-12`. Any constant living in an sbatch is immune to the sandbox sync | `extract_embeddings --layers all` asks the encoder how many hidden states it returns (default); C1 no longer names a layer count. The gate skips (layer, site) cells absent from the store but **refuses to print a verdict** when the a-priori cell was requested and is missing, instead of falling through to the best cell |
| C2-1 | high | C2 (job 32723) died in PLS/gesdd on L0/last: the embedding-layer vector of the final token is the same `</s>` for every text, so the cell is one row repeated 1,187× (ridge mc −0.08±0.000) | cells with zero feature variance are skipped and named; a PLS fold that still fails numerically returns NaN instead of killing the sweep |
| C2-2 | high | `sync_sandbox` used `checkout -B <branch> FETCH_HEAD`, which **resets** the branch to the remote and silently drops commits a previous job made but could not push (the token had expired) — both C1 meta commits vanished | checkout the branch as-is (create only if absent) then rebase local commits onto FETCH_HEAD; C2 re-lands the C1 extract meta |
| C2-3 | result | **C2 gate ran (job 33316).** Ridge/mean ρ .27–.30 akk, .39–.42 eng, placebo and ruler-block null ≈ 0 everywhere → pipeline reproduces the M.Sc. signal. A-priori cell PLS-k2/L8/mean = .126 akk, below the .15 investigate line | the M.Sc. probe row-L2-normalised before PLS, C2 did not; `--row-l2` added, like-for-like rerun queued via the runner (003). Cell not re-picked. See `docs/gate_reference.md` |

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
- [x] **all wave-2 findings closed** (W2-1, W2-9, W2-10, W2-13, W2-T).
      80 tests green; the mutation audit passes.
- [ ] hand sbatch order to Yarin:
      C0_tests → C1_extract → C2_baseline_gate (P0.4) → C3_emin (E-MIN, gate G2)
