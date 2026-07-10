# Stress-Tests — Session Handoff / Context

> Read this first. A fresh agent should be able to carry on from here.
> Branch: `claude/stress-test-timeline-analysis-9sh2vs`. **The cluster runs from
> `main`** (every sbatch does `git pull --rebase origin main` … `git push origin
> HEAD:main`). This agent's environment now BLOCKS direct pushes to `main`
> (auto-mode classifier), so new code is pushed to the feature branch and must be
> fast-forwarded onto `main` on the cluster login node (not blocked there):
> `git checkout main && git merge --ff-only origin/claude/stress-test-timeline-analysis-9sh2vs && git push origin main`.
> The feature branch = `origin/main` + the new commits, so it's always a clean FF.
> All work under `v_1/src/stress_tests/`. The user runs every cluster job by
> pasting `sbatch`; the agent NEVER SSHes.

---

## Session update (2026-07-01) — decisive control landed + imbalance diagnosis + sbatch hardening
- **DECISIVE control resolved (§7.1):** `random Qwen3-8B king_last = 0.643 PLS / 0.495 Ridge`
  (see `results/RESULTS_stress_tests.md`). It is **as high as the pretrained models**
  (8B 0.480, 32B/gpt-oss 0.645, MLM 0.704). So the high king_last is **NOT a learned
  chronology** — it is **name-token identity readout**: the king-name span is a near
  one-hot ruler id and `year` is a function of ruler, so any pooling that reads the name
  token recovers the date, even with random weights. The §1 claim's "date is linearly
  recoverable at the king token" leg must be **reinterpreted** accordingly. The robust,
  surviving dissociation: models *know* dates behaviorally (T9) and the date is *trivially*
  readable from name identity, but it is **not diffused into a text-level geometry**
  (mean-pool ≈ random ≈ 0.35–0.40, flat across scale/objective/prompting/training).
- **WHY king_last is "so easy" — data-side diagnosis:** `eda/class_imbalance_analysis.py`
  → `results/eda/class_imbalance.md` + 4 PNGs. Balanced-MC uses **only 8 rulers × k=21**
  (k capped by the smallest class, Sîn-šarru-iškun = 21 frags; 33/41 rulers dropped).
  `year` is an **8-level step function of ruler**. King-name coverage is very uneven
  (Sennacherib 0.67 … **Nebuchadnezzar II 0.00**), so the king pool shrinks to
  **~62 frags/draw (~37%)** over **~7/8 groups** — with GroupKFold folds of 1–2 rulers
  (1–2 distinct years) Spearman is coarse, high, and high-variance (±0.3–0.4; the
  `ConstantInputWarning` in the logs). This is the sample-size story behind the inflated
  king_last / the strong random baseline.
- **sbatch hardening (fixes the log errors):** new `sbatch/_common.sh` (`sync_main`,
  `push_main`, `commit_push`) serializes git with `flock`, always rebases onto a single
  `FETCH_HEAD` with `--autostash`, and clears stale rebase/index locks → fixes the J8
  "unstaged changes" and J5b "Cannot rebase onto multiple branches" races. All 18 sbatch
  files migrated. **J3r is now a per-model array job** (`--array=0-3`, one model each) —
  the old serial job TIMED OUT at 3h; now each model gets its own wall clock. Log-name
  mismatches fixed (J3r/J7/J4/J4b `--output` now match the script stem).
- **Rerun order:** `sbatch J3a_t10_qwen3.sbatch` (GPU, re-extract 8b/32b prompted acts if
  missing) → then `sbatch J3r_t10_reprobe_mc.sbatch` (CPU array; gpt-oss task skips cleanly
  if no acts) → `sbatch J11_aggregate.sbatch`. Land code on `main` first via the header FF.

### `maximal-with-kings` config (NEW — fairer 3-site comparison)
Motivation: king_last's high score is name-token identity (random matches it), and the old
setup compared `mean` (maximal) vs `king_*` (tier0) — not apples-to-apples. This config puts
**all 3 sites (mean / king_last / king_mean) on ONE cleaning** and rebalances so the random
baseline is a real control.
- **Cleaning** `shared/cleaning.py::clean_maximal_keepking`: full `maximal` on the context but
  the commissioning ruler's name span is frozen (name-aware truncation keeps it), so king
  coverage = tier0 ceiling while context is truly maximal. Activation tag = `_maxking_*`.
- **Subset** `p1_gurnee_tegmark/build_maxking_subset.py` → `…/balanced_subset_maxking/`
  (committed): **5 rulers** (dropped Neb II / Tiglath-pileser III / Nabonidus, E[king-found/draw]<6),
  **k=9** (capped by Sîn-šarru-iškun's 9 king-found), draws from **king-found only** (identical
  fragment set for all 3 sites).
- **Probe** `shared/mc_maxking.py` + `p1_gurnee_tegmark/probe_maxking.py` → three analyses per
  site/layer: `year_group` (legacy GroupKFold Spearman — degenerate for a per-king-constant
  label, kept for continuity), `year_strat` (StratifiedKFold Spearman/MAE/**±10yr acc**),
  `ruler_clf` (**StratifiedKFold macro-F1 control** vs chance + shuffle). Best layer by ruler-F1.
- **Finding on `year`:** it is a single constant per king (nunique=1 except Esarhaddon), so
  year-probe ≡ ruler-id; adjacent king labels are 12–38 yr apart → **±50 too coarse, ±10 used**.
- **Jobs:** `J12_maxking_extract` (GPU array: qwen3×3 + thal×2 + umt5), `J12b`(gpt-oss gpu:4),
  `J12c`(random Qwen3-8B) → then `J13_maxking_probe` (CPU). EDA: `eda/class_imbalance_analysis.py`,
  `results/eda/fig_year_tolerance.png` (counts + tolerance bands). MLM maxking = TODO (needs a
  maxking variant of extract_mlm_king_acts.py).

---

## Gap-fix wave (2026-07-05) — submitted after the advisor-deck review
New jobs closing the review gaps (all code on `main` once the feature branch is FF'd):
- **J2c** — T9 kp1 rerun for gpt-oss @ `max_new_tokens=2048` (the J2b run used 256 and
  truncated the reasoning channel); re-runs parse/score + `eda/rescore_t9_kp1.py`.
- **J14** — P2 geography under **site-balanced MC**: 10 coordinate-merged sites × k=21 ×
  200 draws (`p2_godey_geography/build_site_balanced_subset.py`, draws committed in
  `balanced_subset_sites/`), GroupKFold-by-merged-site within each draw, tier0+maximal,
  PLS-k sweep + Ridge → `p2_godey_geography/results/mc/p2_geo_mc__*.json`.
- **J9b** — P7 v2: classification **and k-sparse year-regression** × 3 cleanings
  (tier0/maximal/maxking) → `p7_ksparse/results/v2/`; `plot_p7_curves.py` renders the
  3-panel per-k curves.
- **J3b/J3c/J3r** — T10 completed: gpt-oss on gpu:8 (J3b, array = cleaning), qwen ×
  {maximal, maxking} prompted variants (J3c), MC reprobe array = model × cleaning (J3r,
  `--array=3-11`; extractor gained `--cleaning`, reprobe gained `--tag`).
- **J15** — P3 v2 `timeline_p3_pls.py`: PLS(k=3) fit on the 153 anchors vs year →
  3-D anchor manifold → Isomap-1D geodesic timeline (`geo1`) + PLS-comp-1 (`pls1`) →
  texts projected via `pls.transform`, year by inverse-distance interpolation over the
  5 nearest anchors + ruler-F1 via nearest ruler-anchor; × 3 cleanings →
  `p3_matter_of_time/results/pls/`.
- **J16** — `eda/dump_gui_coords.py`: PCA/t-SNE 2-D coords of maxking-mean embeddings
  (+P3 anchors) → committed `v_1/src/viz/maxking_coords.json`; GUI merge happens
  off-cluster after it lands.
- Known caveat to carry: the year-anchor grid spans 7–1132 BCE and includes the bogus
  min-digit years (e.g. "year 7 BCE", "Unidentified NB Ruler"); kept identical to the
  original 153 for comparability — a filtered-anchor rerun is an open option.

## T11 — generated-answer dating (2026-07-10, Yarin's idea)
T10 only probed the ACTIVATIONS under dating prompts; T11 reads the model's actual
ANSWER: each fragment is shown to the chat LLM ("expert Assyriologist… respond with
JSON {\"year_bce\": int, \"basis\": …}"), the answer is parsed (thinking/harmony
channels stripped, JSON → range → "N BCE" → century → bare-int fallbacks; selftest in
`score_gen_dating.py --selftest`), and scored with the SAME 200 balanced 8×21 draws
as every probe → directly comparable Spearman. Also logs whether the answer text
names the true ruler (diacritic-folded) and conditions Spearman on that, separating
name-lookup dating from style dating. Interpretation grid: probe>answer = linearly
recoverable but not verbalized; answer>probe = knowledge a linear probe misses; both
≈ random = strengthens "no timeline".
- Code: `t11_gen_dating/{generate_dates.py, score_gen_dating.py}` (raw JSONLs
  gitignored; `results/t11_gen__{model}__{cleaning}.json` committed by the jobs).
- Inputs: tier0 / maximal / maxking / engtier0 (Thalesian English), ALL capped at
  300 words (T10 gpt-oss parity → identical inputs across models).
- Jobs: **J18a** (qwen3 ×4 cleanings, array 0-11, gpu:2), **J18b** (gpt-oss ×4,
  array 0-3, gpu:8, 768 new tokens). Greedy decoding; Qwen thinking disabled,
  gpt-oss reasoning_effort=low; resumable (appends to existing JSONL).

## P8 — supervision-dial spectral probe (2026-07-10, Yarin's kernel-probe idea)
Implements §4 of Yarin's June-2026 working note ("Supervised Nonlinear Dimension
Reduction for Manifold-Structured Chronology"): ONE generalized eigenproblem
`[(1-λ)HK_yH − λL]z = γDz` whose dial λ∈[0,1] interpolates pure manifold geometry
(λ=1 = Laplacian eigenmaps, embedding never sees the year) and pure supervised
kernel dependence (λ=0 = HSIC/supervised-PCA with an RBF year kernel). The λ-curve
answers "how much supervision is needed to align the embedding with chronology" in
one figure. Out-of-sample = LPP linear projection (train-fitted PCA→V), so held-out
rulers are leakage-free; protocol = the SAME 200 balanced draws × GroupKFold-by-ruler.
Readouts: `align1` = |Spearman(leading coord, y)| held-out; `pred` = ridge-on-Z₃.
Theory + references in `p8_lambda_probe/MATH_NOTES.md` (Gretton HSIC, Belkin–Niyogi,
He–Niyogi LPP, Barshan SPCA, Ky Fan). Selftest (`lambda_probe.py --selftest`):
y-on-dominant-axis ⇒ flat/high; y-on-transverse-axis ⇒ λ=1 end collapses (0.11 vs
0.97 at λ=0) — the dial demonstrably works.
- Code: `p8_lambda_probe/{lambda_probe.py, run_tfidf_local.py, run_acts.py,
  plot_lambda_curves.py}`; results → `p8_lambda_probe/results/p8_lambda__*.json`,
  figures `fig_p8_lambda__*.png`.
- TF-IDF arm runs LOCALLY (char_wb(2,5) + SVD-512, k∈{5,10,20}, d=3).
- **J20** sbatch: all 9 methods × {tier0, maximal} on stored mean acts (CPU, per-layer
  sweep, best layer surfaced at λ=1-align1 and λ=0-pred; resumable — skips done JSONs).
- Expected outcome given the nulls so far: "flat and low ≈ random" on Akkadian; the
  interesting cells are engtier0-style and whether ANY trained model separates from
  random at the λ=0 end. **TF-IDF result (local, 200 draws, committed): FLAT AND LOW
  at every k** — align1 0.32–0.35 (tier0) / 0.23–0.26 (maximal) across the whole dial,
  pred ≤ the P1b PLS baseline; supervision adds nothing. J20 now sweeps all 4 cleanings
  (tier0/maximal/maxking/engtier0; missing acts dirs skip cleanly).
- Formal correctness proofs (paper-style, for the thesis): `p8_lambda_probe/correctness.tex`
  (exact Ky Fan optimality; endpoints = LPP-eigenmaps / SPCA; HSIC-with-linear-kernel
  ⇔ mean-independence; leakage-freeness; Lipschitz/convex λ-path). Compiles with pdflatex.

## P9 — G-KPLS, the geodesic kernel PLS (2026-07-10, working note §2)
Kernel PLS on the geodesic Gram matrix K_G = −½·H·D_G²·H (Isomap = kPCA on K_G), the
one-stage supervised repair of "Isomap→PLS". Nyström out-of-sample: test points connect
to their k nearest TRAIN points and route through the FIXED train graph (note eq. 5) —
leakage-free LORO/GroupKFold. Three arms per fold: **gkpls** (a∈{1,2,3,5} best-a),
**rbfkpls** (Euclidean RBF KPLS — isolates geodesic vs kernel), **krr_geo** (kernel ridge
on K_G via TRUNCATED spectral inverse — the clipped MDS Gram has a large null space that
a dense solve amplifies catastrophically; isolates PLS vs kernel). Same balanced MC.
Selftest (widening spiral, isometric 25-D lift): gkpls 0.90 / krr 0.95 / rbf 1.00.
- Code: `p9_gkpls/{gkpls.py, run_acts.py}`; results → `p9_gkpls/results/p9_gkpls__*.json`.
- **J21** sbatch: 9 methods × 4 cleanings (tier0/maximal/maxking/engtier0), CPU,
  resumable. Interpretation grid is the note's §5 table (G-KPLS vs Isomap→PLS vs RBF).

## T11 partial results (10/16 cells, J18a jobs 12677_0-9)
`t11_gen_dating/results/t11_vs_probe.md` (rebuild: `python t11_gen_dating/build_t11_table.py`).
Headlines: **qwen3-8B on Akkadian answers 98.5% but is ANTI-correlated** (MC −0.27 on
tier0/maximal/maxking; MAE ~286yr) — hypothesis: "Babylon ⇒ Old-Babylonian/Hammurabi-era"
prior inverts the ranking (NB texts are the LATEST but get dated ~1750 BCE; verify from
raw JSONLs on the cluster). **On English translations 8B is genuinely good**: MC +0.52,
acc@50 0.65, ρ=0.876 when it names the true ruler (45% naming rate) — behavioral
counterpart of the translation-probe result. 1.7B declines 94–100%; 32B declines 87–98%
on Akkadian (answer-rate collapses with scale? gpt-oss will tell). Pending: 32B
maxking/engtier0, gpt-oss ×4 (12678), E5 (12679).

## E5 — word-shuffle control (2026-07-10)
Does word ORDER matter to the embedding year signal? Each fragment is word-capped
FIRST (both variants keep the exact same words), then shuffled (seed 42); the
shuf + unshuf twins are extracted with identical settings (immune to historical
extraction-setting mismatch) and probed with the standard balanced MC. If
delta(unshuf − shuf) ≈ 0, the probe reads bag-of-tokens — TF-IDF-like — not
composition; the unshuf rows also sanity-check against the historical P1 numbers.
- Code: `e5_shuffle/{extract_shuffled_acts.py, probe_e5_mc.py}` (reuses T11's
  `fragment_texts` so the texts match exactly). Acts dirs:
  `{method}_{shuf|unshuf}{cleaning}_mean` (npz gitignored).
- Scope: qwen3_8b (300 words / 2048 tokens) + thalesian_cunei400m (120 words /
  512-token umT5 window) × {tier0, maximal, maxking, engtier0}. Note maximal/
  maxking texts are ≤~32 words anyway (the maximal truncation filter).
- Job: **J19** (array 0-1, gpu:2); commits `e5_shuffle/results/e5_mc__<model>.json`.

---

## 1. The thesis question and the current (refined) finding
We stress-test the "LLMs build a world-model timeline" literature (Gurnee–Tegmark,
Godey geography, A Matter of Time, k-sparse "Finding Neurons in a Haystack") on
**low-resource, indirect, no-web-leakage Akkadian dating** (ORCC royal inscriptions,
year BCE from ruler).

**Refined claim from the balanced-MC results (see §6):** the date is **linearly
recoverable at the king's-name token** (`king_last` ≈ 0.5–0.7 Spearman) but the
whole-text **mean-pool ≈ 0.33–0.41 ≈ random (0.376)** — i.e. the model encodes the
date *locally at the explicit carrier* (replicating Gurnee–Tegmark's last-entity-token
result) but **does NOT diffuse it into a recoverable text-level chronological geometry**,
and this is not moved by scale, prompting, or objective. Declarative knowledge is
present locally (king token) and statable behaviorally (T9), yet absent as a global
structure over text. **The decisive missing control is `random king_last`** (§7.1).

---

## 2. Data + protocol (must respect)
- Corpus: `v_1/data/evaluation/corpora/orcc_corpus.parquet` (1,202 texts, `year` BCE
  with 9 nulls, 41 imbalanced rulers, `provenance`).
- **Pooling sites:** `mean` (tier0+maximal) and `king_last`/`king_mean` (last / mean of
  the commissioning ruler's name span, **tier0 ONLY** — maximal strips logographic names).
- **King-name coverage was ~37–44% and is now ~48%** after the 2026-07-05 spelling fix.
  The old claim "Neo-Babylonian texts are admin, never name the king" was WRONG: the NB
  royal inscriptions name the king as the opening word, but SYLLABICALLY
  (`d-AG-ku-dur2-ri-u2-ṣur` / `d-na-bi-um-ku-du-ur2-ri-u2-ṣu-ur2` = Nabû-kudurri-uṣur =
  Nebuchadnezzar II, usually with NO `m-` determinative), while `ruler_spellings.csv`
  only listed logographic `NIG2-DU-URU3` forms that occur 0× in ORCC. Discovered via the
  Thalesian English translations (J17: translator named Neb II in 63% of his texts while
  our list found 0%), then corpus-mined and cross-validated (mined 60% vs translated 63%,
  overlap 50/55). New coverage: Neb II 0→0.60, Nabonidus 0.18→0.56, Nabopolassar 0→0.80.
  NOTE: all existing king-token activations (J4/J12) predate this fix — a re-extraction
  would add the NB kings to king-site analyses and could add Neb II + Nabonidus to the
  maxking retained set (E[found/draw] ≈ 12–13 ≥ 6), widening its span from 612–705 to
  539–705 BCE.
- **Two CV protocols, both present:**
  - **balanced-MC** = `draws_matrix.npy` (200 balanced draws) × GroupKFold-by-ruler within
    each, best-k Spearman averaged. THIS is the thesis-headline protocol (the 0.41 lineage).
    Files: `shared/mc_probe.py`, `p1_gurnee_tegmark/probe_p1_mc.py`, `redo_t10_prompt/reprobe_king_mc.py`.
  - GroupKFold-by-ruler (single) = the earlier run (`probe_p1.py`, `reprobe_king_pv.py`).
- Random baseline = Qwen3-8B, `from_config`, seed 42, bf16 (matches on-disk `random` acts).

---

## 3. Code map (all committed on `main`)
```
shared/
  king_token.py       locate commissioning ruler's name span (word + tokenizer offsets)
  probe_sites.py      mean / king_last / king_mean poolers
  extract_lib.py      model load (causal/encoder; sdpa; random=from_config; umt5 fallback) + pooling
  mc_probe.py         balanced-MC engine (draws_matrix × GroupKFold; partial-coverage king)
  geo_loader.py       unambiguous import of geodesic/utils.py (avoids utils.py name clash)
  metrics.py          reuse pls_utils.compute_metrics + proximity_error + great_circle
  anchors.py          P3 ruler/year anchor prompts
  ruler_spellings.csv NEEDS EXPERT REVIEW (raises king coverage)
  sites_gazetteer.csv provenance -> lat/lon/region (P2), 97.5% row coverage
p1_gurnee_tegmark/  extract_king_acts.py (J4/J4c HF king), extract_mlm_king_acts.py (J4d
                    MLM king+mean, sign-level), probe_p1.py (GKF), probe_p1_mc.py (MC)
p2_godey_geography/ probe_p2.py (J7) — now sweeps PLS k + Ridge, best-k per lat/lon
p3_matter_of_time/  extract_anchor_acts.py (J5/J5b/J5c; --random flag), timeline_p3.py (J8)
p7_ksparse/         probe_p7.py (J9)
redo_t9_knowledge/  uses round2_phase1a run_kp/parse_kp/score_kp (J2)
redo_t10_prompt/    extract_prompted_king_acts.py, reprobe_king_pv.py (GKF), reprobe_king_mc.py (MC)
aggregate_tables.py J11 — builds results/RESULTS_stress_tests.md (labels + TF-IDF cite)
sbatch/             J2a,J2b,J3a,J3b,J3r_t10_reprobe_mc,J4,J4b,J4c_king_random,J4d_king_mlm,
                    J5,J5b_p3_anchors_gptoss,J5c_p3_anchors_random,J6_p1_probe,J6_p1_mc,
                    J7,J8,J9,J11_aggregate, submit_all.sh
```

**mc_probe / probe_p1_mc / probe_p2 now report BOTH PLS (swept k∈{1,2,3,5},
best-k surfaced + full per_k) AND a Ridge arm** (the user wanted both). Result
JSONs gained `best_k`, `per_k`, and `ridge{spearman_mean,…}`; old printers still
read the flat best-k keys. Re-run J6_p1_mc + J7 to regenerate with these.

## 4. The jobs (what each wanted to do)
| Job | Purpose | GPU |
|---|---|---|
| J2a/J2b | T9 direct knowledge (kp0/kp1/kp2) on qwen3×3 / gpt-oss | yes |
| J3a/J3b | T10 prompt-reprobe (pv0-3), extract prompted acts (mean+king) | yes |
| J3r_t10_reprobe_mc | T10 reprobe under balanced-MC on existing prompted acts | CPU |
| J4/J4b | king-token extraction (tier0) qwen3×3+thal×2+umt5 / gpt-oss | yes |
| J4c_king_random | king-token extraction for RANDOM Qwen3-8B (the control) | yes |
| J4d_king_mlm | MLM king+mean acts on balanced-MC setup (mlm_{tier0,maximal}_mean + kinglast/kingmean) | yes |
| J5 | P3 anchor embeddings (qwen×3, thal×2, umt5) | yes |
| J5b/J5c | P3 anchors for gpt-oss-120B / random-Qwen3-8B | yes |
| J6_p1_probe | P1 year-probe (GroupKFold) | CPU |
| J6_p1_mc | P1 year-probe balanced-MC (mean + king sites) | CPU |
| J7 | P2 geography (positive control) | CPU |
| J8 | P3 timeline (3a anchors-form-line, 3b texts-project); now incl. gpt-oss + random | CPU |
| J9 | P7 k-sparse localization | CPU |
| J11_aggregate | build results/RESULTS_stress_tests.md (P1+P2 tables, labels, TF-IDF cite) | CPU |

## 5. Where results live
- **In git / local (pull `main`)** — all *result JSONs*:
  - `p1_gurnee_tegmark/results/*.json` (GKF) + `results/mc/*.json` (balanced-MC)
  - `p2_godey_geography/results/*.json`, `p7_ksparse/results/*.json`
  - `p3_matter_of_time/results/p3_timeline__*.json`
  - `redo_t9_knowledge/direct_kp_*/scores|parsed|raw/*.json`
  - `redo_t10_prompt/results/*__t10_king_summary.json` (GKF) + `*__t10_mc_summary.json` (MC)
  - king coverage: `…/orcc__embed/activations/<method>_tier0_king{last,mean}/{metadata,king_coverage}.json`
- **Cluster-only (gitignored, `*.npz` + `*.out`)** — activations (mean/king/prompted/anchor)
  and job logs. NEVER commit these (they broke pushes; `*.out` and `**/logs/` are gitignored).

## 6. Results so far (headline)
- **T9 knowledge (kp0 ±50yr):** gpt-oss 8/8, qwen 1.7B 7/8, 8B 7/8, 32B 6/8 → models KNOW dates.
- **P2 geography (positive control, PASSES):** find-spot decodes 174–207 km, skill +0.22–0.35 vs
  centroid; random +0.221; thalesian-cunei400m best +0.347. Pipeline valid; mild scale effect.
- **P7 k-sparse (chance 0.58):** best macro-F1 only 0.67–0.72; random 0.667 → date weak/distributed.
- **P1 balanced-MC (200 draws) — Spearman:**
  | model | mean t0 | mean max | king_last | king_mean |
  |---|---|---|---|---|
  | qwen3-1.7B | 0.371 | 0.355 | 0.622 | 0.214 |
  | qwen3-8B | 0.365 | 0.363 | 0.507 | 0.207 |
  | qwen3-32B | 0.399 | 0.340 | 0.658 | 0.209 |
  | gpt-oss-120B | 0.404 | 0.330 | 0.666 | 0.224 |
  | thal-akk300m | 0.344 | 0.322 | 0.691 | 0.083 |
  | thal-cunei400m | 0.411 | 0.411 | 0.574 | 0.072 |
  | umt5-base | 0.334 | 0.295 | 0.454 | 0.272 |
  | random Qwen3-8B | 0.376 | 0.303 | **PENDING (J4c→J6)** | PENDING |
  | MLM (J4d) | ~0.42* | PENDING | PENDING | PENDING |
  | TF-IDF (cited) | 0.407 | — | n/a (no token) | n/a |
  (*MLM mean tier0 ≈ 0.424 in the former `balanced_mc_scoreboard.json`; J4d adds its
  maximal-mean + king sites. TF-IDF cited from that scoreboard: PLS 0.407 / Ridge 0.355.)
  (null/shuffled ≈ 0.01 everywhere.) mean-pool ≈ random & flat across scale/objective;
  king_last much higher; king_mean washes out. **These PLS numbers are pre-ridge;
  re-run J6_p1_mc to add the Ridge column + best-k, then J11 rebuilds the table.**
- **T10 balanced-MC (qwen3-1.7B):** mean = 0.406 across ALL pv0–pv3 (prompting doesn't change it);
  king_last 0.49–0.53; king_mean ≈ 0.

## 7. NEXT STEPS (what's left)
0. **Land code on `main`** (see header FF command). All the ridge/best-k, MLM (J4d), P3
   gpt-oss/random (J5b/J5c), and J11 code is on the feature branch only; the cluster won't
   run it until `main` is fast-forwarded.
1. **Re-run wave (in order) to regenerate with Ridge + best-k + MLM + random king:**
   `sbatch J4c_king_random.sbatch` (if not done) ‖ `sbatch J4d_king_mlm.sbatch`, then AFTER both:
   `sbatch J6_p1_mc.sbatch` (now covers mlm + fills random king_last), `sbatch J7_p2_geography.sbatch`
   (adds lat/lon best-k + Ridge), `sbatch J3r_t10_reprobe_mc.sbatch`; P3: `sbatch J5b…` `sbatch J5c…`
   then `sbatch J8_p3_timeline.sbatch`; finally `sbatch J11_aggregate.sbatch` (or run
   `python v_1/src/stress_tests/aggregate_tables.py` locally). **[DECISIVE] random `king_last`:**
   if ≈0 while pretrained ≈0.66 → "date-at-the-name" is real signal (claim airtight); if high →
   name-token identity, reinterpret.
2. **T10 balanced-MC for qwen3-8B/32B (+gpt-oss).** J3r_t10_reprobe_mc only produced qwen3-1.7B
   (others' prompted acts may not have been on disk at run time). Re-run J3a for 8B/32B if their
   `acts_<model>/prompted_king` are missing, then `sbatch J3r_t10_reprobe_mc.sbatch`.
3. **P3 timeline** (`p3_matter_of_time/results/p3_timeline__*.json`) — results exist now; interpret
   3a (anchors form ordered line) vs 3b (texts project) for the dissociation figure.
4. **gpt-oss-120B T10** never succeeded (OOM even sdpa+gpu:4). Optional; ladder stands without it.
5. **ruler_spellings.csv** expert review to raise king coverage above ~44%.
6. **GUI (J10)** — add new embeddings to `v_1/src/viz/seal_eda*.html`. **J11 aggregate DONE**
   (`aggregate_tables.py` → `results/RESULTS_stress_tests.md`, labels + TF-IDF cite); rerun after
   the wave above to fill in Ridge/best-k/MLM/random-king cells.
7. Write-up: the claim in §1, with the balanced-MC table as the centerpiece.

## 8. Operational notes (cluster + git)
- Cluster: Slurm `voltagepark`, `conda activate thesis`, repo `~/projects/HUJI-THESIS--YARIN`, work on `main`.
- **Only commit result JSONs.** `*.out` and `**/logs/` are gitignored; `*.npz` gitignored. Big logs
  previously broke pushes (100 MB limit) — if it happens: `git rm --cached` the *.out / delete them.
- Divergence fix (if `git pull` complains): `git config pull.rebase true` then `git pull`; or grab
  specific files with `git checkout origin/main -- <path>`.
- HF ids: Qwen/Qwen3-{1.7B,8B,32B}, openai/gpt-oss-120b, Thalesian/AKK_300m,
  Thalesian/cuneiformBase-400m, google/umt5-base. draws_matrix + fragment_order live in
  `v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/`.
- The stop-hook "Unverified commits" warning is cosmetic (no signing key) — ignore.
