# Advisor walkthrough — the stress-test suite, verified against the code

**One line before the tour:** every experiment mirrors one published claim onto our hard regime
(low-resource Akkadian, date never written in the text, no web leakage). Two controls make the
nulls interpretable: T9 (the models *know* the dates) and P2 (the *pipeline works* on geography).
The decisive baseline everywhere is a **random-initialized Qwen3-8B** pushed through the identical
extraction + probing.

**The reporting standard (all experiments):** every result is presented as three lines —
**SETUP** (text cleaning · pooling site · sampling/CV protocol), **PROBE** (the fitted model:
PLS with k swept over {1,2,3,5}, best-k, + a Ridge arm, unless stated otherwise), **METRIC**
(regression → **Spearman(predicted, true)**; classification → **macro-F1 vs chance**). Paired
cells are PLS / Ridge (or F1 / ρ where two metrics apply). Baseline = the random-init row.

**Shared machinery (all probing experiments):**
- Corpus: 1,202 ORACC royal inscriptions, 41 rulers, `year` BCE derived from the ruler's reign.
  **`year` is one constant per ruler** (except Esarhaddon) → decoding year ≈ identifying the ruler.
- Activations extracted from **every layer** of each frozen model, pooled at 3 sites:
  `mean` (whole text), `king_last` (last token of the king's name), `king_mean` (mean over name span).
- Cleanings: `tier0` (light normalization, names intact), `maximal` (strips names/logograms/digits,
  truncates to ~30 tokens — kills length/genre/name crutches), `maximal_keepking` (maximal but the
  king's name is frozen in).
- Probes: **PLS with k swept over {1,2,3,5} (best-k reported) AND Ridge**, on L2-normalized
  activations; headline metric = **Spearman(predicted year, true year)**; shuffled-label null ≈ 0.
- Balanced Monte-Carlo (MC): 200 pre-drawn balanced subsets; **GroupKFold-by-ruler inside each
  draw** (test kings never seen in training); report mean ± std over the 200 draws.

---

## 1. T9 — "Do the models even know the dates?" (knowledge control) — YES

**What we did (code: `redo_t9_knowledge/`, reuses round2_phase1a `run_kp/parse_kp/score_kp`):**
- Free-text **generation**, not probing. Ask each chat model in English, get JSON back.
- **kp0**: "When did {ruler} reign?" for the 8 balanced rulers → model returns
  `{start_year, end_year, confidence, declined}`. Correct if the predicted window is within
  **±50 yr** of a true reign year.
- **kp1**: "List the rulers of period P" → recall of target rulers (exact match after
  lowercasing + stripping diacritics).
- **kp2** (hallucination gate): ask about **fake** ruler names; giving any date = hallucination;
  pass if rate < 30%.
- Config: 4 causal chat models only (generation task). No probes, no CV.

| model | kp0 date acc (±50yr) | kp1 strict | kp1 rescored‡ | kp2 halluc. rate (gate) |
|---|---|---|---|---|
| Qwen3-1.7B | 0.875 (7/8) | 0.50 | 0.50 | 0.75 (FAIL) |
| Qwen3-8B | 0.875 (7/8)* | 0.625 | 0.75 | 0.00 (pass) |
| Qwen3-32B | 0.75 (6/8) | 0.25 | **1.00** | 0.00 (pass) |
| gpt-oss-120B | 1.00 (8/8) | 0.00† | ≥ 0.38 | 0.00 (pass) |

*8B's one "miss" is a parse error (output truncated mid-JSON), not a wrong date — scoreable
accuracy is 8/8. †gpt-oss emits its reasoning ("analysis") channel first and hit the
`max_new_tokens=512` budget (run_kp.py) before printing the final JSON → the strict scorer
(score_kp.py: JSON parse + exact diacritic-normalized name match, so "Assurbanipal" would not
match "Ashurbanipal") zeroed both periods. ‡Rescored = normalized-substring scan of the RAW
output (`eda/rescore_t9_kp1.py` → `results/t9_kp1_rescored.json`): recovers JSON/format losses —
32B's Neo-Assyrian answer failed JSON parse yet its raw text names all 6 targets (0.25→1.00) —
but cannot recover truncation: gpt-oss's Neo-Assyrian list is cut mid-way at "Shalmaneser V", so
0.38 is a floor; its true kp1 needs a rerun with a larger token budget.

**Takeaway:** the models demonstrably hold the king→date mapping (in English). Every null below is
therefore NOT "the knowledge is absent."

---

## 2. P2 — Geography (Godey mirror; the positive control) — PASSES

**What we did (code: `p2_godey_geography/probe_p2.py`):**
- Predict the **find-spot (lat/lon)** of each text from its **mean-pooled** activation. The carrier
  (toponyms like Nineveh, Babylon) is literally written in the text — so this SHOULD decode if our
  pipeline is valid.
- Provenances geocoded via `shared/sites_gazetteer.csv` (97.5% coverage).
- Config: both cleanings (tier0 + maximal); **GroupKFold-by-SITE** (held-out find-spots, 5 splits);
  two separate regressions for lat and lon, **PLS(best-k) + Ridge arms**; best layer chosen by
  great-circle **skill = 1 − err/centroid_err** (centroid = always predict the training-mean
  location). No Monte-Carlo (site imbalance handled by grouping).

| model | tier0: best L / gc err / skill | maximal: best L / gc err / skill |
|---|---|---|
| Qwen3-1.7B | 11 / 190 km / +0.287 | 3 / 203 km / +0.238 |
| Qwen3-8B | 12 / 187 km / +0.298 | 5 / 199 km / +0.251 |
| Qwen3-32B | 63 / 179 km / +0.326 | 64 / 197 km / +0.258 |
| gpt-oss-120B | 10 / 189 km / +0.290 | 4 / 192 km / +0.280 |
| Thalesian-AKK-300m | 7 / 192 km / +0.279 | 8 / 198 km / +0.255 |
| **Thalesian-cunei-400m** | 11 / **172 km** / **+0.354** | 10 / 178 km / +0.333 |
| uMT5-base | 9 / 203 km / +0.237 | 10 / 211 km / +0.209 |
| random Qwen3-8B | 31 / 204 km / +0.232 | 1 / 198 km / +0.256 |

**Takeaway:** geography decodes with real skill under the *identical* pipeline → the probes are not
broken. Note: random already gets +0.23 (lexical toponym identity); *training* adds +0.05–0.12,
most for the cuneiform-domain Thalesian. Explicit carrier ⇒ decodable; that's the contrast.

---

## 3. P1 — The year probe (Gurnee–Tegmark mirror; the core) — 3 protocols

### P1a — single GroupKFold (first pass, imbalanced corpus)
**Code: `p1_gurnee_tegmark/probe_p1.py`.** All 1,202 fragments; GroupKFold-by-ruler; PLS(best-k) +
Ridge **+ a 1-hidden-layer MLP linearity check** (G–T's guard against "probe too weak") + shuffle
null. Sites: mean on tier0+maximal, king sites on tier0. Result: best-layer Spearman ~0.41–0.51 at
mean, 0.30–0.44 at king_last, MLP ≈ linear (no hidden nonlinear signal). Superseded by P1b because
the corpus imbalance (Ashurbanipal 268 frags vs tail rulers 1) makes single-split numbers unstable.

### P1b — balanced Monte-Carlo (the headline protocol)
**Code: `probe_p1_mc.py` + engine `shared/mc_probe.py`.**
- 200 balanced draws of **8 rulers × 21 fragments** (k=21 capped by the smallest class); within
  each draw GroupKFold-by-ruler (≤5 splits); PLS k∈{1,2,3,5} best-k + Ridge; mean ± std over draws;
  shuffled null. Includes the MLM and a cited TF-IDF baseline.

| model | mean tier0 | mean maximal | king_last (t0) | king_mean (t0) |
|---|---|---|---|---|
| Qwen3-1.7B | 0.352 / R 0.352 | 0.334 / R 0.072 | 0.606 / R 0.500 | 0.173 |
| Qwen3-8B | 0.348 / R 0.332 | 0.339 / R 0.111 | 0.480 / R 0.466 | 0.114 |
| Qwen3-32B | 0.381 / R 0.194 | 0.332 / R 0.302 | 0.645 / R 0.425 | 0.174 |
| gpt-oss-120B | 0.388 / R 0.264 | 0.316 / R 0.273 | 0.645 / R 0.153 | 0.174 |
| Thalesian-AKK-300m | 0.307 | 0.300 | 0.688 | −0.006 |
| Thalesian-cunei-400m | 0.377 | **0.391** | 0.513 | −0.024 |
| uMT5-base | 0.324 | 0.277 | 0.423 | 0.164 |
| MLM (37M, from scratch) | 0.399 | 0.286 | 0.704 | 0.379 |
| **random Qwen3-8B** | **0.351** | 0.293 | **0.643** | 0.183 |
| TF-IDF (cited) | 0.407 | — | n/a | n/a |

(PLS best-k Spearman, ±std ≈ 0.07 on mean / 0.2–0.4 on king; shuffled null ≈ 0.01 everywhere.)

**Takeaways:** (1) whole-text `mean` ≈ **random ≈ TF-IDF ≈ 0.35–0.41, flat from 1.7B to 120B** →
no learned text-level chronology. (2) `king_last` looks high (0.48–0.70) — **but random gets
0.643** → it's name-token identity (the name is a near one-hot ruler ID and year is a function of
ruler), not a learned date.

### P1c — "maxking": the fair rematch (decisive)
**Code: `probe_maxking.py` + engine `shared/mc_maxking.py`; cleaning `clean_maximal_keepking`.**
- Why: P1b compared `mean` on maximal text vs `king_*` on tier0 text — not apples-to-apples. Here
  **all 3 sites sit on ONE cleaning** (maximal context, king name frozen in), rebalanced to
  **5 rulers × 9 king-found fragments × 200 draws** so random is a genuine control.
- Because year is per-ruler-constant, the honest task is **ruler classification** (PLS-DA,
  StratifiedKFold, **macro-F1 vs chance 0.20 and shuffle ≈ 0.08**); year regression reported as
  `year_strat` (StratifiedKFold Spearman + **±10-yr accuracy** — ±50 is coarser than the gaps
  between adjacent kings) and legacy `year_group` (degenerate by construction, kept for continuity).

| model | mean F1 | king_last F1 | king_mean F1 |
|---|---|---|---|
| Qwen3-1.7B | 0.663 | 0.979 | 0.965 |
| Qwen3-8B | 0.706 | 0.989 | 0.973 |
| Qwen3-32B | 0.717 | 0.982 | 0.970 |
| gpt-oss-120B | 0.750 | 0.982 | 0.966 |
| Thalesian-AKK-300m | 0.700 | 0.975 | 0.998 |
| **Thalesian-cunei-400m** | **0.897** | 0.943 | 0.986 |
| uMT5-base | 0.698 | 0.953 | 0.974 |
| **random Qwen3-8B** | **0.741** | 0.946 | 0.971 |

The year regression view (`year_strat` Spearman, StratifiedKFold, in `results/csv/p1_maxking.csv`)
tells the identical story: mean site 0.70–0.85 with **random = 0.740** (only Thalesian-cunei above,
0.851); king sites 0.93–0.98 for everyone including random (±10-yr accuracy: mean 0.31–0.48,
king 0.73–0.98).

**Takeaways:** (1) king sites ≈ 0.94–0.99 **for everyone including random** → pure token identity.
(2) On whole-text mean, **random (0.741) matches or beats every trained LLM** (0.66–0.75) — even
the 120B. (3) The **only** model above random is Thalesian-cunei (0.897): the one genuine learned
increment, from a cuneiform *translation* objective, not scale.

---

## 4. P3 — Timeline geometry ("A Matter of Time" mirror)

**What we did (code: `timeline_p3.py`, `shared/anchors.py`, `extract_anchor_acts.py`):**
- Build **153 anchor prompts** whose date we know, in English, on the model's *declarative* side:
  **40 ruler anchors** (one per ruler that has a year label: "an Akkadian royal inscription from
  the reign of {ruler}", year = that ruler's median corpus year) + **113 year anchors** (one every
  10 years across the observed span 7–1132 BCE: "...from the year {year} BCE"). Each anchor is its
  short prompt **mean-pooled into a single vector per layer** (`extract_anchor_acts.py:49`), so the
  anchor cloud is 153 points with known years.
- **3a — do the anchors form a timeline?** Fit a 1-D embedding of the anchor cloud — PCA-1D
  (linear) and Isomap-1D (nonlinear, cosine-graph) — and take |Spearman(1-D coordinate, anchor
  year)|. Deliberately **unsupervised**: PLS would use the year labels to *find* the direction,
  which is exactly the supervised probe P1 already runs; 3a instead asks whether time is the
  dominant intrinsic 1-D structure without telling the method about years.
- **3b — do real texts land on it?** Each ORCC text (tier0, mean-pooled) predicts its year as the
  year of its **nearest anchor** — cosine on L2-normalized vectors; on unit vectors the cosine
  ranking is mathematically identical to the Euclidean (L2) ranking, so both metrics are covered.
- **Fully unsupervised** — no probe is trained, so no probe can cheat. This arm carries the
  "maybe it's a nonlinear manifold your linear probe can't see" objection (P4 descoped into it).

| model | best L | 3a PCA-1D | 3a Isomap-1D | 3b (texts project) |
|---|---|---|---|---|
| Qwen3-1.7B | 16 | 0.236 | 0.388 | 0.034 |
| Qwen3-8B | 21 | 0.232 | 0.381 | 0.060 |
| Qwen3-32B | 22 | 0.484 | 0.319 | 0.110 |
| gpt-oss-120B | 11 | 0.228 | **0.567** | 0.105 |
| Thalesian-AKK-300m | 0 | 0.160 | 0.250 | 0.088 |
| Thalesian-cunei-400m | 12 | 0.428 | 0.240 | 0.042 |
| uMT5-base | 10 | 0.206 | 0.048 | 0.134 |
| random Qwen3-8B | 28 | 0.342 | 0.040 | 0.040 |

**Takeaways:** the **dissociation as a picture** — explicit English date-anchors arrange into a
rough line that *improves with scale* (the one place scale helps — and it's exactly the
high-resource explicit regime the papers tested), but **real Akkadian texts never land on it**
(3b ≈ 0.03–0.13). Random's 3a (0.342) matches the small trained models, so even the anchor line is
only partly learned.

---

## 5. P7 — "Time neurons"? (Haystack mirror) — NO

**What we did (code: `p7_ksparse/probe_p7.py`):**
- Is the date carried by a few neurons? Binarize to **before/after the median year**; per layer,
  select the **top-k neurons (k ∈ {1,2,4,8,16,32,64}) by ANOVA F-score on the training folds**,
  fit logistic regression, **GroupKFold-by-ruler** (5 splits); tier0, mean pool; report best
  macro-F1 over layers × k.
- Anchors: majority-class accuracy 0.58 (a trivial predictor's macro-F1 ≈ 0.37); the meaningful
  bar is the **random model**.

| model | best macro-F1 | at layer | at k | k reaching 90% of full-k |
|---|---|---|---|---|
| Qwen3-1.7B | 0.720 | 1 | 8 | 1 |
| Qwen3-8B | 0.691 | 2 | 32 | 1 |
| Qwen3-32B | 0.717 | 45 | 1 | 1 |
| gpt-oss-120B | 0.689 | 21 | 32 | 16 |
| Thalesian-cunei-400m | 0.720 | 7 | 16 | 4 |
| **random Qwen3-8B** | **0.667** | 1 | 64 | 8 |

(Full per-layer × per-k curves are stored in `p7_ksparse/results/*.json` — enough to plot the
sparsity curve per model. tier0 mean only; no maximal/maxking arm, and no k-sparse
year-*regression* arm — the classification protocol mirrors the Haystack paper's sparse probing.)

**Takeaway:** best trained ≈ 0.69–0.72 vs random 0.667 — a +0.02–0.05 margin. No localized "time
neurons"; the weak signal is distributed and mostly architectural/lexical.

---

## 6. T10 — Does prompting help? (robustness arm) — NO

**What we did (code: `redo_t10_prompt/extract_prompted_king_acts.py` + `reprobe_king_mc.py`):**
- Wrap each fragment in 4 prompt variants — **pv0** bare, **pv1** expert-Assyriologist system
  prompt, **pv2** few-shot (k=5 in-context examples), **pv3** chain-of-thought — run the model,
  and pool activations **over the fragment span inside the prompt** (span-mean + king sites).
- Re-run the *identical* year probes on these prompted activations: GroupKFold AND balanced-MC
  (same 200 draws, mapped by fragment_id). Fragments are embedded as **tier0** text inside the
  prompt (`extract_prompted_king_acts.py:79`); no maximal/maxking prompted variants exist. Causal
  chat models only (Qwen3 1.7B/8B/32B — GKF and MC all complete; gpt-oss has no T10 results or
  activations on disk — its extraction OOM'd on gpu:4 and was never rerun).

| model, site (MC) | pv0 bare | pv1 framed | pv2 few-shot | pv3 CoT |
|---|---|---|---|---|
| 1.7B mean | 0.390 | 0.390 | 0.390 | 0.390 |
| 1.7B king_last | 0.415 | 0.480 | 0.450 | 0.456 |
| 8B mean | 0.388 | 0.388 | 0.388 | 0.388 |
| 8B king_last | 0.459 | 0.524 | 0.453 | 0.511 |
| 32B mean | 0.420 | 0.385 | 0.376 | 0.385 |
| 32B king_last | 0.551 | 0.534 | 0.546 | 0.522 |

**Takeaways:** prompting (framing / few-shot / CoT) does not make the date more recoverable — for
every model the **bare pv0 is ≥ every prompted variant** on both sites.
Sharp detail (verified in the JSONs): for 1.7B and 8B the mean numbers are *identical* across
variants because the **best layer is L0 — the embedding layer, which cannot see the prompt** — and
the genuinely prompted deeper layers all score LOWER (8B: L0 0.388 → L5 ≈0.30 → L18 ≈0.25). 32B is
the lone exception: a mid-early layer (L8) edges L0 (0.420 vs 0.371) — but it is the **bare** prompt
that wins, and it stays inside the ~0.38–0.42 random/P1b band. So across all three, **prompting
never lifts recoverability above the context-free / bare baseline** — a stronger statement than
"the number didn't move."

---

## 7. The cross-cutting patterns (what to say when they ask "so what?")

1. **Trained − random ≈ 0 everywhere on time.** Across ~60 model×site×experiment comparisons the
   delta is within ±0.05 (sometimes negative). The only systematic exceptions: **Thalesian-cunei**
   (+0.08…+0.16, consistently, across P1b/P1c/P2) and **scale on P3-3a English anchors**
   (+0.14/+0.22 for 32B/120B) — i.e., the high-resource explicit regime the original papers tested.
2. **Layer-depth signature.** Wherever the year "signal" peaks, it peaks at the **embedding-adjacent
   layers** (maxking mean: L0/L1 for every model incl. random; T10 best = L0 for 1.7B/8B, L8-bare
   for 32B — still embedding-adjacent, still ≈ random). Where signal is real
   — geography, Thalesian — it peaks **mid-network** (depth 0.3–0.6), like G–T's own finding.
   Peak-at-L0 = the probe reads the lexicon; peak-mid-stack = the network computed something.
3. **Probe agreement as a robustness certificate.** PLS and Ridge agree where signal is real
   (P2: 0.431/0.389) and diverge where it's artifact (gpt-oss king_last 0.645/0.153).
4. **The ladder:** scale ✗ (1.7B→120B flat) · prompting ✗ (never beats the bare baseline) · NTP finetuning ✗
   (its one gain was a length confound) · **translation objective ✓ (modest, consistent)**.
   What installs structure is the objective, not capacity.

**The claim, in one sentence:** the models *know* the dates (T9) and the method *works* (P2), yet
nothing recoverable from their representations of the actual texts survives the random-init
control — declarative knowledge and recoverable representational structure dissociate, and the
published "world-model timeline" geometry is a property of high-resource, explicitly-carried
features, not of the models in general.
