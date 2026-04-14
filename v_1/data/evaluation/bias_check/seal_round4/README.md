# SEAL Bias Check Results (Phase C re-run, 2026-04-14)

Script: `v_1/src/bias_check/06_bias_check_cv.py`
Data: `v_1/data/evaluation/corpora/seal_corpus.parquet` — 384 fragments (SEAL=302, DLL=44, LBPL=38)

> **Round 5 re-run (2026-04-14):** Chungrong delivered corrected CSVs on 2026-04-14 with partially
> resolved period labels. Phases 0→A→B→C were re-run on the new data. Key change: `period` now has
> 9 classes (was 6) after splitting `Middle Babylonian/Assyrian` (65→24+6+35) and
> `Neo-Assyrian and Late Babylonian` (44→18+26). See Section 19 of the pipeline plan for details.

## Method

For each of 6 tasks × 2 text cleanings (12 combinations):

1. **Text cleaning**: `tier0` = strip @-markup / non-breaking spaces / subscript-x.
   `maximal` = tier0 + strip digits, logograms, long-vowel markers, truncate to 30 tokens, etc.
2. **Vectorizer**: TF-IDF `char_wb(2–5)`, 10k features, `sublinear_tf=True`
3. **Classifier**: LogisticRegression, `class_weight="balanced"`, C chosen by grid search
   `[0.001, 0.01, 0.1, 1.0, 10.0]` over the same CV splitter
4. **Cross-validation**: Stratified k-fold, **k = min(5, smallest_class_size)**
   — `domain` uses k=5 (smallest class has 38 fragments); all other tasks use k=2
   (smallest surviving class has exactly 2 fragments). Singletons (N=1) dropped before CV.
5. **Permutation test**: 1000 label-shuffled CV runs → null distribution → p-value
   (Ojala & Garriga, JMLR 2010). Scoring: macro-F1 throughout.

All metrics are **out-of-fold** (OOF) — no data leakage. TF-IDF is fit inside each fold.

## Summary Table

```
Task                  N  Classes |  Acc(t0)  F1m(t0)  F1w(t0) |  Acc(mx)  F1m(mx)  F1w(mx) |    ΔAcc    ΔF1m
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
domain              384        3 |    0.979    0.952    0.979   |    0.948    0.876    0.947   |  -0.031  -0.076
period              383        9 |    0.757    0.473    0.751   |    0.590    0.352    0.621   |  -0.167  -0.121
genre               384       16 |    0.602    0.362    0.610   |    0.365    0.269    0.400   |  -0.237  -0.093
sub_genre           246       43 |    0.333    0.286    0.301   |    0.305    0.267    0.284   |  -0.028  -0.019
provenance          374       25 |    0.259    0.171    0.243   |    0.214    0.122    0.213   |  -0.045  -0.049
sub_provenance      374       25 |    0.259    0.171    0.243   |    0.214    0.122    0.213   |  -0.045  -0.049
```

All 12 runs: **p = 0.001** (minimum with 1000 permutations). Signal is genuine for every task.

**Metric key**:
- `Acc` = OOF accuracy
- `F1m` = macro-F1 (equal weight per class — the headline metric; robust to imbalance)
- `F1w` = weighted-F1 (weighted by class size; closer to accuracy)
- `Δ` = maximal − tier0 (negative = cleaning hurts)

## Interpretation

**Why macro-F1 and not accuracy?** All tasks are class-imbalanced. The majority class alone
accounts for 60% of `period` (Old Babylonian), 42% of `genre` (incantations), 30% of
`provenance` (Unknown). A classifier predicting only the majority class would get high
accuracy for free. Macro-F1 penalises ignoring minority classes.

**The cleaning delta (Δ) is the key diagnostic:**

| Task | Δ F1m | Interpretation |
|------|------:|----------------|
| `genre` | −0.092 | Large drop: genre signal is heavily tied to writing conventions (logograms, determinatives, subscript digits) |
| `period` | −0.121 | Large F1m drop: period markers tied to writing conventions. F1m lower than round 4 (0.608→0.473) due to 9 classes vs 6; new small classes (Middle Assyrian=6, Archaic=2) score F1=0 |
| `sub_genre` | −0.019 | Almost no drop: sub-genre signal lives in content vocabulary, not markup |
| `domain` | −0.076 | Corpus-level separation survives aggressive cleaning — SEAL/DLL/LBPL differ in vocabulary, not just conventions |
| `provenance` | −0.049 | Moderate drop; provenance is the weakest signal overall |

**`period` collinearity with `domain`** (see also Section 16.4 of pipeline plan):
DLL fragments are *exclusively* "Neo or Late Babylonian" (after round 5 split); LBPL fragments
are *exclusively* "Late Babylonian"; only SEAL spans 7 period values (see table below).
This means any classifier that can identify the source corpus gets `period` nearly for free.
The `period` F1=0.473 is therefore partly a `domain` proxy, not purely diachronic signal.
The within-SEAL period breakdown is the more informative question for Phase D.

```
Period value                  Corpus   N   Notes
──────────────────────────────────────────────────────────────────────────
Old Babylonian                SEAL   229
Middle Babylonian/Assyrian    SEAL    35   ← 35 genuinely ambiguous (round 5 remainder)
Middle Babylonian             SEAL    24   ← split from compound in round 5
Neo or Late Babylonian        DLL     26   ← entire DLL, still compound (26 ambiguous)
Neo-Assyrian                  DLL     18   ← split from compound in round 5
Late Babylonian               LBPL    38   ← entire LBPL corpus
Middle Assyrian               SEAL     6   ← split from compound in round 5
Old Assyrian                  SEAL     5
Archaic/Old Akkadian/Ebla     SEAL     2
Later Periods (SB, NA, LB)   SEAL     1   ← singleton, dropped from CV
```

**`provenance` = `sub_provenance`**: identical results at both cleanings. These are
parallel columns (ancient site name vs. modern excavation name) with a 1:1 mapping
across all 384 fragments.

## Per-task results and plots

To open a plot: `open v_1/data/evaluation/bias_check/seal_round4/<task>/<cleaning>/plots/<plot>.png`

---

### `domain` — corpus membership (SEAL / DLL / LBPL)
N=384 | 3 classes | k=5 | F1m: **0.952** (t0) / **0.876** (mx) | p=0.001

Sanity-check task. High score confirms the three corpora are lexically distinct even after
aggressive cleaning. DLL/LBPL errors increase under maximal — writing conventions help
separate corpora, but vocabulary alone is already sufficient. Slight drop vs round 4
(0.889→0.876 maximal) reflects the 1-word DLL change (5,694→5,693).

| Plot | tier0 | maximal |
|------|-------|---------|
| Confusion matrix | [confusion.png](domain/tier0/plots/confusion.png) | [confusion.png](domain/maximal/plots/confusion.png) |
| Permutation null | [perm_null.png](domain/tier0/plots/perm_null.png) | [perm_null.png](domain/maximal/plots/perm_null.png) |
| Per-class F1 | [per_class_f1.png](domain/tier0/plots/per_class_f1.png) | [per_class_f1.png](domain/maximal/plots/per_class_f1.png) |

Full reports: [tier0/report.md](domain/tier0/report.md) · [maximal/report.md](domain/maximal/report.md)

---

### `period` — temporal period
N=383 | 9 classes | k=2 | F1m: **0.473** (t0) / **0.352** (mx) | p=0.001

*(Round 4 had 6 classes, F1m=0.608/0.464. Round 5 split increased class count to 9, dropping F1m
as expected — more fine-grained labels, many small classes.)*

Second-largest F1m cleaning drop (−0.121) of all tasks. Period signal is heavily tied to writing
conventions. Note: period is nearly collinear with domain — DLL is entirely
"Neo or Late Babylonian", LBPL entirely "Late Babylonian". Main confusion is
`Middle Babylonian` ↔ `Old Babylonian` and `Neo-Assyrian` ↔ `Neo or Late Babylonian`
(see confusion matrix in tier0/report.md). New small classes (`Middle Assyrian`=6,
`Archaic/Old Akkadian/Ebla`=2) have F1=0 due to insufficient samples for k=2 CV.
Under maximal, domain-level separation (`Late Babylonian`, `Neo or Late Babylonian`) starts
to blur once writing conventions are stripped.

| Plot | tier0 | maximal |
|------|-------|---------|
| Confusion matrix | [confusion.png](period/tier0/plots/confusion.png) | [confusion.png](period/maximal/plots/confusion.png) |
| Permutation null | [perm_null.png](period/tier0/plots/perm_null.png) | [perm_null.png](period/maximal/plots/perm_null.png) |
| Per-class F1 | [per_class_f1.png](period/tier0/plots/per_class_f1.png) | [per_class_f1.png](period/maximal/plots/per_class_f1.png) |

Full reports: [tier0/report.md](period/tier0/report.md) · [maximal/report.md](period/maximal/report.md)

---

### `genre` — text genre (16 classes)
N=384 | 16 classes | k=2 | F1m: **0.361** (t0) / **0.269** (mx) | p=0.001

Largest accuracy drop of all tasks (−0.234). Genre signal is strongly tied to writing
conventions — logograms and determinatives are genre-diagnostic (e.g. Sumerian logograms
dominate incantations). After maximal cleaning, accuracy halves (0.599 → 0.365).
`incantations` is the dominant class (161/384) so weighted-F1 remains higher than macro-F1.

| Plot | tier0 | maximal |
|------|-------|---------|
| Confusion matrix | [confusion.png](genre/tier0/plots/confusion.png) | [confusion.png](genre/maximal/plots/confusion.png) |
| Permutation null | [perm_null.png](genre/tier0/plots/perm_null.png) | [perm_null.png](genre/maximal/plots/perm_null.png) |
| Per-class F1 | [per_class_f1.png](genre/tier0/plots/per_class_f1.png) | [per_class_f1.png](genre/maximal/plots/per_class_f1.png) |

Full reports: [tier0/report.md](genre/tier0/report.md) · [maximal/report.md](genre/maximal/report.md)

---

### `sub_genre` — sub-genre (43 classes, SEAL only)
N=246 | 43 classes | k=2 | F1m: **0.286** (t0) / **0.267** (mx) | p=0.001

Smallest cleaning drop of all tasks (−0.019). Sub-genre signal lives in content vocabulary,
not markup — stripping writing conventions barely changes performance. Many classes have
only N=2 fragments (the CV floor), so per-class F1 is highly variable. Still 12× above
macro-chance (1/43 ≈ 0.023).

| Plot | tier0 | maximal |
|------|-------|---------|
| Confusion matrix | [confusion.png](sub_genre/tier0/plots/confusion.png) | [confusion.png](sub_genre/maximal/plots/confusion.png) |
| Permutation null | [perm_null.png](sub_genre/tier0/plots/perm_null.png) | [perm_null.png](sub_genre/maximal/plots/perm_null.png) |
| Per-class F1 | [per_class_f1.png](sub_genre/tier0/plots/per_class_f1.png) | [per_class_f1.png](sub_genre/maximal/plots/per_class_f1.png) |

Full reports: [tier0/report.md](sub_genre/tier0/report.md) · [maximal/report.md](sub_genre/maximal/report.md)

---

### `provenance` — ancient site name (25 classes)
N=374 | 25 classes | k=2 | F1m: **0.171** (t0) / **0.122** (mx) | p=0.001

Weakest signal of the 6 tasks. `Unknown` dominates (112/374 = 30%). Results are
identical to `sub_provenance` — the two columns are 1:1 parallel namings (ancient vs
modern site) with no structural difference. Raised with Chungrong.

| Plot | tier0 | maximal |
|------|-------|---------|
| Confusion matrix | [confusion.png](provenance/tier0/plots/confusion.png) | [confusion.png](provenance/maximal/plots/confusion.png) |
| Permutation null | [perm_null.png](provenance/tier0/plots/perm_null.png) | [perm_null.png](provenance/maximal/plots/perm_null.png) |
| Per-class F1 | [per_class_f1.png](provenance/tier0/plots/per_class_f1.png) | [per_class_f1.png](provenance/maximal/plots/per_class_f1.png) |

Full reports: [tier0/report.md](provenance/tier0/report.md) · [maximal/report.md](provenance/maximal/report.md)

---

### `sub_provenance` — modern excavation site name (25 classes)
N=374 | 25 classes | k=2 | F1m: **0.171** (t0) / **0.122** (mx) | p=0.001

Identical results to `provenance` at both cleanings (same fragments, same labels in
different notation). Kept as a separate task pending Chungrong's clarification on whether
these are intended to be distinct experimental conditions.

| Plot | tier0 | maximal |
|------|-------|---------|
| Confusion matrix | [confusion.png](sub_provenance/tier0/plots/confusion.png) | [confusion.png](sub_provenance/maximal/plots/confusion.png) |
| Permutation null | [perm_null.png](sub_provenance/tier0/plots/perm_null.png) | [perm_null.png](sub_provenance/maximal/plots/perm_null.png) |
| Per-class F1 | [per_class_f1.png](sub_provenance/tier0/plots/per_class_f1.png) | [per_class_f1.png](sub_provenance/maximal/plots/per_class_f1.png) |

Full reports: [tier0/report.md](sub_provenance/tier0/report.md) · [maximal/report.md](sub_provenance/maximal/report.md)

---

## Output layout

```
seal_round4/
├── README.md                   — this file (summary + all plot links)
├── label_issues.md             — confusion matrices + fragment IDs for Chungrong review
├── <task>/
│   ├── tier0/
│   │   ├── task_summary.json   — N, classes, singletons dropped, k used
│   │   ├── metrics.json        — CV scores, C grid, per-class F1, perm test
│   │   ├── report.md           — human-readable summary + confusion matrix table
│   │   └── plots/
│   │       ├── confusion.png
│   │       ├── perm_null.png
│   │       └── per_class_f1.png
│   └── maximal/
│       └── (same structure)
└── (6 task directories)
```

To open all tier0 confusion plots at once:
```bash
open v_1/data/evaluation/bias_check/seal_round4/*/tier0/plots/confusion.png
```
