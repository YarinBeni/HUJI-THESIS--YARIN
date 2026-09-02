# PAPER_NOTES — running log for the eventual paper (started 2026-09-02)

Purpose: everything a paper draft will need, kept current as work happens —
motivation, data, method, each experiment with its numbers, every decision
with its reason, negative results, and how to reproduce. Written so a reader
who was not in the room can draft from it.

## 1. Motivation (from the M.Sc.)

LLMs place famous historical figures on a linear time axis (Spearman ≈ .88)
yet barely order Akkadian royal-inscription fragments (≈ .3, hardly above a
random-weights control on the cleaned text). Phase-2 interpretability work
(LEACE ladders, logit/tuned lens, SAE features) found the document signal at
fixed features to be weak, entangled with provenance/length, and not readable
as calendar tokens. The PhD hypothesis: a small head trained to be
*invariant to the confounds* (name masking, formula stripping, cropping)
while *ordering* documents would recover chronology the frozen probe cannot.

## 2. Data

ORACC royal inscriptions with a catalogue year: **1,193 fragments / 40
rulers** on the raw (`tier0`) Akkadian tier, 1,187 on the cleaned (`maximal`)
tier. English gloss (`eng_tier0`) for each. Metadata: ruler, period,
sub-genre (object type), provenance, length. Known quirks, all documented in
`chrono/data/contract.py` and `STATUS.md`:
* `year` holds a CENTURY for 6 rulers / 13–14 docs → mapped to mid-century,
  flagged `t_quality = century`.
* 39 of 40 rulers carry ONE distinct year (ICC = 1): the exchangeable unit
  is the ruler, so every null is a ruler-level permutation and every
  per-fold statistic is pooled (SLA §7).
* 136 Akkadian docs share byte-identical text in 50 groups (ORACC
  exemplars); 4 groups span rulers.

**Decision (2026-09-02): keep duplicates and century docs.** Sensitivity
re-read of the frozen out-of-fold scores with either or both removed moves
mc ρ by ≤ .01 in every arm checked (head .609 → .613; ridge .447 → .448).
Recorded in `reports/sensitivity_dup_century.md`. Revisit only if a later
claim depends on the last .01.

## 3. Method

Frozen encoder → per-view vector (mean pooling, a fixed layer) → **AdapterHead**
(Linear–GELU–Linear; a scalar lateness axis *s* and a projection *p*) trained
with: Bradley–Terry ordering on *s* over same-language document pairs whose
years differ; Barlow-Twins invariance + redundancy reduction on *p* between a
document view and its corrupted view (EMA target branch); a variance floor.
Views: name masking (`<RULER>` over the pre-masked ORACC tier), formula
stripping, crop-16/32, span dropping, orthographic normalisation. Read-out:
pooled Spearman over centred out-of-fold scores on frozen ruler-grouped folds,
and mean Spearman over 200 frozen 8-ruler draws; ruler-block permutation null.

Baselines on identical features/folds/read-out: PLS (k = 2, the M.Sc.
convention), ridge on original views, ridge on ALL views (augmentation as
plain data augmentation — isolates the objective's contribution).

## 4. Experiments and results

### E-MIN v1 — AKK_300m (weak encoder), cleaned text — 2026-09-02
Head ≈ ridge on clean text (.398 ± .022 vs .376 mc ρ); +.02–.04 under
corruption; name masking hurts nothing. **Negative-ish result**, kept:
`reports/EMIN_RESULT.md`. Diagnosis (Yarin): text already over-cleaned,
encoder too weak.

### E-MIN v2 — tier0, strong encoders, language arms — 2026-09-02
| Akkadian arm, mc ρ | PLS | ridge | ridge-all-views | **head** |
|---|---|---|---|---|
| cuneiformBase-400m L12 | .44 | .45 ± .02 | .46 ± .02 | **.61 ± .01** |
| Llama-2-7B L16 | .22 | .35 ± .02 | .43 ± .01 | **.54 ± .03** |
| Qwen3-8B L18 | .20 | .26 ± .02 | .32 ± .01 | **.43 ± .02** |
Baseline ± from 4 train-subsample refits (C3v2c); head–ridge-all gap = 5–8 combined sd.
Block null .00 ± .02. Head most robust to crop-16 in every Akkadian arm.
Akkadian > gloss with a competent encoder (.55 vs .41 per-language inside the
mixed arm). Head *loses* on the gloss with the Akkadian-only encoder (.39 vs
.42). Full tables `reports/EMIN2_TABLES.md`; narrative `reports/EMIN2_RESULT.md`.

### Robust finding across v1 + v2 (3 encoders, 2 tiers)
**Masking the ruler's name does not reduce dating accuracy for any method**
(often +.01–.05). The representation does not date by the name. → P1 ladder.

### P1 — erasure ladder, frozen probe (C4) — 2026-09-02
Erasing **provenance** alone removes 65–100 % of the linear dating signal
(cunei .45 → .14; Llama .36 → .00; Qwen .26 → −.20). Period −.10…−.20, object
type −.04…−.15, length ≤ .10. Controls (ruler, year-decile) crush the signal.
The Akkadian-native encoder keeps a site-independent residue (.14); the LLMs
keep none. **The representation dates mainly by find-spot, then period — not
by the ruler's name.** `reports/LADDER_RESULT.md`. Head ladder (C5) running.

### P1 — head ladder (C5) — 2026-09-02
Head trained on LEACE-erased features. After erasing **provenance** the ridge
probe reads .14 / .00 / −.20 (cunei / Llama / Qwen) while the head reads
**.36 / .38 / .29** — retaining 59–70 %; margin over the probe grows to
+.22 / +.38 / +.49. Period costs the head −.15…−.25, object type −.03…−.14,
length ≈ 0. The head's chronological signal is not linearly reducible to
find-spot, period, object type or length; the frozen probe's largely is.
Pending: nonlinear-recovery check (can an MLP read provenance from the head's
hidden layer after erasure?). `reports/HEAD_LADDER_RESULT.md`.

### P1 — nonlinear-recovery check — 2026-09-02 (**negative result, important**)
After LEACE, provenance is linearly at chance (.05) but an MLP still reads it from
the erased features (.25 cunei / .35 Llama / .42 Qwen). **The head re-linearises
it on LLM features**: from its hidden layer a *linear* probe reads provenance at
.42 / .46 (raw: .42 / .41). So the head's post-erasure ρ on Llama/Qwen is a
reconstruction of site, not non-site chronology. On cuneiformBase-400m the head
does not reconstruct site (linear .06, MLP .17 < input .25) — the one arm where
the retained .36 survives. → Next ingredient: HSIC/adversarial deconfounding
against provenance in the objective; this probe is its acceptance test.
`reports/HEAD_LADDER_RESULT.md` (last section).

### P2 step 1 — HSIC deconfounding (C6) — 2026-09-02 (**null result**)
λ·HSIC(h, provenance) with λ ∈ {1, 10} changed nothing: provenance from h stays
.43 / .41 / .28 (raw .44 / .42 / .41), ρ unchanged; λ = 1 ≡ λ = 10. Scale problem:
raw RBF-HSIC on a 256-row batch is O(10⁻²) against an O(5) Barlow term. → C6b:
kernel CKA (scale-free, [0, 1]) with λ ∈ {1, 5}, penalty value logged per run.
`reports/HSIC_RESULT.md`.

### Gate reference (P0.4), re-pinned on tier0
cuneiformBase-400m L12 mean, Akkadian: ridge mc .447 ± .060, PLS k=2 .431 ±
.056 (single cross-fit, C3v2 gate). The M.Sc.'s .352 (AKK_300m, maximal, PLS)
does **not** reproduce under the pooled read-out (best .148); its estimator
averaged the 3 of 5 folds where Spearman was defined. To be raised with the
advisors before any thesis text quotes it. `docs/gate_reference.md`.

## 5. Decisions log
| date | decision | reason |
|---|---|---|
| 09-01 | verdict cell fixed a priori (PLS/L8/mean; later per-encoder mid/top layer) | a max over 34–52 cells of ruler-block noise reaches ρ ≈ .7 |
| 09-01 | pooled read-out, ruler-block null | ICC = 1 |
| 09-02 | tier0 Akkadian as base text | maximal already a heavy corruption |
| 09-02 | encoders: cunei400m, Llama-2-7B, Qwen3-8B | AKK_300m too weak; these were the M.Sc.'s best |
| 09-02 | keep duplicates + century docs | sensitivity ≤ .01 |
| 09-02 | per-language read-out `<cond>@<lang>` | gloss vs transliteration was hidden by pooling |
| 09-02 | ladder readability check within train, classes ≥10 docs | across ruler folds the number measured distribution shift (random features read .60) |
| 09-02 | head ladder on LEACE-erased features (C5) | the only way to tell 'head reads site better' from 'head finds non-site chronology' |
| 09-02 | nonlinear-recovery probe is the acceptance test for any deconfounding claim | LEACE is linear; the head re-linearised provenance on LLM features |

## 6. Open / next
CKA deconfounding (C6b) judged by the nonlinear-recovery probe; if it fails too, an adversarial provenance classifier; ladder readability re-pass (5 tables pending);
Assyriologist review of `docs/dating_criteria.md`; P2 factorisation; P3
held-out-ruler calibration.

## 7. Reproducibility
Branch `yarin-sandbox`. Cluster jobs `chrono/sbatch/C*.sbatch`, driven by the
git runner (`chrono/agent/`). Every job pushes its slurm log to
`reports/logs/`; every result row goes to `reports/results.parquet`
(run_id, git_sha, config_sha). Tables regenerate with
`scripts/aggregate_emin2.py`, `scripts/sensitivity_readout.py`.
