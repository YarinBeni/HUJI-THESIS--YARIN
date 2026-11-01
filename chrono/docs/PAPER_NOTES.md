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

**Data bug found 2026-09-02 (S0 census):** the unified corpus has 574,744 exact
duplicate ORACC word rows; the true size is ≈ 1.88 M words / 40,429 texts, not the
2.45 M quoted in the M.Sc. data notes. Corrected in the SSL corpus build.

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

### P2 step 1b — CKA (C6b) — null; deconfounding line CLOSED (2026-09-02 evening)
CKA λ 1/5 drove the batch statistic to ≈ .1 yet a linear probe still reads
provenance from h at .4; ρ dropped .61 → .58/.54. An adversary prototype
collapsed. **Decision (Yarin):** the method is text-view invariance
(SSL/contrastive); forcing a metadata variable out of the head is not the
design. Find-spot stays as a *stated limitation* in the paper, the penalties as
a failed side experiment. Possible future SSL-consistent form: a `mask_place`
text view. `reports/HSIC_RESULT.md`.

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
| 09-02 | **deconfounding penalties dropped; method = text-view invariance only** | Yarin: interpretation question ≠ training objective; HSIC/CKA null, adversary unstable |

## 6. Open / next

**Direction change (2026-09-02, PI):** the method is augmentation invariance;
the deconfounding penalties are parked. Next programme: SSL pretraining on all
~40k Akkadian texts + a scaling sweep — see `docs/PLAN_SCALE_SSL.md`.

(deconfounding line closed — see above); Assyriologist review of `docs/dating_criteria.md`; ladder readability re-pass (5 tables pending);
Assyriologist review of `docs/dating_criteria.md`; P2 factorisation; P3
held-out-ruler calibration.

## 7. Reproducibility
Branch `yarin-sandbox`. Cluster jobs `chrono/sbatch/C*.sbatch`, driven by the
git runner (`chrono/agent/`). Every job pushes its slurm log to
`reports/logs/`; every result row goes to `reports/results.parquet`
(run_id, git_sha, config_sha). Tables regenerate with
`scripts/aggregate_emin2.py`, `scripts/sensitivity_readout.py`.

## Related work — the erasure family and where we sit (2026-09-04)

Three generations, mapped onto our problem (source = the concept to erase):

1. **Linear, closed form.** INLP -> RLACE -> LEACE (guarantee: no linear
   probe can read the concept). Our LEACE-in-the-loop arm. Verified live:
   linear source probe .97 -> .21, but an MLP still reads .88 — the linear
   guarantee is real and insufficient.
2. **Nonlinear by density matching.** KRaM, LEOPARD (ECAI 2025): learned
   rank-r orthogonal projection + MMD between class-conditional densities.
   Our `leopard` arm is this, trained jointly with the SSL objective.
3. **Unlearning-flavoured, 2026.** Double Projections (arXiv 2604.10032):
   two closed-form projections, the second constrained to the left
   nullspace of representations to PRESERVE — built for erasing a target
   concept from a generative model while protecting neighbours. SCOPE
   (arXiv 2608.02058): input-conditional gating of the projection, and an
   "entanglement frontier" — a proven limit on how much a FIXED projection
   can erase without destroying retained information.

Two things transfer to us; the machinery mostly does not (their setting is
"erase concept X, keep siblings", ours is invariance to an always-present
attribute, and their protect-lists need labels for what to retain, which SSL
does not have):

- **The entanglement frontier names what we measured.** Erasing source
  linearly dropped the linear PERIOD probe from .90 to .63 — period and
  source are entangled in this corpus, and the retain-cost of erasure is
  visible in one number. Good framing for the paper.
- **If every fixed-projection arm fails the C18 gate,** the SCOPE-style move
  — a projection gated per input — is the natural next arm, since a fixed
  projector provably cannot cross the frontier.

## The success criterion, stated once (advisor, 2026-09-04)

A run counts as a win only if all three hold on the final h:

1. **linear source probe ≈ chance** (.20) — the linear trace is gone;
2. **MLP source probe ≈ chance** — the nonlinear trace is gone too;
3. **C18 Spearman ≥ the frozen baseline** (gkf .419) — and the interesting
   regime is toward the supervised head's .61.

Anything that clears 1-2 but collapses 3 has erased the concept together
with the signal (the entanglement frontier made real); anything that clears
3 but not 1-2 is dating through the corpus shortcut again.

## The paper's arc (advisor's framing)

representation learning (Barlow/JEPA/BYOL/InfoNCE over 31,905 unlabelled
Akkadian texts) -> interpretability of what was actually learned (probe
battery: source .92-.98, period ≈ source, the shortcut made visible) ->
controlling what is learned (erasure in-training: LEACE-in-the-loop,
adversarial, density matching; and post-hoc, the unlearning family). Each
stage's failures are findings: the adversary is fooled only by its own
co-trained probe (.36 in-training, .96 post-hoc); the linear eraser leaves
the MLP trace (.21 linear vs .88 MLP).
