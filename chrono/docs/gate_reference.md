# P0.4 gate — where the reference number comes from

Written BEFORE C2 produced any number, so the gate is judged against a
stated expectation rather than against its own output.

## The plan's 0.41 is not usable

The plan addendum already flagged that `Thalesian L11 rho = 0.41 ± 0.02`
was pinned on the 1,202-fragment / 41-ruler frame. Two further problems
surfaced while preparing C1/C2:

1. **L11 does not exist in this encoder.** `Thalesian/AKK_300m` has 8
   encoder blocks; with the embedding layer that is 9 hidden states,
   indices 0..8. L11 can only have come from `cuneiformBase-400m` (12
   blocks) or from a transcription slip.
2. **No cell of the M.Sc. sweep reaches 0.41.** The highest mean-pooled
   PLS Spearman anywhere in the grid is 0.352.

So C2 stays UNPINNED. This file records what we should expect instead.

## What the M.Sc. actually measured

`v_1/src/linear_probing/results/orcc__probe_pls/pls_results_thalesian_akk300m.json`
(maximal cleaning, year-raw, n = 1,193 labeled / 40 ruler groups — within
6 documents of the chrono frame of 1,187 / 40, so the frames are
comparable). Spearman by layer, mean pooling:

| layer | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| L00 | .235 | .219 | .204 | .102 |
| L01 | .206 | .307 | .221 | .217 |
| L02 | .195 | .314 | .232 | .275 |
| L03 | .199 | .312 | .275 | .281 |
| L04 | .211 | .320 | .295 | .241 |
| L05 | .194 | .304 | .281 | .268 |
| L06 | .204 | .306 | .313 | .330 |
| L07 | .166 | .286 | .314 | .306 |
| **L08** | .196 | **.352** | .336 | .289 |

Last-token pooling is worse at every layer (max .265), which is why the
a-priori site is `mean`.

Two consequences, both applied to `run_baseline_gate.py`:

* **a-priori layer = L8** — the top encoder block, and the M.Sc.'s best
  mean-pooled cell.
* **a-priori PLS components = 2** — the selected k at 7 of the 9
  mean-pooled layers. k=5 already has negative R² at L8, so the previous
  default of 20 would have overfitted ~40 exchangeable ruler blocks and
  failed the gate for a reason unrelated to the representation.

## Why C2's number will not equal .352 even if nothing is wrong

The M.Sc. figure is a **mean of per-fold Spearman over the folds where it
was defined — 3 of 5**. The other two folds returned NaN: with ICC = 1
(39 of 40 rulers carry exactly one distinct year) a held-out fold can
contain a constant target, and Spearman is undefined there. Silently
averaging the surviving folds is exactly the read-out SLA section 7
forbids; C2 uses the **pooled** read-out over cross-fitted out-of-fold
scores, plus a block null.

These are different estimators of different quantities. The pooled
estimate uses all 40 rulers instead of the subset that happened to give a
defined fold statistic, so it should be the more stable of the two, and
there is no reason for it to land on .352 exactly.

## The expectation, stated in advance

* **Consistent with the M.Sc.:** pooled ρ at PLS / L8 / mean in roughly
  **.25–.40**, with the block null centred near 0 and the doc placebo
  near 0.
* **Investigate:** pooled ρ below ~.15 (something broke between the M.Sc.
  activations and the chrono EmbStore — sign convention, text variant,
  fold assignment) **or** a block null far from 0 (ruler leakage).
* **Do not** re-pin `--gate-rho` from C2's own output and then re-run to
  get a PASS. If C2 lands in the expected band, the gate is reproduced;
  record the pooled value as the reference for later phases and say that
  is what it is.

## C2 outcome (job 33316, 2026-09-02) — read against the band above

Pooled mc ρ, Akkadian (`akk`), mean pooling:

| layer | ridge | PLS k=2 |
|---|---|---|
| L2 | .274 | .124 |
| L3 | .295 | .103 |
| L4 | .276 | .134 |
| L6 | .273 | .147 |
| L7 | .293 | .140 |
| **L8** | **.287** | **.126** ← a-priori cell |

Doc placebo ≈ 0 and **ruler-block null ≈ 0 (±.13–.16) in every cell** — no
leakage, no fold artefact. L0/last skipped (constant `</s>` vector).

English gloss (`eng`): ridge .39–.42 at L5–L8, PLS .35–.43 at L4–L7, PLS L8
.154. The gloss dates *better* than the Akkadian on an Akkadian-trained
encoder — same asymmetry the M.Sc. saw between cells B′ and C.

**Reading.**
* **Pipeline reproduces the M.Sc. signal**: ridge sits in the pre-stated
  .25–.40 band at every mid/late layer, with clean nulls. Extraction,
  splits, read-out and null machinery are sound. C3 is unblocked.
* **The a-priori cell itself (PLS k=2, L8, mean) is below the .15
  "investigate" line at .126.** Not re-picked. Likely cause, found by
  reading the M.Sc. probe code after the fact: the M.Sc. **row-L2-normalised**
  every vector before PLS (`pls_utils.l2_normalize`); C2 column-standardised
  only. T5's final layer norm leaves a few outlier dimensions that a
  2-component PLS latches onto; ridge's shrinkage does not care, which is
  exactly the ridge-vs-PLS gap in the table. A like-for-like rerun with
  `--row-l2` is queued (reports `*_rowl2.txt`). Whatever it shows is
  recorded here; the a-priori cell stays PLS/L8/mean either way.
* The plan's "0.41" is now doubly unusable: no Akkadian cell reaches it
  under either estimator; only the *English gloss* does.

### Row-L2 rerun (job 33317)

| cell | plain | row-L2 |
|---|---|---|
| akk PLS k=2 L8 mean (a-priori) | .126 | **.148** |
| akk ridge L8 mean | .287 | .296 |
| eng PLS k=2 L8 mean | .154 | **−.027** |
| eng ridge L8 mean | .392 | .333 |

Row-L2 does **not** recover the M.Sc.'s .352. The pre-stated explanation
(outlier dimensions) was wrong, or at most a small part. What is left is the
estimator: the M.Sc. figure is a mean over the 3 of 5 folds where Spearman
was defined; the pooled read-out uses all 40 rulers. Under the honest
estimator a 2-component PLS at the top layer is simply weak and unstable
(it even flips sign on the gloss), while ridge is steady at .27–.30 (akk)
across every mid/late layer with nulls at zero.

**Standing decision.** The a-priori cell stays PLS/L8/mean and is recorded
as *below band*. The gate is nevertheless read as *reproduced*, on the
ridge rows and the clean nulls, and C3 proceeds on L8/mean features. The
discrepancy is a finding about the M.Sc. read-out, to be raised with the
advisors before the thesis text quotes .352 anywhere.
