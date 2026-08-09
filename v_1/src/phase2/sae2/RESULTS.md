# SAE2 results — Karvonen Qwen3-8B dictionary

Status: F22 run 2 (job 23760) landed; F23 interventions (23761) pending;
labeled-65k rerun queued (see README progress).

## Step 0 — instrument scan (the gate picks the config)

38 files, layers {9, 18, 27}, 4 trainers/layer × offsets {0, 1}, all scored by
FVU on cell-A last-token activations:

| layer | trainer (width) | offset 1 FVU |
|---|---|---|
| 9 | trainer_1 (16k) | **.0171** (raw best) |
| 9 | trainer_2 (65k) | .0179 (labeled-width pick after fix) |
| 9 | trainer_3 (65k) | .0185 |
| 9 | trainer_0 (16k) | .0197 |
| 18 | best (16k t1) | .5608 — **gate FAIL**; others 3.7–127 |
| 27 | best (16k t1) | .5000 — **gate FAIL**; others .82–8.9 |

**Only layer 9 is usable.** Deep-layer configs of this release do not
reconstruct our last-token vectors at all (FVU up to 127 — consistent with
Qwen3's exploding deep-layer residual norms / massive activations). So SAE2
is an EARLY-layer instrument, complementary to Qwen-Scope's layer 24 (F7–F8,
F11): the two dictionaries look at opposite ends of the stack.

## Step 1 — FVU gate, four populations (layer 9, offset 1, 16k trainer_1)

| population | FVU |
|---|---|
| cell-A entities | .0171 ✅ |
| eng tier-0 glosses | .0110 ✅ |
| akk maximal | .0077 ✅ |
| cell-B ruler names | **1.39 ⚠️** |

Cell B is the anomaly: reconstruction "worse than the mean" on a small
(n≈204), low-variance population — the FVU denominator (within-population
variance) is tiny for near-duplicate royal-name prompts, so this is flagged
as a population-statistics artifact to verify, not read as "SAE broken on
rulers" (absolute error is not comparable across populations).

## Step 2 — feature hunt (2,615 candidates ≥2% firing)

- top |ρ(strength, death-year)| = **.44** (SAE1 layer 24: .57)
- cos(decoder row, cell-A ridge direction) ≤ **.10** → **replicates F8**: the
  ridge time direction is distributed over many features in this dictionary too.
- median firing of the top-50 year candidates: cellA .277, cellB .037,
  eng .029, akk .0059.

## Step 4 — token-level firing (pre-registered F11 rules)

| population | median fired-anywhere | fire fraction | thirds (early/mid/late) |
|---|---|---|---|
| cell-A entities | .636 | .200 | .31/.34/.36 |
| eng glosses | **.853** | .029 | .39/.32/.29 |
| akk maximal | **.441** | .013 | .35/.33/.32 |

Verdicts (pre-registered):
- `eng_midtext_firing_replicates` = **True** (.853 ≥ .10): English documents
  light up the year features mid-text in this dictionary too — second
  independent confirmation of F11's firing half.
- `akk_non_engagement_replicates` = **False** (.441 ≥ .02): at layer 9 the
  year-feature candidates DO engage on Akkadian (44% of fragments, ~1.3% of
  tokens). Honest synthesis: **F11's "never engages on Akkadian" is a
  deep-layer (L24) phenomenon, not universal** — early lexical features fire
  on the script, and the non-engagement emerges by the late layers where the
  entity-gated time machinery lives. This sharpens, not weakens, the story:
  the Akkadian signal dies between the early lexical stage and the late
  entity-time stage.

## Step 3 — labels: FAILED in run 23760, fixed

All 50 Neuronpedia fetches 404'd. Root causes: (1) the scan had picked the
16k trainer whose indices don't exist in Neuronpedia's 65k source; (2) the
source id `9-resid-batchtopk-65k` was a guess. Fix: labeled-width preference
in step 0 (picks trainer_2 65k at FVU .0179) + a (model, source) probe grid
in `fetch_labels.py` that records what actually answers.

## Step 5 — interventions (F23)

Pending (job 23761, running on the 16k pick; valid without labels).
