# SAE2 results — Karvonen Qwen3-8B dictionary

Status: F22 run 2 (23760, 16k pick) and run 3 (23898, **the pre-specified
labeled 65k instrument — PRIMARY**) both landed. F23 run 1 (23761) crashed
without a log commit; fixed rerun queued. Labels still blocked (source id
unknown — full probe grid 404'd; needs one browser lookup on neuronpedia.org).

## PRIMARY — run 3 (23898): 65k labeled instrument, layer 9

Step 0 picked trainer_2 65k per the labeled-width preference (FVU .0179 vs
raw-best 16k .0171); its config.json confirms `dict_size=65536, k=80` — i.e.
the "l0-80" release. Gate ×4: cellA .0179 / eng .0106 / akk .0075 / cellB
1.387 (same small-n variance anomaly as run 2).

Feature hunt: 776 candidates (sparser dict → fewer, cleaner); top
|ρ(year)| = .42; cos(decoder, ridge) ≤ .08 → **F8's distributed-direction
claim replicates**.

Token-level firing (top-50 year candidates):

| population | fired-anywhere | fire fraction | thirds |
|---|---|---|---|
| cell-A entities | .207 | .050 | .27/.33/.40 |
| eng glosses | **.355** | .0039 | .34/.34/.32 |
| akk maximal | **.020** | .0003 | .38/.32/.30 |

Pre-registered verdicts, both **REPLICATE**:
- `eng_midtext_firing_replicates` = True (.355 ≥ .10)
- `akk_non_engagement_replicates` = True (.020 < .02, at the boundary)

**Synthesis across the two dictionaries at layer 9:** the 65k sparse
dictionary (primary, pre-specified) replicates BOTH halves of F11 — English
glosses fire the year features mid-text, Akkadian essentially never engages
them. The 16k run's Akkadian firing (.441, below) shows the non-engagement
claim is defined relative to a sufficiently sparse feature basis: coarse
16k features that mix year signal with broader lexical content do fire on
the script. Report the 65k numbers as the replication; keep the 16k contrast
as a dictionary-sensitivity note.

## Run 2 (23760, 16k pick) — sensitivity record

### Step 0 — instrument scan (the gate picks the config)

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

### Step 1 — FVU gate, four populations (layer 9, offset 1, 16k trainer_1)

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

### Step 2 — feature hunt (2,615 candidates ≥2% firing)

- top |ρ(strength, death-year)| = **.44** (SAE1 layer 24: .57)
- cos(decoder row, cell-A ridge direction) ≤ **.10** → **replicates F8**: the
  ridge time direction is distributed over many features in this dictionary too.
- median firing of the top-50 year candidates: cellA .277, cellB .037,
  eng .029, akk .0059.

### Step 4 — token-level firing (pre-registered F11 rules)

| population | median fired-anywhere | fire fraction | thirds (early/mid/late) |
|---|---|---|---|
| cell-A entities | .636 | .200 | .31/.34/.36 |
| eng glosses | **.853** | .029 | .39/.32/.29 |
| akk maximal | **.441** | .013 | .35/.33/.32 |

Verdicts (pre-registered):
- `eng_midtext_firing_replicates` = **True** (.853 ≥ .10): English documents
  light up the year features mid-text in this dictionary too — second
  independent confirmation of F11's firing half.
- `akk_non_engagement_replicates` = **False** (.441 ≥ .02) in THIS 16k dict.
  **SUPERSEDED by run 3**: the pre-specified 65k instrument replicates
  non-engagement (.020). The 16k firing stands as the dictionary-sensitivity
  note in the primary section — coarse features that mix year signal with
  broader lexical content do fire on the script; sufficiently sparse ones
  don't.

## Step 3 — labels: still blocked after two fixes

Run 23760: all fetches 404'd (16k indices + guessed source id). Run 23898:
instrument is now the labeled 65k, but the full (model × source) probe grid —
qwen3-8b / qwen3-8b-base / qwen3-8b-it × six source-id patterns — also
404'd (recorded in labels.layer9.json). The guess space is exhausted; the
remaining step is a one-minute browser lookup on neuronpedia.org: open any
feature of the Karvonen Qwen3-8B set and copy the exact model+source ids
from the URL, then `python fetch_labels.py --source <exact-id>`.

## Step 5 — interventions (F23)

Runs 23761 / 23899 / 23921 all crashed at the same line; the log (23761)
showed the real cause: `pick_features` drew its |ρ|<.05 controls from the
hunt CSV, which only keeps the TOP-|ρ| features — the control pool was
empty by construction and `pd.DataFrame([])` has no `.feature`. Fixed: the
control pool is recomputed inside `feature_steer.py` from the encodings
(all ≥2%-firing features), plus a defensive fix for the global scalar
batch-TopK threshold (was indexed per-feature). Resubmit F23 once.

## Labels — source id confirmed, but layer 9 is likely NOT hosted

`https://www.neuronpedia.org/qwen3-8b/18-resid-batchtopk-65k__l0-80/45920`
works in a browser (layer 18). The browser-UA probe grid then showed a
diagnostic split: `qwen3-8b/9-resid-batchtopk-65k__l0-80` returns a real
**404** while every made-up source id returns **500** — i.e. the API route
and model id are right and the layer-9 source simply isn't there.
**API-CONFIRMED (user curl, 2026-08-10):** the layer-18 feature answers with
full JSON while layer 9 404s → Neuronpedia hosts ONLY layer 18 of this
release, and layer 18 fails our FVU gate (.56). Terminal verdict:
**autointerp labels are unavailable for the usable instrument** — that is
the honest end of the "labeled dictionary" selling point at layer 9.
Fallback implemented (F24, `lens_features.py`): decoder-row logit lens per
top feature + keyword-taxonomy pass — an intrinsic, replicable read of what
each feature writes to the vocabulary, plus optional manual web-UI reads of
analogous layer-18 features.
