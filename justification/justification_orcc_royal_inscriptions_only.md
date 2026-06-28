# Justification — Why dating is evaluated only on ORCC royal inscriptions

> **Thesis claim this supports:** "We benchmark chronological dating on the ORCC royal
> inscriptions (~1,200 fragments) and not on the much larger letter corpus, because *only the
> royal inscriptions carry usable temporal labels* — their dates are recoverable from the
> named king. The 4,957 letters have no such anchor."

## 1. The decision, in one sentence

The dating task is supervised by **year/ruler labels that exist only for ORCC royal
inscriptions**, so ORCC is the only corpus on which a dating model can be trained and scored
with ground truth; the letters and SEAL fragments are used elsewhere (validation, qualitative
EDA), not as the labeled dating benchmark.

## 2. The reason: labels come from kings, and only inscriptions name them datably

- The working corpus is ~2M Akkadian words across **4,957 letters, 1,202 ORCC royal
  inscriptions, and 384 SEAL fragments** (`thesis_plan.md:1956`; canonical sizes in memory
  `project_canonical_sizes.md`).
- **"Precise temporal labels (derived from kings' names) are restricted to a fractional
  subset of roughly 1,200 texts"** (`thesis_plan.md:1956`) — i.e. the ORCC royal inscriptions.
  Royal inscriptions are formulaic dedications that name the commissioning king, whose regnal
  dates are externally known, so each fragment can be assigned a year. Administrative/literary
  letters generally do **not** name a datable king, so they cannot be given a ground-truth year.
- This is stated bluntly in the plan: **"Only 1.2k labeled texts"** (`thesis_plan.md:1555,
  2939`) is listed as a core constraint of the project, with the recommended response being
  "convert labels into balanced ordinal constraints" — i.e. exactly the
  [[justification_balanced_mc_protocol]] + ordinal-dating design, run on that 1.2k.

## 3. Why this is a principled choice and not a convenience

- **Train-on-inscriptions / test-on-letters is acknowledged as the *aspirational* transfer
  test, not the core benchmark.** The plan repeatedly lists "Train on royal inscriptions, test
  on letters/fragments where possible, or vice versa" (`thesis_plan.md:1196, 1851, 2527,
  3204`) — but always hedged with *"where possible."* It is a generalisation probe gated on
  having labels, not the supervised benchmark itself.
- **The letters already played their role earlier in the pipeline.** In Round-1 probing the
  letters were the *easy* case (99.1% on the letter task, `thesis_state.md` Phase 4); the
  scientific tension — and therefore the dating benchmark — lives in ORCC, where the same
  probes *failed*, which is what the whole geodesic/maximal/finetune arc investigates.
- **SEAL (384 fragments)** is too small and is used for the cross-corpus metadata study
  (`justification/seal_round4_pipeline_plan.md`), not as the dating benchmark.

## 4. The cost we accept, and how we mitigate it

Restricting to ~1,200 labeled royal inscriptions is the project's central data-scarcity
bottleneck (`thesis_plan.md:1956, 1969`). We mitigate it rather than ignore it:

- **GroupKFold by ruler + name-masking** so the small labeled set can't be solved by
  king-name memorisation (see [[justification_pls_regression]],
  [[justification_balanced_mc_protocol]]).
- **The unlabeled-manifold question** is posed explicitly: *"Is the chronological signal
  already a global manifold in the unlabeled corpus, or only a supervised artifact visible in
  the labeled set?"* (`thesis_plan.md:1613, 2975`) — i.e. the geodesic phase uses the large
  unlabeled corpus geometrically precisely *because* only ORCC is labeled.

## 5. Supporting literature

- **Gurnee & Tegmark — "Language Models Represent Space and Time"**
  (`papers/txt/Geometric Representation papers/`). Their temporal probes are trained on
  entities with *known* dates and then read out; the requirement for ground-truth temporal
  anchors to supervise a dating/space-time probe is the same constraint that forces us onto
  the king-dated inscriptions. **[supporting — establishes that temporal readouts need dated
  anchors.]**
- **Fetaya et al. — "Filling the Gaps in Ancient Akkadian Texts"**
  (`papers/txt/Ancient Language papers/`). Same Akkadian setting; documents the extreme
  label/data scarcity that defines what is and isn't a usable supervised target here.
  **[supporting — domain data-scarcity context.]**

## 6. Numbers & sources to pull when writing

- Corpus sizes & label restriction: `thesis_plan.md:1956`; memory `project_canonical_sizes.md`
  (letters = 4,957 · ORCC = 1,202 / 893 labeled · SEAL = 384).
- ORCC build + label derivation: `v_1/src/corpus/03_build_orcc_corpus.py`.
- "Only 1.2k labeled texts" constraint: `thesis_plan.md:1555, 2939`.

> **Note for write-up:** memory records the ORCC labeled count as **893** (of 1,202) in
> `project_canonical_sizes.md`, while the plan rounds to "~1,200" / "1.2k." Quote **893
> labeled of 1,202** as the precise figure and reserve "~1,200" for prose.
