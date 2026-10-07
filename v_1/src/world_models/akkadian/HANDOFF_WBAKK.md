# HANDOFF — WB-akk: the language cliff at the name level

You are editing (a) the HTML presentation and (b) the Overleaf paper for
Yarin Benizri's M.Sc. thesis. This document hands you one NEW experiment and
one NEW figure. Insert them into the existing narrative exactly where
described. Do not overclaim: every number here is a measurement; keep
mechanisms and speculation in the discussion section only.

## 1. Where this goes in the story

The existing narrative ladder is:

1. Replication of Gurnee & Tegmark: LLMs date FAMOUS entities and headline
   text in English very well (rho ~ .9 / ~.79).
2. Our move to OBSCURE entities (Assyrian/Babylonian rulers, English
   spelling): rho drops to ~.66. That is the FAME cliff.
3. Our move to full ANCIENT DOCUMENTS: LLMs fail (~.26-.33); the domain
   encoder cunei400m wins on Akkadian documents.

**The new experiment slots between (2) and (3).** It asks: when we go from
"Esarhaddon" (English) to the ruler's own attested transliterated spelling
("m-aš-šur-PAP-AŠ"), holding EVERYTHING else fixed — same rulers, same
carrier templates, same train/test split — how much is lost to the language
of the name alone? This isolates the LANGUAGE cliff from the fame cliff and
from the name-to-document (granularity) cliff.

## 2. Method (3 sentences for the paper; expand from here if needed)

Attested royal-name spellings were harvested by aligning each inscription's
Akkadian text against its [PN]-masked tier; only the FIRST personal name of
each document is a candidate (royal inscriptions open with the royal
titulary), with a dominance filter (>=70% of a spelling's documents must
belong to one ruler). The harvest was validated, not assumed: the chosen
spellings identify the correct king in 96.0% of 476 test documents
(E1-E4 in `results/VALIDATE_ENTITY_AKK.md`; histogram figure
`results/validate_histograms.png`). Probes are identical to the English
entity protocol: frozen activations, ridge, Monte-Carlo entity splits,
Spearman rho at the best layer; `eng25` re-scores the English probe on
exactly the 21 rulers shared with the Akkadian set, so the eng25-vs-akk
comparison is language-only.

## 3. The headline table (MC Spearman rho, bare name, ent_last, best layer)

| model | eng (40 rulers) | eng25 (matched 21) | Akkadian name | cliff |
|---|---|---|---|---|
| Llama-2-7B | .527 | .519 | .138 | .38 |
| Qwen3-8B | .596 | .684 | .252 | .43 |
| OLMo-2-7B | .526 | .643 | .129 | .51 |
| gpt-oss-120B | .663 | (pending, CPU re-probe) | .265 | .40 (vs eng) |
| cuneiformBase-400m | .456 | .415 | .451 | **-.04 (none)** |

Sentences you may write as claims (they are backed):
- "Four independent LLM families lose 60-80% of their entity-dating signal
  when the same ruler's name is written in his own script."
- "A 120B-parameter model pays the same language cliff as 7B models: scale
  does not buy script-invariant historical knowledge."
- "The domain encoder has no language cliff (.456 -> .451): its (lower)
  knowledge of the rulers is orthography-invariant."
- Pooling nuance (one sentence, do not drop it): with mean pooling over the
  name span (`ent_mean`) the Akkadian numbers partially recover (Llama .36,
  Qwen .44, gpt-oss .39, cunei .51) — part of the cliff is read-out at the
  last token of a long transliterated name, not missing knowledge. The
  cliff narrows but survives.

Caveats that MUST appear near the table: only 25 rulers carry a surviving
spelling (21 shared with the English split); MC sd in the Akkadian cells is
large (±.4-.5) — the ranking is stable, exact magnitudes are not; R² is
negative throughout (high-dim ridge, tiny n) so rho is the reported
statistic; ~1 residual non-royal spelling was caught and removed by the
validation (Teumman under Ashurbanipal).

## 4. The new figure: `results/cliff_lattice.png`

This replaces any earlier "cliff ladder/tree" sketch. How to read it (put
this logic in the slide's speaker notes and the paper's caption):

- The design is a 2x2x2 factor cube: FAME (famous/obscure) x LANGUAGE
  (English/Akkadian) x GRANULARITY (name/full text). The score (Spearman
  rho) is the 4th dimension, drawn as HEIGHT.
- x-axis = how many factors have been flipped away from the LLM comfort
  zone (famous-English-name). The cube is flattened into this lattice; the
  two famous-Akkadian cells do not exist (no Akkadian corpus for Cleopatra)
  and are honestly absent.
- Every edge is a SINGLE-factor move, colored by which factor moved
  (amber = fame, red = language, blue = granularity). The slope of an edge
  IS that cliff's price at that position.
- Filled dot = best LLM in that cell; open dots = each LLM.
- The figure's one-line takeaway for the slide: **"Each cliff is paid
  once."** From obscure-English-name (.66), flipping language costs -.40
  and flipping granularity costs -.34 — but after either, the other flip
  costs almost nothing (-.07 / -.01). All hard routes converge on the same
  floor (rho ~ .26), which is where the real Akkadian documents live.
- Cell values (best LLM): famous-name-EN .90, famous-text-EN .79,
  obscure-name-EN .66, obscure-text-EN .33, obscure-name-AKK .27,
  obscure-text-AKK .26.
- Protocol footnotes already under the figure (keep them): name cells are
  entity probes; text cells are ridge probes on mean-pooled document
  embeddings and exclude gpt-oss; famous-text is NYT-headline dating —
  a different corpus and time scale, comparable in spirit, not protocol;
  cunei400m is excluded from this figure because it has no language cliff.

## 5. Assets (all on branch `main`)

- `v_1/src/world_models/akkadian/results/cliff_lattice.png` — the figure.
- `v_1/src/world_models/akkadian/results/LANGUAGE_CLIFF.md` — distilled
  result + caveats (source of truth for the table above).
- `v_1/src/world_models/akkadian/results/VALIDATE_ENTITY_AKK.md` and
  `validate_histograms.png` — harvester validation (96% ruler ID).
- `v_1/src/world_models/akkadian/results/RESULTS_entity.md` — full
  per-model, per-pooling-site tables (`ent_last`, `ent_mean`, `last`,
  `mean`), datasets `assyrian_ruler`, `assyrian_ruler_akk`,
  `assyrian_ruler_eng25`.
- `v_1/src/world_models/akkadian/build_entity_akk.py`,
  `validate_entity_akk.py` — the harvester and its validation code.
- `v_1/src/world_models/data/entity_datasets/assyrian_ruler_akk.csv` — the
  dataset (same schema as the English one).

## 6. What NOT to write

- Do not say the LLMs "cannot read" Akkadian — the separate alignment
  analysis shows their document geometry is largely language-shared; the
  failure is access to entity knowledge from the transliterated surface
  form, plus a read-out effect (see the pooling nuance).
- Do not average the cliff across models into one number; report per-model.
- Do not use the eng25 gpt-oss cell until it lands; the table marks it
  pending.
