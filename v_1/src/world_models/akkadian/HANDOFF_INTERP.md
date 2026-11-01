# HANDOFF — the interpretability wave, for the Overleaf paper agent

You are writing the thesis paper (repo `YarinBeni/Yarin-Thesis`, working
branch `claude/thesis-latex-build-qc28vp`, file `main.tex`). This document
hands you every experiment run in the interpretability wave (Oct 2026),
with the numbers, the figure files, and exactly where each asset lives.
Write measurements as claims; keep mechanisms and speculation in the
discussion.

## Context paper

**NLMCD** = arXiv:2610.01821, https://arxiv.org/html/2610.01821v1 — the
document-manifold concept-discovery paper. It finds emergent nonlinear
manifolds in LLM document-embedding spaces (UMAP to low dimension +
HDBSCAN clustering) and interprets the manifolds as concepts. Cite it
wherever this handoff says "NLMCD-inspired": Experiment B replicates its
manifold analysis on the Akkadian corpus, and the atlas figures
(Experiment C) follow its figure programme. When Yarin says "the manifold
paper", he means this one.

## Scope

This handoff covers the work that started from Yarin's 2026-10-06
question: "did we run entity in Akkadian and not English? maybe the cliff
is language AND text — two cliffs?" plus the NLMCD reading. It is the
WB-akk entity experiment, the manifold analysis, the alignment atlas, and
all their controls. The earlier SSL / chrono-rank representation-learning
programme is OUT of scope for this handoff — do not pull its numbers.

## 0. Repo map (checked 2026-11-01 — no duplicate repos)

| repo | visibility | role |
|---|---|---|
| `YarinBeni/Yarin-Thesis` | private | THE PAPER: main.tex, drafts, figs/. Branches: `main` + your working branch `claude/thesis-latex-build-qc28vp` — only these two exist, no duplicates. |
| `YarinBeni/HUJI-THESIS--YARIN`, branch `main` | public | EVERYTHING: the M.Sc. code, the WB-akk experiment, and the full interpretability wave (all `chrono/` scripts and reports). `yarin-sandbox` was merged into `main` on 2026-11-01 (merge `05a7ad33`); read every path below from `main`. |

The old **** / **** distinction is gone: every path in this
document is on `main` of `YarinBeni/HUJI-THESIS--YARIN`. (`yarin-sandbox`
still exists as the historical working branch; ignore it.)

## 1. Background the paper already has

LLMs date FAMOUS entities and English headline text at rho ~ .9/.79
(Gurnee–Tegmark replication). The thesis corpus is 1,193 dated Akkadian
royal inscriptions (40 rulers), where LLM document dating is weak and the
domain encoder cuneiformBase-400m wins. Everything below asks WHY.

## 2. Experiment A — the language cliff at the name level (WB-akk)

**What we did.** Same rulers, same carrier templates, same ruler-grouped
split; the only change is the name's form: "Esarhaddon" vs the attested
transliterated spelling "m-aš-šur-PAP-AŠ", harvested from the ruler's own
inscriptions. `eng25` re-scores English on exactly the 21 rulers shared
with the Akkadian set, so eng25-vs-akk is language-only. Probes: frozen
activations, ridge, Monte-Carlo entity splits, Spearman rho, best layer.

**Headline table** (bare name, ent_last; source of truth
`v_1/src/world_models/akkadian/results/LANGUAGE_CLIFF.md`):

| model | eng (40) | eng25 (21 matched) | akk name | cliff (eng25−akk) |
|---|---|---|---|---|
| Llama-2-7B | .527 | .519 | .138 | .38 |
| Qwen3-8B | .596 | .684 | .252 | .43 |
| OLMo-2-7B | .526 | .643 | .129 | .51 |
| gpt-oss-120B | .663 | .526 | .265 | .26 |
| cuneiformBase-400m | .456 | .415 | .451 | **−.04 (no cliff)** |

**Backed claims.** Four LLM families lose most of their entity-dating
signal when the name is written in its own script; 120B parameters do not
remove the cliff; the domain encoder is orthography-invariant at a lower
ceiling. Pooling nuance (keep it): with mean pooling over the name span
(`ent_mean`) the akk numbers partially recover (Llama .36, Qwen .44,
gpt-oss .39, cunei .51) — part of the cliff is read-out at the last token
of a long transliterated name, not missing knowledge.

**Harvester validation (advisor-demanded).** The chosen spellings identify
the correct king in 96.0% of 476 documents; leakage is genealogy; one enemy
name (Teumman) was caught by the test. Report:
`v_1/src/world_models/akkadian/results/VALIDATE_ENTITY_AKK.md`; figure:
`results/validate_histograms.png`. Full per-site tables:
`results/RESULTS_entity.md` (includes the final gpt-oss eng25 row).

**Caveats to print.** 25 rulers with surviving spellings (21 matched); MC
sd in akk cells ±.4–.5 (ranking stable, magnitudes not); R² negative
throughout (tiny n, high dim) so rho is the statistic; famous-vs-obscure
and name-vs-document comparisons cross corpora/protocols — say so.

## 3. Experiment B — document manifolds (NLMCD-inspired, arXiv:2610.01821)

**What we did.** UMAP (8d) + HDBSCAN over embeddings of 30,729 undated
documents, per model/layer; ask what the emergent clusters correspond to.

**Results.** cunei400m L12: 22 manifolds, all explained by corpus×genre;
Qwen: 8. One mixed eBL cluster carries real internal period structure
(silhouette +0.152 vs permutation null −.001, p=0.00). Script
`chrono/scripts/nlmcd_concepts.py`; outputs in
`chrono/reports/`.

**Backed claim.** Unsupervised document geometry organizes by
corpus and genre, not by time.

## 4. Experiment C — the alignment atlas (the good figures)

All under `chrono/reports/tier0/atlas/`; script
`chrono/scripts/alignment_atlas.py`. 1,193 documents embedded twice
(Akkadian + English translation), mean pooling, per layer.

- **Fig B** `B_language_alignment.png` (+ `.csv`): ARI between the two
  languages' document clusterings, per layer. Peaks: Llama .90 @L20,
  OLMo .87 @L9, Qwen .53 @L27, cunei .31 @L4. Right panel: century
  silhouettes are negative at every layer of every model — clusters are
  never time-shaped. CAVEAT: curves are single-seed and jagged;
  seed-averaging is still pending — do not read small wiggles.
- **Fig A** `A_layer_alignment_<model>.png`: layer×layer ARI within a
  model/language. Llama's Akkadian clustering freezes from L6 to L32
  (ARI .85–1.0); the English side keeps reorganizing — the Fig B rise is
  the English geometry converging onto the frozen Akkadian one.
- **Fig C** `C_time_depth.png` (+ `.csv`): ridge probe rho per layer and
  language. cunei peaks early (akk .35 @L9–12). LLM Akkadian curves fade
  after mid-depth; their English curves persist to the last layer
  (Llama-eng .30 @L32). Entity overlays included.
- **Fig D** `D_entity_maps.png`: UMAP of the name activations colored by
  year — a visible year gradient in English (clearest in Llama), shuffled
  colors in Akkadian, cunei weakly ordered in both. This is the picture of
  the name-level cliff.

## 5. Experiment D — the controls (what the alignment is NOT)

Reports in `chrono/reports/tier0/atlas/`:

| control | file | result |
|---|---|---|
| chance (20 shuffled pairings) | `B_controls.md` | null ≈ .000 ± .02 |
| length could fake it? | `B_controls.md` | length-bins alone align at .65 — a real threat |
| do the models use length? | `B_controls.md` | ARI(real clusters, length bins) = .002–.17 — no |
| bin-count sensitivity | `CONTROLS4.md` | label-level control swings .47–.81 with q=16/8/4 — hence the next row |
| model-level length ('aaa' texts through the encoder) | `CONTROLS4.md` | OLMo −.01, Qwen .03, Llama .32, cunei .21 |
| names masked, full text | `CONTROLS3.md` (V1b) | Llama .90→.86, OLMo .88→.84 (survive); Qwen .53→.13 (name-carried); cunei .31→.01 |
| names masked + crop to 32 words | `CONTROLS2.md` (V1) | all LLMs collapse (.07/.12/−.01); cunei keeps .26 |

**Backed claim (the paper's best-defended).** Mid-layer Llama-2 and OLMo-2
assign the same documents to the same clusters in Akkadian and English
(ARI .84–.86 with royal names masked), and this is not chance, not
document length, and not the names: it is distributed document content.
Qwen's weaker alignment is name-carried. Make no cunei alignment claim:
its numbers are incoherent across conditions (single-seed instability).

## 6. Experiment E — is time categories or order? (`CONTROLS2.md` V3, `CONTROLS3.md` V3b)

ARI(document clusters, century bins) ≈ 0 for every model; yet within-century
ridge probes order documents by year at rho .62–.71. Strong cell: 6th
century BC — 728 docs, 17 distinct years, majority-year share 37%,
rho .61–.71. The 7th/5th-century cells are thin (8/5 distinct years, 60%/51%
majority) — supporting only. **Claim:** time in these representations is a
continuous direction read by regression, not a category structure visible
in clustering (consistent with the all-negative silhouettes).

## 7. Experiment F — small robustness results

- Target convention (`CONTROLS2.md` V4): first/median/last attested year
  as probe target changes rho by ≤.01 — one methods sentence.
- Alignment vs dating skill: across 52 model-layer points, Fig B ARI does
  not correlate with probe rho (Spearman ≈ 0); cunei has the LOWEST
  alignment and the HIGHEST Akkadian probe rho. Alignment and dating are
  different properties.

## 8. Figures: use / don't use

USE: the four atlas figures (Section 4), `validate_histograms.png`,
`D_entity_maps.png`. Rebuild Fig B only after seed-averaging if time allows.
DO NOT USE: `results/cliff_schema.png` and `results/cliff_lattice.png`
 — superseded drafts of the fame×language×form summary; Yarin has
not approved a final design for that figure yet. Leave a placeholder.

## 9. Pooling (for the methods section)

Entity probes report four read-outs: `ent_last` (last token of the name),
`ent_mean` (mean over name tokens), `last`, `mean` (whole input). Headline
entity numbers are `ent_last`; in the `bare` template ent_last==last and
ent_mean==mean. ALL document-level analyses (Sections 3–6) use mean
pooling over document tokens (BOS excluded); a `last`-pooling replication
of the atlas was not run (deep layers were extracted mean-only).

## 10. Still open (do not wait for these)

seed-averaged Fig B; `last`-site atlas replication; the final
fame×language×form summary figure design.
