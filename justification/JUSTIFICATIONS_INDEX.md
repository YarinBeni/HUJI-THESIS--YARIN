# Justifications index — methodology decisions, evidence-backed

Purpose: a thesis-writing aid. Each file below pins **one methodological decision** to **our
own committed results/figures** plus the **papers** that support it, so a thesis paragraph can
be pulled directly. Every claim cites a file path or a paper in `papers/`. Where a paper link
is the *literal* precedent it is marked **[direct]**; where it is a methodological analogue it
is marked **[analogous/supporting]**.

## Decision dossiers (this round)

| File | Decision justified | Headline evidence |
|---|---|---|
| [justification_maximal_cleaning_regime.md](justification_maximal_cleaning_regime.md) | Why the **maximal** regime (clean_maximal 11 filters + ≤32-word truncation) is the honest benchmark | TF-IDF year-Spearman collapses 0.474→0.245 under truncation while Thalesian holds 0.430→0.407 (`T_headlines.csv`, `T1_year_pls_maximal.csv`) |
| [justification_balanced_mc_protocol.md](justification_balanced_mc_protocol.md) | Why **class-balanced, 200-MC-draw** scoring (8 rulers) | 38-class ruler dist. is extreme (620 vs 25 frags); every headline carries ±std over 200 draws (`scoreboard.md`) |
| [justification_orcc_royal_inscriptions_only.md](justification_orcc_royal_inscriptions_only.md) | Why dating is benchmarked **only on ORCC royal inscriptions** | Temporal labels (from kings' names) exist only for ~893/1,202 ORCC; letters have no datable anchor (`thesis_plan.md:1956`) |
| [justification_spearman_metric.md](justification_spearman_metric.md) | Why **Spearman ρ** is the headline metric | Ordinal task + scale/dimensionality invariance across 8 heterogeneous models; reported with shuffled-year null |
| [justification_pls_regression.md](justification_pls_regression.md) | Why **PLS** is the readout head | Supervised low-rank projection onto year-covarying subspace; normalises differing hidden dims; k=3 fixed to kill selection bias |
| [justification_finetune_null_result.md](justification_finetune_null_result.md) | Why we conclude **NTP finetuning fails** (scale × depth ablation) | Δ Spearman ≈ 0 at maximal for all 4 families; the +0.048 gpt-oss tier0 "gain" is a length confound that vanishes at maximal |
| [justification_thalesian_autopsy.md](justification_thalesian_autopsy.md) | Why Thalesian wins via the **cuneiform finetune (objective)**, not tokenizer/architecture | Control ladder (maximal): Thalesian 0.411 vs vanilla uMT5 0.297 (= random floor 0.301, below size-matched Qwen-1.7B 0.355); uMT5/Thalesian tokenizers are the *least* efficient (`factor_ladder_bars.png`, `T1_year_pls_maximal.csv`, `tokenization_audit.csv`) |
| [justification_bias_check_gate_and_metadata.md](justification_bias_check_gate_and_metadata.md) | Why we **bias-checked before probing** + treat metadata as confound | Permutation p=0.001 (signal real); `corpus_source` = near-perfect period proxy; metadata-only dating Sp 0.616→0.203 balanced (`T8`) |
| [justification_probing_over_prompting.md](justification_probing_over_prompting.md) | Why **probe activations, not prompt** the model | Models know king dates ~88% (T9) but prompting to date a fragment is ~chance (ruler F1 ≈0.13 vs 0.059, T10) |
| [justification_no_vocab_expansion.md](justification_no_vocab_expansion.md) | Why **no tokenizer/vocab expansion** for finetune | Domain BPE saturates ~4k pieces (≤8–16% gain); ~11M tokens too small; top candidates are royal/divine names = leakage (`TOKENIZER_EDA.md`) |

## Pre-existing dossiers (already in this folder)

- [justification_mlm.md](justification_mlm.md) — MLM over NTP for *restoration* (Fetaya 2021, MMBERT).
- [justification_sign_level_tokenization.md](justification_sign_level_tokenization.md) — sign-level tokenization.
- [justification_aeneas_twin_architecture.md](justification_aeneas_twin_architecture.md) — Aeneas twin architecture.
- [model_selection_phase2.md](model_selection_phase2.md), [data_source_summary.md](data_source_summary.md),
  [chunrong_data_cleaning_decisions.md](chunrong_data_cleaning_decisions.md),
  [seal_round4_pipeline_plan.md](seal_round4_pipeline_plan.md),
  [cdli_oracc_metadata_matching.md](cdli_oracc_metadata_matching.md), and the validation/audit logs.

## Paper → decision map (what each cited paper backs)

- **Gurnee & Tegmark, "Language Models Represent Space and Time"** → Spearman metric [direct],
  PLS linear readout [direct], probing-over-prompting [direct], need for dated anchors
  (royal-only) [supporting].
- **Yoffe, Dershowitz, Vishne & Sober, "…Sequentially Correlated Literary Properties…"** →
  MC-balancing / surrogate-label nulls [analogous], maximal confound removal [supporting],
  bias-check-against-null [supporting].
- **Nathan (preprint, title TBD)** → MC-balancing / imbalance design [primary — placeholder].
- **Ojala & Garriga (JMLR 2010), permutation tests** → bias-check gate [direct — add to bib].
- **Geometry-of-representations set** (hidden-rep geometry; categorical/hierarchical concepts;
  hidden lattice geometry) → PLS low-rank temporal subspace [direct].
- **RepEng surveys** (Representation Engineering survey; Taxonomy of RepEng) →
  probing-over-prompting [supporting].
- **"The Medium Is Not the Message" (linear concept erasure)** → confound-aware linear readout
  [supporting].
- **Fetaya et al. "Filling the Gaps"; MMBERT** → MLM/objective-over-exposure argument; feeds the
  finetune null + no-vocab-expansion decisions [supporting].

## Open items to resolve before final write-up

1. **Nathan preprint citation** — the MC-balancing design follows a **Nathan preprint** not yet
   in `papers/`; cite as "Nathan (preprint)" for now and **fill in title/author/venue when it is
   public**. Keep this separate from the **SEAL = Wasserman & Streck** *data* citation — they are
   different references. (See `justification_balanced_mc_protocol.md` §4.)
2. **TF-IDF tier0 number** — **0.474 is NOT from the fair test** (confirmed by Yarin); it
   predates the mean/balanced/200-MC/maximal/PLS protocol. Use committed-table **0.407 → ~0.29**
   as authoritative; 0.474→0.245 is illustrative only.
3. **ORCC labeled count** — quote **893 labeled of 1,202** (memory `project_canonical_sizes.md`),
   reserve "~1,200" for prose.
4. **Add to bib** — Ojala & Garriga (JMLR 2010), permutation tests (referenced by the bias-check
   gate); RepEng surveys (probing-over-prompting).

## Candidate decisions still worth a dossier (not yet written)

- Frozen-representation probing vs. end-to-end finetuning of the readout.
- GroupKFold-by-ruler + name-masking + **leave-one-ruler-out** (LORO drop is small —
  `T5_loro.csv`: 0.027 / 0.055) as the unified leakage-control design (currently spread across
  the balanced/PLS dossiers — could be its own).
- Mean vs. last-token pooling (the dropped fig-1 panel B).
- The geodesic/Isomap manifold readout as a *label-free* corroboration of the supervised signal
  (`T6_phase_d.csv`: geodesic Spearman ~0.25, pairwise-order acc ~0.72 with no labels).
