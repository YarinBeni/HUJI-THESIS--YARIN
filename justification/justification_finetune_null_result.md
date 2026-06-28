# Justification — The finetuning null result (NTP does not improve dating at maximal)

> **Thesis claim this supports:** "Continued NTP (next-token-prediction) finetuning on Akkadian
> lowers perplexity but does **not** improve confound-controlled dating, at **any** model scale
> (1.7B → 8B → 32B → 120B-LoRA) or **any** unfreezing depth (cut at block 0/9/19/25). The one
> apparent gain is a re-introduced length confound that disappears under the maximal regime."

## 1. The decision/finding, in one sentence

We finetuned across a **scale × unfreezing-depth ablation** and re-probed every checkpoint
with the *same* `pls__mc_balanced_maximal` protocol; the scoreboard shows **Δ Spearman ≈ 0**
versus base, so we report a negative result — exposure to more Akkadian via NTP is not the
missing ingredient for dating.

## 2. The evidence (from `v_1/src/finetune/results/scoreboard.md`)

**Maximal (honest) regime — base vs best NTP arm, year-PLS Spearman:**

| family | base | best ft arm | Δ |
|---|---|---|---|
| qwen3_1b7 | 0.3549 | 0.3556 | +0.001 |
| qwen3_8b | 0.3633 | 0.3646 | +0.001 |
| qwen3_32b | 0.3398 | 0.3398 | **0.000** |
| gpt_oss_120b | 0.3301 | 0.3301 | **0.000** |

For 32B and 120B the frozen-layer arms are **byte-identical to base** (the most recent commit
notes the 32B frozen-layer signature is byte-identical) — a clean confirmation that nothing
changed in the layers the probe reads.

**The apparent exception is a length confound, not a win:** at **tier0** (full text)
`gpt_oss_120b` ft00 jumps 0.4038 → **0.4514** (+0.048). But tier0 leaves the length crutch in
(see [[justification_maximal_cleaning_regime]]); under maximal the same arm is back at 0.3301 =
base. So the finetuned model learned to exploit *text length*, not chronology — the exact
artifact the maximal regime was built to catch.

## 2b. What the finetuning corpus actually was (data + word counts)

The NTP corpus is fragment-level Akkadian text built from the **canonical unified splits**
(`v_1/data/unified/{train,val}.parquet`) by `v_1/src/finetune/prepare_ntp_data.py`, constructed
*identically* to the ORCC probing corpus: words sorted by `(line_num, word_idx)`, space-joined
`value_clean` (fallback `value_raw`), tier0-cleaned (minimal markup strip). The **test split is
left untouched**. Every model family (1.7B → 8B → 32B → 120B-LoRA) trained on this same corpus.

**Amounts (authoritative — `v_1/data/finetune/metadata.json`, created 2026-06-10):**

| split | role | fragments | words | chars | ORCC-probe overlap |
|---|---|---|---|---|---|
| train | NTP training | 32,343 | **1,960,636** | 15,025,222 | 504 |
| val | perplexity eval | 4,042 | **253,798** | 1,947,804 | 67 |

**Corpus-source breakdown (words):**

| source | train words | val words |
|---|---|---|
| ORACC | 1,113,966 | 145,384 |
| eBL | 794,027 | 101,868 |
| Archibab | 52,643 | 6,546 |

One-line for the write-up: continued-pretrained (NTP) on **≈1.96 M words** of Akkadian (32,343
transliterated fragments from ORACC/eBL/Archibab), with a held-out **≈254 K-word** validation
set for perplexity. Train↔probe leakage was tracked and minimal (504 train / 67 val fragment-ID
overlaps with the ORCC probing corpus). Data artifacts: `v_1/data/finetune/ntp_train.parquet`,
`ntp_val.parquet`.

## 3. Why the experiment was designed this way (so the null is credible)

- **Scale axis (1.7B→120B):** rules out "the model was just too small."
- **Unfreezing-depth axis (cut00/09/19/25 + LoRA, `train_ntp.py --unfreeze-from`):** rules out
  "we let the wrong part of the network learn." Probing every cut shows no depth recovers
  dating.
- **Same protocol as the frozen baselines:** identical canonical seed-42 splits, GroupKFold by
  ruler, 200 MC balanced draws, maximal cleaning — so the comparison is apples-to-apples and
  the Δ is attributable to finetuning alone.
- **Leakage tracked:** `prepare_ntp_data.py` records ORCC-probe overlap in `metadata.json`, so
  any "gain" can't come from the finetuning corpus having memorised probe fragments.

## 4. Why this is theoretically expected (the thesis argument)

The plan predicts exactly this: *"NTP rewards 'predict the next sign/word in this local
context.' It does not force the model to separate [chronology from style/genre]… That is why
NTP finetuning can lower perplexity but barely move maximal-balanced Spearman"*
(`thesis_plan.md:752`). The null result is therefore not a disappointment but the **motivating
evidence** for the ordinal/deconfounded ChronoRank head: if exposure (NTP) and scale both fail
under honest evaluation, the missing ingredient must be the *objective* (ordinal, deconfounded,
chronology-encoding), which is the thesis's forward direction (`thesis_plan.md:112, 727`).

## 5. Supporting literature

- **Gurnee & Tegmark — "Language Models Represent Space and Time."** Temporal structure is a
  property of *representations*; whether NTP training instills it is an empirical question we
  answer negatively for Akkadian dating. **[supporting — frames what NTP would have to produce.]**
- **MMBERT** (`papers/txt/Transfer Learning papers/`) and **Fetaya et al., "Filling the Gaps"**
  (`papers/txt/Ancient Language papers/`): both argue the *objective/architecture* (bidirectional
  MLM, translation grounding) matters more than raw exposure in low-resource ancient languages —
  consistent with NTP-exposure being insufficient here. See [[justification_mlm]]. **[supporting.]**

## 6. Figures & tables to pull when writing

- Scoreboard: `v_1/src/finetune/results/scoreboard.md`, `scoreboard_best.csv`,
  `scoreboard_layers.csv`.
- Null-result comparison figs: `v_1/src/finetune/results/figures/maximal_pls_bestlayer.png`,
  `maximal_pls_layerwise.png`.
- Per-arm finetune curves (perplexity vs depth): `figures/ftcurves_qwen3_1b7_{tier0,maximal}.png`,
  `ftcurves_qwen3_8b_*`, `ftcurves_qwen3_32b_*`, `ftcurves_gpt_oss_120b_*`.
- Per-checkpoint probe summaries: `v_1/src/finetune/results/probes/*pls__mc_balanced_maximal__summary.json`.
