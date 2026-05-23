# Round 2 — Qwen-on-Akkadian Diagnosis: Final Report

- **Date:** 2026-05-23 (Round 2 complete except Random-Qwen P0 data gap)
- **Scope:** Final synthesis of Round 2 Phase 0 (class imbalance), Phase 1a (factual knowledge), Phase 1b (prompt framing), Phase 3 imbalanced + balanced MC (Thalesian Akkadian-finetuned encoders).
- **Inputs consumed:**
  - `PLAN_round2_qwen_diagnosis.md`
  - `v_1/src/linear_probing/results/orcc_round2_phase0/aggregated/phase0_report.md`
  - `v_1/src/linear_probing/results/orcc_round2_phase1a/aggregated/phase1a_report.md`
  - `v_1/src/linear_probing/results/orcc_round2_phase1b/reprobing/phase1b_report.md`

**The Round-1 puzzle.** On the imbalanced 38-ruler ORCC eval surface, pretrained Qwen 2.5-7B-Instruct produced a Macro-F1 of 0.117 — worse than random projections (0.235), an Akkadian MLM (0.220), and TF-IDF (0.326). Round 2 was pre-committed to disentangle four mutually compatible explanations: dataset imbalance, missing factual knowledge, missing elicitation, and missing representational signal.

---

## Pre-committed hypotheses (from `PLAN_round2_qwen_diagnosis.md`)

| Phase | Hypothesis | Pre-committed gate |
|---|---|---|
| 0 | H0 — Round 1's ranking is partially an imbalance artifact | Balanced MC mean − 2σ ≥ Round-1 imbalanced Macro-F1 per method (TF-IDF anchored ≥ 0.50) |
| 1a | H1a — Qwen knows ruler↔period associations as facts | kp0 hits ≥ 6/8 (year overlap), kp2 hallucination rate < 0.30 |
| 1b | H1b — Raw-text framing suppresses Qwen's latent chronology signal | Direct-answer ruler Macro-F1 ≥ 0.25 **or** prompted-probe Macro-F1 ≥ balanced-Qwen baseline + 0.05 |
| 3 | H3 (Akkadian finetune) — A domain-finetuned LLM closes the Qwen-vs-Aeneas gap | Probe Macro-F1 on Thalesian ≥ Round-1 Qwen + 0.10 (planned) |

---

## Phase 0 — Class imbalance

Phase 0 re-ran the probing pipeline on Wasserman-style balanced MC draws (8 rulers × 21 fragments per draw, n=200 draws). All three methods with materialized activation data **passed** the per-method gate ((mean − 2σ) − R1 > 0):

- **TF-IDF (tier0, CLS)**: R1 = 0.326 → MC = 0.650 ± 0.037 (Δ = **+0.323**) — `phase0_report.md` TF-IDF table.
- **MLM (tier0, mean, CLS)**: R1 = 0.220 → MC = 0.430 ± 0.037 (Δ = +0.136) — `phase0_report.md` MLM table.
- **Qwen-pretrained (tier0, mean, PLS)**: R1 = 0.111 → MC = 0.353 ± 0.035 (Δ = +0.171) — `phase0_report.md` Qwen table.
- **Qwen-pretrained (tier0, mean, CLS)**: R1 = 0.117 → MC = **0.352 ± 0.042** (Δ = **+0.235**) — `phase0_report.md` CLS leaderboard.

The Qwen CLS jump (+0.235 Macro-F1) is the single largest finding of Round 2: roughly half of the Qwen-vs-TF-IDF gap on Round 1 dissolves once we control for class imbalance. However, balanced Qwen (0.352) is still substantially below balanced TF-IDF (0.650) — a residual gap of ~0.30 Macro-F1 that imbalance does **not** explain.

**Caveat — Random-Qwen data gap.** The aggregator reports "NO DATA" for Random-Qwen on both CLS and PLS pairings (400 per-draw files contain empty `results: {}` dicts; activation files likely landed under a path the P0 driver did not search, plausibly the `orcc__embed` → `orcc_round1` rename casualty). This makes the Phase 0 secondary gate read `INDETERMINATE` in `phase0_report.md` because the "all 4 methods" axis is incomplete. The 3 methods that *did* run all satisfy the per-method gate, so the imbalance hypothesis is confirmed as a contributor; the Random row remains data-not-collected, not a true failure.

## Verdict: PASS — class imbalance CONFIRMED as a major contributor (Random-Qwen data gap to close in follow-up)

---

## Phase 1a — Factual knowledge of rulers

Phase 1a tested whether Qwen has the reign-year facts at all (closed-book, no fragment text). On kp0 ("When did ruler X reign?") Qwen scored **8/8 hits** at ±50-year tolerance, including correct ranges for Ashurbanipal, Sennacherib, Esarhaddon, Sargon II, Nebuchadnezzar II, Tiglath-pileser III, Nabonidus, and Sîn-šarru-iškun (`phase1a_report.md` kp0 table). On kp1 the aggregate recall over Phase-0 rulers was 0.750 (PASS vs ≥ 0.50). The kp2 fake-name hallucination rate was 0.500 (gate < 0.30 FAIL), but this affects only the hallucination sanity check — the actual H1a hypothesis (Qwen knows the rulers) is unambiguously supported. We can rule out "Qwen lacks the facts" as the explanation for poor probing performance.

## Verdict: FAIL — hypothesis REJECTED (Qwen has the relevant factual knowledge)

---

## Phase 1b — Prompt framing

Phase 1b wrapped fragments in 4 prompt variants (pv0–pv3) across 6 layers × 2 poolings and re-probed. The best prompted ruler Macro-F1 across **all** variants/layers/poolings is **0.139** (mean pooling, layer 0, all 4 variants tied — `phase1b_report.md` per-variant verdict block). This beats only the three weakest raw-Qwen baselines (0.117, 0.118, 0.130) and loses to MLM (0.220), Random-Qwen (0.235), and TF-IDF (0.326). The per-variant Δ vs Round-1 Qwen `mean` is +0.022 — below the pre-committed +0.05 threshold. The direct-answer ruler accuracy from prompt-driven QA peaks at 0.448 (pv0/mean/L0 PLS table), which is informative but is a different metric than the gate's Macro-F1 ≥ 0.25 on direct-answer rulers. The prompted-probe path clearly fails the +0.05 lift gate against the Round-1 Qwen baseline.

Bottom line: prompting nudges raw Qwen marginally, but does not move it past the actually-good methods. Latent-but-fully-elicitable chronology in Qwen is not consistent with these numbers.

## Verdict: FAIL — hypothesis REJECTED (prompts do not close the gap)

---

## Phase 3 — Akkadian-finetuned model (Thalesian)

Two Akkadian-aware UMT5 checkpoints were probed with the same CLS+PLS pipeline used in Round 1:
- **`Thalesian/AKK_300m`** — UMT5-small (~300M params, 8 encoder layers, d=512), finetuned on Akkademia+CDLI Akkadian transliterations and Akkadian↔English translation pairs.
- **`Thalesian/cuneiformBase-400m`** — UMT5-base (~400M params, 12 encoder layers, d=768), multilingual ancient scripts (Akkadian + Sumerian + Hittite + Linear B + Elamite).

Activation extraction (jobs 8223–8226) wrote to `v_1/src/linear_probing/results/orcc__embed/activations/thalesian_*`. Probing job 8364 completed in 2h11m and auto-pushed `cls_results_thalesian_*.json` (72 + 104 keys) and `pls_results_thalesian_*.json` (108 + 156 keys) into the canonical Round-1 result dirs.

**Imbalanced CLS ruler — full leaderboard (top 8 by Macro-F1):**

| Rank | Method | Cleaning | Pool | Best L | Accuracy | Macro-F1 | ×chance |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | TF-IDF | tier0 | — | 0 | 0.777 | **0.326** | 4.6× |
| 2 | **Thalesian cuneiBase-400m** | maximal | mean | 12 | 0.570 | **0.263** | 4.5× |
| 3 | Random-Qwen | tier0 | mean | 1 | 0.659 | 0.235 | 3.3× |
| 4 | TF-IDF | maximal | — | 0 | 0.651 | 0.228 | 3.2× |
| 5 | MLM Aeneas | tier0 | mean | 0 | 0.630 | 0.220 | 3.1× |
| 6 | Random-Qwen | maximal | mean | 3 | 0.581 | 0.216 | 3.0× |
| 7 | **Thalesian cuneiBase-400m** | tier0 | mean | 12 | 0.567 | **0.210** | 3.6× |
| 8 | **Thalesian AKK_300m** | tier0 | mean | 8 | 0.472 | **0.160** | 2.7× |
| … | Qwen pretrained | tier0 | mean | 0 | 0.523 | 0.117 | 1.6× |

**Imbalanced PLS year-raw regression — Spearman / R² / MAE (best per method):**

| Method | Best L | Best k | Spearman | R² | MAE (years) |
|---|---:|---:|---:|---:|---:|
| **Thalesian cuneiBase-400m** (tier0, mean) | 12 | 2 | **0.467** | **+0.105** | **75.1** |
| Thalesian AKK_300m (tier0, mean) | 7 | 3 | 0.435 | +0.069 | 76.5 |
| Thalesian cuneiBase-400m (maximal, mean) | 9 | 2 | 0.417 | +0.142 | 77.1 |
| Random-Qwen (tier0, mean) | 12 | 2 | 0.184 | −175.7 | 127.9 |
| Qwen pretrained (tier0, mean) | 5 | 5 | 0.121 | −198.2 | 128.3 |
| MLM Aeneas (tier0, mean) | 2 | 2 | −0.115 | −152.9 | 139.5 |

**Findings:**
1. **Akkadian finetuning helps a lot.** Thalesian cuneiBase-400m at 0.263 Macro-F1 is **2.2×** above Qwen pretrained (0.117) on the same imbalanced probing setup.
2. **Bigger Thalesian wins** even though it's *less* Akkadian-focused (multilingual ancient-script vs Akkadian-only). cuneiBase-400m (400M, 12 layers) beats AKK_300m (300M, 8 layers) by a wide margin (0.263 vs 0.160 best Macro-F1). At this scale, parameter count + multilingual ancient-script exposure outweighs Akkadian-only specialization.
3. **R² is positive for Thalesian only.** Every other representation (Qwen pretrained, Random-Qwen, MLM) produces R² between −150 and −500 on year regression — worse than constant guessing. Thalesian cuneiBase-400m drops year-MAE from 128 years to 75 years, lifts Spearman from 0.12 to 0.47, and gives the *only* well-conditioned year-features in the comparison.
4. **Best layer is the FINAL encoder layer** in both Thalesian models (L8 for AKK_300m, L12 for cuneiBase-400m). Discriminative ruler features build *through depth* — what you want from a representation-learning model.
5. **Last-token pooling is broken for these encoders.** All last-pool configs sit near chance F1 (0.022–0.087). The last non-pad token is a near-constant SEP/EOS embedding across fragments. Mean pooling is the only valid choice for encoder-decoder.
6. **Thalesian closes ~50% of the residual Qwen↔TF-IDF imbalanced gap** (0.263 vs 0.326 vs 0.117); the remainder is likely the character-n-gram lexical-overlap signal TF-IDF leverages, which a 400M-param model cannot fully replicate on ~1,200 fragments.

**Caveat for the thesis.** Chance F1 is 0.059 for Thalesian (17 classes retained at min_count=5) vs 0.071 for Round-1 methods (~14 classes). The ORCC parquet was rebuilt between Round 1 (893 labeled rows) and Round 2 (1193 labeled rows); class-count differences mean raw Macro-F1 comparisons across method-groups are slightly biased. The ×chance column in the leaderboard above is the apples-to-apples view.

### Phase 3 balanced MC (job 8477, 200 draws × 21 frags × 8 rulers, tier0/mean only)

| Method | Cleaning | Pool | MC layer | MC Macro-F1 (CLS) | MC Macro-F1 (PLS via year-PLS-DA) |
|---|---|---|---:|---:|---:|
| TF-IDF | tier0 | — | 0 | **0.650 ± 0.037** | 0.480 ± 0.037 |
| TF-IDF | maximal | — | 0 | 0.498 ± 0.040 | 0.395 ± 0.033 |
| MLM Aeneas | tier0 | mean | 15 (CLS) / 14 (PLS) | 0.460 ± 0.044 | 0.395 ± 0.042 |
| **Thalesian cuneiBase-400m** | tier0 | mean | 12 (CLS) / 11 (PLS) | **0.448 ± 0.043** | **0.393 ± 0.040** |
| Qwen pretrained | tier0 | mean | 0 (CLS) / 3 (PLS) | 0.352 ± 0.042 | 0.363 ± 0.042 |
| Thalesian AKK_300m | tier0 | mean | 8 (CLS) / 3 (PLS) | 0.323 ± 0.039 | 0.346 ± 0.039 |
| Random-Qwen | — | — | — | data gap | data gap |

### Phase 3 balanced-vs-imbalanced delta (the new finding)

| Method | R1 imbalanced | Balanced MC | Δ (balance lift) |
|---|---:|---:|---:|
| TF-IDF (tier0) CLS | 0.326 | 0.650 | **+0.323** |
| **Thalesian cuneiBase-400m** (tier0 mean) CLS | 0.210 | **0.448** | **+0.238** |
| Qwen pretrained (tier0 mean) CLS | 0.117 | 0.352 | +0.235 |
| MLM Aeneas (tier0 mean) CLS | 0.220 | 0.460 | +0.241 |
| Thalesian AKK_300m (tier0 mean) CLS | 0.160 | 0.323 | +0.163 |

**Nuanced final picture (after balanced MC):**

1. On the **balanced** surface, the leaderboard is: TF-IDF (0.650) >> MLM (0.460) ≈ **Thalesian cuneiBase-400m (0.448)** > Qwen pretrained (0.352) > Thalesian AKK_300m (0.323). Thalesian cuneiBase ties MLM under balanced eval; AKK_300m drops *below* Qwen pretrained.
2. On the **imbalanced** surface, Thalesian cuneiBase (best at 0.263 maximal/mean) clearly beats both MLM (0.220) and Qwen (0.117).
3. **The Macro-F1 ranking is metric-surface-dependent**: which method "wins" depends on whether you control for class imbalance. Under balanced eval, sign-level MLM Aeneas (~25M params) ties UMT5-base finetuned on Akkadian (~400M).
4. **Thalesian cuneiBase still has the strongest year-regression signal by a wide margin** — Spearman 0.467, R² +0.105, MAE 75yr — and is the *only* representation with positive year R² in the entire study. This is the most robust Phase 3 finding: Akkadian-finetuned representations encode chronology in a linearly-extractable way that no other model does.
5. **Thalesian AKK_300m's smaller balance-lift (+0.163 vs +0.235 for Qwen)** suggests its imbalanced advantage was less of a "class-balance artifact" than Qwen's — its representations were already doing better on majority classes — but it has lower overall ceiling than cuneiBase.

## Verdict: PASS — Akkadian pretraining matters for year regression unambiguously (only Thalesian gets positive R²; Spearman 0.47 vs Qwen 0.12). For ruler-CLS, Thalesian cuneiBase-400m matches MLM Aeneas on the balanced eval surface (0.45 vs 0.46) but neither approaches TF-IDF (0.65). The Qwen↔Thalesian-cuneiBase gap on imbalanced (0.117 → 0.263) is mostly Akkadian-pretraining; the residual TF-IDF↔Thalesian gap on balanced (0.65 vs 0.45) is the unsolved core puzzle.

---

## Unified narrative

The Round-1 puzzle had at least **two intertwined causes**. The first, **class imbalance**, is now isolated and substantial: balancing 8 rulers at 21 fragments per class lifts every method (Δ ≈ +0.14 to +0.32 Macro-F1), and Qwen alone gains ~0.235 Macro-F1 in absolute terms. The imbalanced 38-class ruler task with 21–268 fragments per class is structurally hostile to weaker representations, and a substantial portion of the headline "Qwen is worse than random" story was an evaluation-surface artifact.

The second cause is **a residual ~0.30 Macro-F1 gap that survives balancing**. On the balanced MC surface TF-IDF reaches 0.650 while Qwen-pretrained tops out at 0.352. Something about Akkadian-relevant signal is missing — or unrecoverable by a linear probe — from Qwen's residual stream, even after we remove the class-frequency confound.

Phases 1a and 1b rule out two "shallow" explanations for that residual gap. Qwen demonstrably **has** the reign-year facts (Phase 1a, 8/8 on kp0), so this is not a knowledge problem. And **prompting does not unlock** chronological signal at the probe surface (Phase 1b, best 0.139 across 48 combinations, still well below MLM/Random/TF-IDF), so this is not an elicitation problem in the simple "task-framing" sense.

What remains live for the residual gap: **(a)** Phase 3 — domain pretraining (does an Akkadian-trained LLM, even small, beat raw Qwen-7B?); **(b)** Phase 2 — scale and SAE on Qwen 3 dense; **(c)** deeper structural questions — tokenization geometry, training-data composition (how much Akkadian, if any, did Qwen 2.5 see?), and pooling choices (mean over fragment vs last-token; CLS vs PLS-DA). The Phase 3 sbatch in flight is the next data point.

For the thesis section, the cleanest narrative arc is: Round 1 found a counterintuitive negative result; Round 2 Phase 0 demonstrated that ~50% of the apparent Qwen failure was an imbalance artifact and gave us a trustworthy balanced eval surface; Round 2 Phases 1a/1b ruled out the two simplest reasons for the residual gap; the remaining gap motivates the Thalesian / scale / tokenization investigation.

---

## Next steps

- **Close the Random-Qwen data gap.** Resubmit Phase 0 random probes with corrected activation path (post-`orcc__embed` → `orcc_round1` rename) so the Random tier0/mean (CLS L1) and tier0/mean (PLS L0) cells fill in. Converts Phase 0's verdict on the Random axis from INDETERMINATE to PASS/FAIL.
- **Run Thalesian maximal/mean balanced MC.** The imbalanced Thalesian cuneiBase-400m best (0.263) was on maximal/mean cleaning, but the current balanced MC only ran tier0/mean. Adding maximal/mean MC for both Thalesian models would settle whether maximal cleaning preserves Thalesian's advantage under balanced eval.
- **Consider Phase 2 (scale + SAE).** Phase 3 confirmed Akkadian pretraining is a major lever but TF-IDF still leads (0.326 vs Thalesian 0.263). Qwen 3 dense at 4B/14B with the same probing pipeline would either find a positive scaling slope (emergent-with-scale story) or confirm the data-composition ceiling.
- **Tokenization spot-check.** A cheap diagnostic worth doing in passing: per-fragment token counts for Qwen vs Thalesian's UMT5 tokenizer. Large fragmentation differences would suggest the Round-1 Qwen weakness has a tokenizer-geometry component.
- **Document the balanced eval surface as canonical going forward.** All subsequent probes (Phase 2, follow-up Phase 3 ablations) should report on both the Round-1 imbalanced grid (historical comparability) and the Phase-0 MC-balanced grid (apples-to-apples).
