# E-MIN v2 result — 2026-09-02

**Design change after the v1 review.** (i) Akkadian base text = **tier0** (raw
transliteration; `maximal` was already a heavy cleaning, so corrupting it further
could leave nothing by construction). (ii) Encoders = the M.Sc.'s best LLMs on
documents, **Llama-2-7B (L16)** and **Qwen3-8B (L18)**, plus the stronger Akkadian
encoder **cuneiformBase-400m (L12)**; vectors come through the thesis's own
extraction code. (iii) **Language arms** akk / eng / mix, with a per-language
read-out inside mix. 1,193 docs / 40 rulers; frozen gkf_ruler folds; SLA §7 read-out
(pooled OOF; mc = mean over 200 frozen draws); 5 seeds for the head; every baseline
is one cross-fit on the same folds. Full tables: `EMIN2_TABLES.md`.

## Akkadian arm (tier0), mc ρ — the headline

| encoder | PLS k=2 | ridge (orig views) | ridge (all views) | **Chrono-Barlow** | under crop16: ridge-all → head |
|---|---|---|---|---|---|
| cuneiformBase-400m | .44 | .45 | .46 | **.61 ± .01** | .38 → **.49** |
| Llama-2-7B | .22 | .35 | .44 | **.54 ± .03** | .33 → **.45** |
| Qwen3-8B | .20 | .26 | .33 | **.43 ± .02** | .36 → **.39** |

Ruler-block null of the reported statistic: .00 ± .02 in every arm. Doc placebo ≈ 0.

## What it says

1. **With a competent encoder and raw text, the head beats every baseline — including
   ridge trained on the same augmented views — by .10–.15 mc ρ on Akkadian.** In v1
   (AKK_300m, maximal) the gap was .02. The v1 null result was an encoder/text
   problem, not a method problem.
2. **Augmentation-as-data alone recovers part of the LLM gap** (Llama .35 → .44,
   Qwen .26 → .33) but not all of it; the objective adds another ~.10. For the
   cuneiform encoder plain augmentation adds nothing (.45 → .46) and the objective
   adds .15.
3. **The head is the most robust to cropping** in every Akkadian arm. Name masking,
   once more, hurts nothing — in several arms it *helps* (Qwen akk head .43 → .49;
   cunei PLS .44 → .49). Three encoders, two text tiers: the ruler name is not what
   these representations date by. That is now a finding, not a caveat.
4. **Akkadian beats the English gloss** once the encoder knows Akkadian: mix-arm
   per-language read-out gives akk .55–.56 vs eng .41–.42 (Llama, cunei); the v1
   ordering (gloss > transliteration) was an artefact of the weak encoder.
5. **The head is not magic:** on the English gloss with the Akkadian-only encoder it
   *loses* to the frozen probes (.385 vs ridge .42, PLS .42). It amplifies structure
   the encoder has; it cannot supply structure the encoder lacks.
6. **Absolute level.** .61 on held-out rulers on raw Akkadian is above the M.Sc.'s
   best neural number (.31, Llama-2-7B on maximal) and above its TF-IDF ceiling (.54).

## Caveats

* Baselines are single cross-fits; the head has a seed spread. Give ridge a spread
  (bootstrap fold assignment) before quoting the margins as ±.
* PLS k=2 is competitive only on the small encoder (cunei .44); on 4096-d LLM
  features it is not a serious baseline.
* Duplicate texts (136 akk docs in 50 groups) and the 13 century-coded docs are
  still in the corpus; both decisions pending.
* One augmentation menu, one head size, one training length — no tuning was done
  on any arm, which is deliberate for a gate but leaves headroom unknown.

## Next

P1 ladder (single-variable erasures, per encoder) is now motivated; start with
cunei400m + Llama on tier0 Akkadian. Add `lang` to every score schema consumer
(done in trainer and baselines). Give baselines a seed spread. Decide the
duplicate/century policy in the corpus contract.
