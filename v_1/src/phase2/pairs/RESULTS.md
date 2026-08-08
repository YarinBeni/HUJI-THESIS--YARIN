# E1 results — pairwise chronology ("which fragment is earlier?")

**Short version: the collapse survives the change of task format, and the
English/Akkadian contrast sharpens. On raw Akkadian no arm separates from the
char n-gram floor and the random twins interleave with the trained models; on the
English gloss trained OLMo-2 and Qwen3-8B beat the floor while OLMo's random twin
falls to the bottom of the table. The model's own Yes/No behaviour is degenerate
even where its representations carry the signal — a probe/behaviour dissociation.**

Protocol: quota m=21 pairs per ruler-pair per draw, 1/m_ij training weights,
metrics macro-averaged over ruler-pairs, both-rulers-held-out 5-fold splits,
100 draws (mean ± sd over draws). 1,187 dated fragments, 40 rulers, 777 eligible
ruler-pairs. Floors and arms identical in machinery; layer chosen on a cheap
selection pass (flagged in each JSON). Jobs F1/F2/F3, cluster 22587–22589.

## 1. The probe table (macro accuracy over unseen-ruler pairs)

### akk_maximal — floor .658 ± .038. Nobody separates.

| arm | macro acc | vs floor |
|---|---|---|
| llama2_13b | .673 ± .035 | +.015 |
| **olmo2_7b_random** | .671 ± .039 | +.013 |
| llama2_70b_random | .664 ± .035 | +.006 |
| random (qwen3-8b init) | .662 ± .043 | +.005 |
| olmo2_7b | .661 ± .044 | +.003 |
| qwen3_1b7 | .660 ± .039 | +.002 |
| **tfidf_char (floor)** | **.658 ± .038** | — |
| llama2_70b | .655 ± .042 | −.004 |
| qwen3_8b | .649 ± .042 | −.009 |
| llama2_7b | .649 ± .043 | −.009 |
| gpt_oss_120b | .642 ± .045 | −.016 |
| qwen3_32b | .631 ± .042 | −.027 |

Every gap is a fraction of one draw-sd, and two of the top three arms are
**untrained**. This is the regression result in pairwise clothing: on raw
Akkadian, nothing any trained LLM adds survives past character n-grams. The
pairwise format was the "maybe absolute calibration was the problem" escape
hatch, and it is now closed.

### eng_tier0 — floor .586 ± .038. Real structure appears.

| arm | macro acc | vs floor |
|---|---|---|
| **qwen3_8b** | **.636 ± .023** | **+.049** |
| **olmo2_7b** | **.634 ± .035** | **+.048** |
| qwen3_32b | .610 ± .039 | +.023 |
| llama2_70b | .601 ± .038 | +.014 |
| qwen3_1b7 | .590 ± .042 | +.004 |
| **tfidf_char (floor)** | **.586 ± .038** | — |
| llama2_13b | .586 ± .036 | −.001 |
| llama2_7b | .584 ± .037 | −.003 |
| llama2_70b_random | .577 ± .038 | −.010 |
| random (qwen3-8b init) | .571 ± .044 | −.016 |
| gpt_oss_120b | .559 ± .037 | −.028 |
| **olmo2_7b_random** | **.553 ± .044** | **−.033** |

The trained-vs-own-twin gaps that raw Akkadian entirely lacks: OLMo-2 **+.081**
over its twin, Qwen3-8B **+.065** over the shared random-init. The floor itself
drops .07 moving from Akkadian to English — the n-grams lose their orthographic
period cues in translation — and that is exactly where the trained models start
earning their keep.

### Where the trained advantage lives: the SHORT gaps

Accuracy by |Δyear| bin, eng_tier0 (floor / twin / trained):

| |Δyear| | tfidf | olmo2_7b_random | olmo2_7b | qwen3_8b |
|---|---|---|---|---|
| 0–25 yr | .48 | .50 | **.54** | **.53** |
| 25–75 yr | .51 | .49 | **.55** | **.56** |
| 75–200 yr | .58 | .53 | **.66** | .62 |
| 200+ yr | .61 | .59 | .61 | .62 |

At 200+ years everyone is similar — coarse era separation is easy. The trained
models' edge is at **fine resolution** (0–75 years), where the floor sits at
chance. On akk_maximal the trained arms are at chance in the short bins too
(.46–.49), like everyone else.

## 2. The behavioural read (F2) is degenerate — informatively so

| arm | variant | yes-rate | order-consistency | macro acc |
|---|---|---|---|---|
| qwen3_1b7 | both | .00 | .00 | .500 |
| qwen3_32b | akk / eng | .00 / .05 | .00 / .08 | .499 / .486 |
| qwen3_8b | akk / eng | .38 / .37 | .52 / .54 | .467 / .451 |

The small and large Qwen answer **No to everything**; qwen3_8b at least varies
but flips its answer when the texts swap places barely half the time, and lands
below chance. Meanwhile a linear probe on the same model's activations reads the
order at .636. The information is present in the representation and not
accessible through the model's own question-answering — the E2 steering
experiment asks exactly why. (Prompt engineering could surely improve the raw
numbers; the probe/behaviour gap is the datum.)

## 3. Robustness (F3, m=100 draws=10)

Same ordering everywhere, every number a touch lower (e.g. olmo2_7b akk .657,
qwen3_8b eng .627, floor akk .626): pushing 5× more weight into the giant
ruler-pair grids does not change the picture, so m=21 is not manufacturing it.
Files: `*.m100.json`.

## 4. What this licenses, and what it does not

**Can be said.** The entity→document collapse is not a task-format artifact:
relative ordering fails on raw Akkadian exactly where absolute regression
failed. On English glosses of the same fragments, mid-size trained models carry
chronological signal beyond character n-grams — concentrated at fine (0–75 yr)
resolution — and their random twins do not. And a pairwise probe learns this
**without ever seeing an absolute year label**, so absolute-label bias cannot be
its source. The learned directions are saved (`results/directions/`) for the E3
cosine comparison against the frozen cell-A name direction.

**The E8 inference (F4, jobs 22607), at ruler-level honesty (n = 40):**

*Permutation with refit* ("is there chronological signal at all", null = ruler
chronology shuffled, 150 perms — .0066 is the attainable floor):

| | akk_maximal | eng_tier0 |
|---|---|---|
| tfidf_char floor | **p = .013** | p = .11 (ns) |
| olmo2_7b | **p = .0066** | **p = .0066** |
| qwen3_8b | — | **p = .0066** |

*Dyadic ruler bootstrap on the paired contrasts* (95% CI, two-sided p):

| contrast | eng_tier0 | akk_maximal |
|---|---|---|
| olmo2_7b − floor | +.035 [−.035, +.105], p=.34 | +.027 [−.031, +.083], p=.36 |
| olmo2_7b − twin | +.074 [−.014, +.165], **p=.098** | −.002, p=.97 |
| qwen3_8b − floor | +.048 [−.056, +.144], p=.33 | — |
| qwen3_8b − twin | +.063 [−.009, +.133], **p=.082** | — |

**The honest summary.** The qualitative dissociation is significant where it
matters: on English glosses the trained models carry statistically real
chronological signal (perm p=.007) while the n-gram floor does NOT reach
significance (p=.11) — and on raw Akkadian the floor itself is significant
(p=.013) with the trained model adding nothing (vs-floor p=.36, vs-twin p=.97).
The direct arm-vs-arm gaps on English are positive but individually marginal
(vs-twin p≈.08–.10): forty rulers buy exactly this much power, and the thesis
should quote the permutation dissociation, not the pairwise contrasts, as the
inferential result.

## 5. Reproducing

```bash
python pairs_data.py                                   # self-test
python probe_pairs.py --method tfidf_char --variant akk_maximal
sbatch sbatch/F1_pairs_probe.sbatch                    # arms (CPU, cluster npz)
sbatch sbatch/F2_pairs_behavioral.sbatch               # Yes/No (GPU)
sbatch sbatch/F3_pairs_robustness.sbatch               # m=100 pass
python aggregate_pairs.py
```
