# OLMo frequency experiment — results

**Short version: the dose-response is weak across the bulk of the data and clear only
at the extremes — 34 years between never-seen and most-seen. And the salience axis the
thesis assumed does not exist in the corpus.**

Counts: `v4_olmo-mix-1124_llama` — OLMo-2-1124-7B's **own** pretraining mix, not a
Dolma stand-in. No substitution, no caveat about corpus mismatch. 7,541/7,541 entities
counted, zero errors.

Figure: `results/figs/fig_frequency_doseresponse.png` (+ `.pdf`)
Numbers: `results/frequency_stats.json` · counts: `results/entity_counts.csv`

---

## 1. The gate: OLMo behaves like the other arms

Before any frequency claim is worth making, OLMo has to be a normal member of the
ladder. It is — held-out Spearman ρ on cell A:

| arm | hist. figure | world place | us place | art | headline |
|---|---|---|---|---|---|
| **OLMo-2-7B** | **.880** | **.900** | **.837** | **.812** | **.787** |
| its random twin | .565 | .537 | .395 | .247 | .558 |
| TF-IDF floor | .798 | .747 | .677 | .318 | .676 |
| *Llama-2-7B* | *.885* | *.922* | *.862* | *.843* | *.777* |

It beats its twin and the n-gram floor everywhere, and sits on top of Llama-2-7B — the
same size, the same behaviour. Median dating error 105.5 years against the twin's 203.4.

## 2. The headline question: does exposure predict accuracy? **No.**

ρ(log training count, |predicted year − true year|), held-out entities only.
Negative would mean *seen more often → dated better*.

| | trained OLMo | random twin |
|---|---|---|
| overall | **−0.040** | +0.181 |
| within century | **−0.102** | +0.013 |
| rarest decile → most-frequent decile | 108.7 yr → 92.7 yr | 173.8 yr → 327.2 yr |
| century bins pointing the right way | 16/19 | 11/19 |

n = 7,192 (single-token names excluded, see §4).

The direction is right and it is consistent — 16 of 19 century bins are negative, and
the within-century figure is *larger* than the overall one, so age is not manufacturing
it. But the size is negligible: across a **40,000-fold** range of training exposure the
median error improves by **16 years**, on a task whose overall error is ~105 years.
ρ = −0.04 is statistically significant only because n is large.

**Read: within this corpus, how often OLMo saw a person's name is not what determines
whether it can date them.**

### The twin's +0.181 is an artefact, and it is instructive

The untrained twin shows a *positive* correlation — apparently "more frequent names are
dated worse", which is nonsense as a causal claim. It is the age confound in pure form:
recent people are written about more (ρ(century, count) = +0.11), and an untrained
network predicts close to the mean year, so its error grows with distance from that
mean. Hence frequent ⇒ recent ⇒ far from the mean ⇒ large error.

Controlling for century collapses it from +0.181 to **+0.013** — the confound
disappears exactly as it should. That is the strongest evidence that the within-century
control is doing its job, and it is why the trained arm's −0.102 can be believed as
small-but-real rather than dismissed as noise.

It also means the trained-minus-twin gap (−0.222 overall) mostly measures *the twin's
artefact vanishing*, not a dose-response appearing. Do not quote the gap as the result.

## 2b. At the extremes the effect is real — the overall ρ was diluted

ρ over all 7,192 entities is a single number describing a relationship that is not
linear in log-count. Splitting the tails says more:

| | never seen (count = 0, n = 583) | top 5% (≥ 40,272, n = 360) | difference |
|---|---|---|---|
| **trained OLMo** | 122.4 yr | **88.5 yr** | **−33.9 yr** |
| random twin | 189.7 yr | 343.2 yr | +153.5 yr |

Between the extremes of a 40,000-fold exposure range the trained model is **34 years
more accurate**, and the twin moves 154 years in the opposite direction. So exposure is
not irrelevant — it is weak across the bulk of the distribution and visible at the ends.
ρ = −0.04 understates this because most entities sit in a middle where the count barely
moves the error.

The more striking half of that table is the left column: on 583 people whose full name
**never appears once** in the training data, the probe still recovers the year to within
122 years — better than the twin manages on the people it saw most.

**The caveat, and its resolution.** A count of zero is zero for the *exact full
string* — "Franz Xaver Feuchtmayer" can score zero while "Feuchtmayer" appears often.
So the surnames were counted too (`count_surnames.py`), and exposure re-read as
max(full name, surname) — a deliberately generous upper bound.

The sceptics were mostly right about the group: **86% of the "never seen" evaporate**
(median surname count among them: 5,597). But the claim survives on what remains:

| exposure = max(full, surname) | n | trained OLMo | random twin |
|---|---|---|---|
| zero under BOTH forms | **51** | **151.8 yr** | 245.4 yr |
| top 5% | 163 | 107.8 yr | 268.6 yr |

Fifty-one people neither of whose name forms appears once are still dated ~94 years
better than the twin manages on them, and the never-seen → most-seen gap *widens* to
44 years under the stricter accounting. The claim stands, at a fifth of its original
sample size and with the honest label: these are the entities for which no string
evidence of exposure exists, not entities provably absent from the corpus.

## 2c. The same question under the PLS read-out

The deck reports ridge for cell A and PLS for the fragment cells, so a frequency claim
read only under ridge is a claim about one of the two read-outs the thesis uses. With
per-entity PLS predictions now written (`probe_eng_pls.py`, best k = 16 for the trained
arm), the whole analysis re-runs on them:

| | ridge | PLS (best k) |
|---|---|---|
| overall ρ, trained | −0.040 | **−0.031** |
| within century, trained | −0.102 | **−0.085** |
| twin, overall | +0.181 | +0.172 |
| never-seen → top 5%, trained | 122.4 → 88.5 yr (−34) | 127.0 → 99.4 yr (−28) |
| never-seen → top 5%, twin | 189.7 → 343.2 yr (+154) | 193.4 → 348.2 yr (+155) |

Every number moves by less than .01 in ρ and by a few years at the tails, and nothing
changes sign. The read-out is not carrying the result: the flat middle, the modest
separation at the extremes, and the twin's age artefact are all properties of the data
rather than of the probe.

Figure: `results/figs/fig_frequency_doseresponse_pls.png`, which names its probe in the
title, as does the ridge version.

## 3. The salience axis: an anecdote that survives normalisation

| entity set | median count in OLMo's training data |
|---|---|
| Assyrian rulers (our "obscure" cell B) | **1,494** |
| historical figures (our "salient" cell A) | **230** |

Taken flat, the Assyrian rulers are **6.5× more common** than the median
famous-Western-person we called salient. Sennacherib appears 449,892 times;
Ashurbanipal 160,086.

**That ratio is partly an artefact of name length, and the correction is worth stating
rather than hiding.** An exact-string count falls by roughly 5× per extra word:

| words in name | historical figures | median count |
|---|---|---|
| 1 | 250 | 4,078 |
| 2 | 3,355 | 877 |
| 3 | 2,109 | 151 |
| 4 | 976 | 43 |

The rulers are 25 one-word and 9 two-word names; the historical figures are mostly two
to four words. The one-word figures are also polluted by common English words — "John"
returns 416,569,300 — which inflates their end of the comparison.

Comparing each ruler only against figures with the **same number of words**, the median
ruler lands at the **42nd percentile**: 15 of 34 above the median figure, 19 below.

So the honest statement is not "the rulers are 6.5× more common". It is the weaker and
more robust one:

> **The two sets are drawn from the same exposure distribution. Normalising for name
> length does not open a gap in either direction.**

Which is the point that matters. The salience axis was a judgement call — "everyone
knows George Washington, nobody knows Tiglath-pileser" — and after controlling for the
one confound that could have manufactured the result, there is still no exposure gap to
explain why cell B is harder than cell A.

**The rulers are also not a homogeneous set**, which the "obscure" label implied. Their
counts span 0 to 449,892:

| ruler | count | percentile vs same-length figures |
|---|---|---|
| Sargon II | 89,026 | 94 |
| Sennacherib | 449,892 | 76 |
| Ashurbanipal | 160,086 | 70 |
| Ninurta-nadin-šumi | 1 | 14 |
| Nabû-šumu-libur | **0** | 0 |
| Kaššû-nadin-ahhe | **0** | 0 |

Two rulers do not appear in the training corpus at all. Treating these 34 names as one
salience level is the part of the framing that does not survive.

Combined with §2, the same conclusion arrives twice from different directions: exposure
is not the variable. The remaining candidates are what the text *says* — a Wikipedia
biography states a birth year in digits, while a Neo-Assyrian royal inscription does
not state a date at all — and how the model can index it. That is a claim about the
**form** of the evidence, not its **quantity**.

## 4. Decisions taken while analysing, and why

- **Held-out rows only.** Generalisation error, not fit. Counting a training-split
  entity would be an API call that can never enter the join, so sampling was restricted
  to `is_test` up front.
- **36 ambiguous names dropped** (73 rows). Two different Adalberts, six centuries
  apart, share one string; one count cannot be attributed to either.
- **250 single-token names excluded** (`--drop-short-names`). "John" returns
  416,569,300 — the English word, not the person. Keeping them barely moves the
  correlation (−0.026 vs −0.040) but stretches the x-axis by four orders of magnitude
  on entities whose counts are meaningless. Both runs are reproducible; the figure uses
  the clean one.
- **Spearman on log10(count+1).** Counts span nine orders of magnitude and the claim
  was only ever about monotonicity.

## 5. What this does and does not license

**Can be said:** OLMo-2-7B replicates the linear-probe result. Within its training
corpus, entity exposure has at most a marginal relationship to how well a probe recovers
an entity's date, and the entities the thesis calls obscure are in fact *more* frequent
than the ones it calls salient.

**Cannot be said:** that frequency is irrelevant — §2b shows a 34-year gap between the tails. This is one model, one
corpus, one target (death year), and a string-count proxy that cannot distinguish "the
person" from "the name". A count of 230 for a Wikidata-obscure person is already far
above zero — there may be no genuinely unseen entities in this sample, and a real
dose-response could live below the exposure floor we can observe.

**Should not be quoted:** the trained-minus-twin gap, for the reason in §2.

## 6. Reproducing

```bash
python count_frequencies.py     # needs outbound HTTPS; ~7.5k calls, ~1h
python analyze_frequency.py --drop-short-names
python plot_frequency_fig.py
```

`count_frequencies.py` is resumable and compacts its output on restart; a blocked run
loses nothing already counted.
