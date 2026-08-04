# OLMo frequency experiment — results

**Short version: the dose-response we went looking for is not there. A different
finding, which we were not looking for, is, and it matters more.**

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

## 3. The finding we were not looking for: "obscure" was the wrong label

| entity set | median count in OLMo's training data |
|---|---|
| Assyrian rulers (our "obscure" cell B) | **1,494** |
| historical figures (our "salient" cell A) | **230** |

The Assyrian rulers are **6.5× more common** in the training corpus than the median
famous-Western-person we called salient. Sennacherib appears 449,892 times;
Ashurbanipal 160,086.

This is a direct problem for how the thesis frames the A → B → B′ → C ladder. The
salience axis was a judgement call — "everyone knows George Washington, nobody knows
Tiglath-pileser" — and the corpus says that judgement is wrong, at least in the sense
of raw exposure. Whatever makes cell B harder than cell A, **it is not that the model
saw those entities less.**

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

**Cannot be said:** that frequency is irrelevant in general. This is one model, one
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
