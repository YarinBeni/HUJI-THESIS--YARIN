# OLMo frequency experiment — the short version

**Status: NOT APPROVED. Nothing runs until you say go.**
Long version with links, file paths and acceptance checks: `PLAN_detailed.md`.

---

## The idea in one line

Right now we say an entity is either **salient** or **obscure**. That is a guess.
This experiment replaces the guess with a **number**: how many times the entity actually
appeared in the model's training data.

## Why it needs OLMo specifically

To count how often "Ashurbanipal" appears in a model's training data, we need a model
whose training data is public. Llama and Qwen do not publish theirs. **OLMo does** — open
weights *and* open corpus. So OLMo is the only arm where this question is answerable at
all. Counting in some other corpus and probing a different model would prove nothing.

## What we do

1. Add OLMo as one more arm, plus its untrained twin. **No new protocols** — it goes
   through the exact same pipeline as the other 15 arms.
2. For each entity, count how often it appears in OLMo's training data (free web API,
   no local index needed).
3. Plot: **x = how often the entity appears in training, y = how badly the probe dates
   it.** One dot per entity.

## What we expect to see

A downhill line: frequent entities are dated well, rare entities badly.

The **untrained twin is the check** — it should show *no* such line. If both show it,
the effect is about the data, not about what the model learned.

## Deliverables

| # | thing | where |
|---|---|---|
| 1 | OLMo arm in the existing figures | it just appears, new teal colour |
| 2 | `entity_counts.csv` — every entity + its training-data count | `results/` |
| 3 | **The dose-response figure** — the actual deliverable | `results/figs/fig_frequency_doseresponse.png` |
| 4 | `RESULTS.md` — the correlation number, trained vs twin, one-paragraph verdict | here |

## The one trap, and how we handle it

Old entities are **both** rarer **and** harder to date. So a downhill line might just be
"old things are hard", not "rare things are hard". We report the correlation **within
century bins** as well as overall. If it survives that, it is real.

## Cost

| step | where | rough time |
|---|---|---|
| extract OLMo activations | GPU | a few hours |
| probe (cell A + cell B) | CPU | ~1 hour |
| count frequencies | web API, no cluster | ~1 hour, rate-limited |
| analyse + figure | laptop | minutes |

Roughly **one day**, mostly waiting on the GPU. Nothing here is expensive or risky.

## What could go wrong

- **OLMo's tokenizer breaks our name-span logic.** Cheap to test first on ten ruler
  names before committing GPU time.
- **String counts are noisy for short or common names.** Flagged per row, and spot-checked
  by hand against the WIMBD web app.
- **The count index does not match the exact checkpoint's training mix.** Then the causal
  link is broken and the result is not worth reporting — so this gets verified *before*
  the runs, and stated in `RESULTS.md`.

## Order of work

1. Tokenizer smoke test (10 names). ← *stop here if it fails*
2. Register the arm, extract, probe. Check OLMo behaves like the other arms first.
3. Only then: counts, analysis, figure.

Step 2 has to look sane before any frequency claim is made. If OLMo does not reproduce
the ordinary result on salient entities, the frequency curve means nothing.

---

**To approve:** say go, and I start at step 1.
