# P1 — single-variable erasure ladder (frozen-probe level) — 2026-09-02 · all 6 arms

**Question.** Three encoders and two text tiers agree that masking the ruler's
*name* does not reduce dating accuracy. So what does the representation date
by? Erase ONE metadata variable at a time with LEACE (fitted on the train
fold), read a ridge probe out on the frozen tier0 features through the SLA
protocol, and compare with the unerased probe. Positive control `year10`
(year deciles) must crush the signal; `ruler` is the joint upper rung (ICC=1).

## mc ρ after erasure (Δ vs nothing erased)

| erased | cunei400m akk | cunei400m eng | Llama-2-7B akk | Llama-2-7B eng | Qwen3-8B akk | Qwen3-8B eng |
|---|---|---|---|---|---|---|
| **none** | .447 | .417 | .356 | .360 | .259 | .395 |
| **provenance** (21) | .142 (−.30) | .144 (−.27) | **−.001 (−.36)** | .078 (−.28) | **−.198 (−.46)** | .019 (−.38) |
| period (6) | .246 (−.20) | .217 (−.20) | .219 (−.14) | .249 (−.11) | .160 (−.10) | .270 (−.13) |
| sub-genre / object (21) | .294 (−.15) | .301 (−.12) | .277 (−.08) | .223 (−.14) | .214 (−.04) | .187 (−.21) |
| length (6) | .423 (−.02) | .357 (−.06) | .255 (−.10) | .282 (−.08) | .193 (−.07) | .328 (−.07) |
| ruler (40) | .039 (−.41) | .000 (−.42) | −.042 (−.40) | −.045 (−.41) | −.023 (−.28) | −.044 (−.44) |
| year10 (7) — control | −.003 | −.177 | .044 | −.177 | .073 | −.098 |

Share of the linear dating signal removed by erasing provenance alone:
cunei 68 % / 65 %, Llama **100 %** / 78 %, Qwen > 100 % / 95 % (on Akkadian the
residual is *anti*-correlated with time).

## Reading

1. **Both controls behave.** Erasing year deciles or ruler identity removes the
   signal entirely (to ≈ 0 or slightly negative), in every arm.
2. **Provenance is the dominant carrier.** The find-spot one-hot (top-20 sites)
   removes two-thirds of the probe's signal on the Akkadian-native encoder and
   *all* of it on Llama. Period costs .10–.20, object type .04–.15, length ≤ .10.
   This is the P1 answer to "if not the name, then what": the frozen features
   date documents mainly through *where the object was found* (Nineveh /
   Babylon / Assur / Kalhu / Dur-Šarrukin map onto reigns), then through
   catalogue period.
3. **The encoder matters for what survives.** After provenance erasure the
   Akkadian-native encoder keeps .14 — a genuine, site-independent
   chronological residue — while the general LLMs keep nothing (Llama) or flip
   sign (Qwen). The v2 advantage of cuneiformBase-400m is therefore not only
   more signal but signal of a different kind.
4. **Anti-correlation after erasure (Qwen −.20)** is beyond the block null
   (±.02 for the mean-over-draws statistic). Not interpreted yet; candidate:
   a length/object-type confound running against time once site is removed.
   To be checked with the joint erasure (provenance + length).

## What this does and does not say about the method

This ladder is on the *frozen probe*. The head ladder (C5, running) trains the
Chrono-Barlow head on the erased features and asks whether its +.11–.15
advantage over ridge survives losing provenance. If it does, the head found
chronology the probe cannot see and that is not site; if it collapses, its
advantage was a better reading of the same site-coded signal.

## Caveats

* Erasure is linear (LEACE). A nonlinear head can in principle recover erased
  concepts; the head ladder is the test of that.
* Provenance one-hot covers the top-20 sites + "other"; 74 distinct sites.
* Readability-after column withheld: the first pass measured it across the
  ruler-grouped fold, where block-constant concepts make the number reflect
  distribution shift, not erasure; the second pass (within-train split,
  classes ≥ 10 docs, before → after) is running and will replace it.

Sources: `reports/tier0/ladder/*_ridge.md`, `scripts/erasure_ladder.py`,
`chrono/eval/erasure.py`.
