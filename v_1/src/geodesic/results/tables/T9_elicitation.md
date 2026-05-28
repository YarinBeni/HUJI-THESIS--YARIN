# Test 9 — Direct elicitation (kp0 / kp1 / kp2)

**What it is:** *prompted* knowledge probes — does the LLM, asked plainly in
English, produce the correct king/date answer? Three variants:

- **kp0 — knows-reign-dates.** "When did <king> rule?" Scored as accuracy
  within a 50-year tolerance window.
- **kp1 — king -> date recall.** Given a historical period, can the model
  recall the kings that fit? Aggregate-recall = total_hits / total_targets.
- **kp2 — hallucination gate.** Given fabricated/uncertain names, does the
  model decline rather than confabulate? Headline = hallucination rate;
  gate passes if the rate falls below `gate_threshold`.

**Evaluation sizes are SMALL by design:** 8 questions per variant — these are
targeted king/period probes, not corpus-wide labels. Read the numbers as
sanity-check signal, not full benchmarks. The **PASS gate** is on kp2 only
(if the model can't suppress hallucinated dates it's a deal-breaker even when
kp0/kp1 look fine).

**CSV `T9_elicitation.csv`** — one row per (model, variant). Headline metric
varies per variant; `extra` carries auxiliary counters. Three models present:
qwen3_1b7, qwen3_8b, qwen3_32b.
