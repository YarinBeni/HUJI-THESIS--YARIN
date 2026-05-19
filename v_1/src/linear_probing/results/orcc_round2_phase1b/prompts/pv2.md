---
variant: pv2_fewshot_k5
target: ruler, year
expected_output_schema: {"ruler": str, "year_bce": int, "confidence": float}
parse_strategy: json
fragment_span_delimiter: "literal delimiter <<FRAG>>...</FRAG>> wrapping {{fragment_text}} in the FINAL user turn"
---

## System prompt

```
You are an expert Assyriologist specializing in Akkadian royal inscriptions from the first millennium BCE. You read transliterated cuneiform texts in standard romanized notation. Your task is to identify the ruler who commissioned an inscription and estimate the approximate year BCE it was written, based on linguistic, onomastic, and formulaic evidence in the text.
```

## User prompt template

The user message consists of: (1) a task header, (2) five labeled in-context examples, (3) the target fragment wrapped in `<<FRAG>>`/`<</FRAG>>`, and (4) the output instruction.

```
This is an Akkadian royal inscription in transliteration. Identify the ruler who commissioned it and estimate the year BCE it was written.

Here are five examples with correct answers:

--- Example 1 ---
Inscription:
<<FRAG_EX1>>
{{example_1_text}}
<</FRAG_EX1>>
Answer: {"ruler": "{{example_1_ruler}}", "year_bce": {{example_1_year}}, "confidence": 0.95}

--- Example 2 ---
Inscription:
<<FRAG_EX2>>
{{example_2_text}}
<</FRAG_EX2>>
Answer: {"ruler": "{{example_2_ruler}}", "year_bce": {{example_2_year}}, "confidence": 0.95}

--- Example 3 ---
Inscription:
<<FRAG_EX3>>
{{example_3_text}}
<</FRAG_EX3>>
Answer: {"ruler": "{{example_3_ruler}}", "year_bce": {{example_3_year}}, "confidence": 0.95}

--- Example 4 ---
Inscription:
<<FRAG_EX4>>
{{example_4_text}}
<</FRAG_EX4>>
Answer: {"ruler": "{{example_4_ruler}}", "year_bce": {{example_4_year}}, "confidence": 0.95}

--- Example 5 ---
Inscription:
<<FRAG_EX5>>
{{example_5_text}}
<</FRAG_EX5>>
Answer: {"ruler": "{{example_5_ruler}}", "year_bce": {{example_5_year}}, "confidence": 0.95}

--- Target ---
Inscription:
<<FRAG>>
{{fragment_text}}
<</FRAG>>

Respond with a JSON object only — no prose before or after it:
{"ruler": "<ruler name>", "year_bce": <integer year BCE as a positive number>, "confidence": <float 0.0–1.0>}

Use the ruler's standard English-language name (e.g. "Ashurbanipal", "Sennacherib", "Esarhaddon", "Sargon II", "Nebuchadnezzar II", "Tiglath-pileser III", "Nabonidus", "Sin-sarru-iskun"). If you cannot determine the ruler or year, use null for that field.
```

**Placeholders:**
- `{{fragment_text}}`: tier-0-cleaned Akkadian transliteration of the TARGET fragment.
- `{{example_N_text}}`, `{{example_N_ruler}}`, `{{example_N_year}}`: filled from the held-out example pool (see below).

## Fragment span convention

The eval harness extracts activations at the last token of the **target** fragment span only — delimited by `<<FRAG>>` and `<</FRAG>>` (not `<<FRAG_EX*>>`). The `<<FRAG_EX*>>` delimiters are NOT pooled for activation extraction; they are in-context examples only.

Implementation: search for `<<FRAG>>` (not preceded by `_EX`) to locate the target span start. `span_end_token` = last token before `<</FRAG>>` (the one that immediately follows `<<FRAG>>...`).

Regex to locate target span in the decoded prompt string:
```python
import re
m = re.search(r'(?<!_EX\d)<<FRAG>>(.*?)<</FRAG>>', prompt, re.DOTALL)
# Or more robustly: find the LAST occurrence of <<FRAG>> in the string
```

## Held-out example pool specification

**CRITICAL:** Examples must NOT overlap with the Phase 0 balanced eval subset (168 fragments drawn from 8 rulers × 21 per draw). The orchestrator/eval-harness-builder is responsible for materializing the example pool AFTER Phase 0's `build_balanced_subset.py` has locked the eval fragment IDs.

Example selection rules:
1. Draw 5 fragments from the ORCC corpus whose fragment IDs are confirmed NOT in any MC draw's eval set.
2. Cover 5 DIFFERENT rulers — one example per ruler, drawn from the 8 eval rulers. Do not use the same ruler twice.
3. Prefer fragments with clear ruler-name mentions or distinctive formulaic language (makes examples educationally useful for the model).
4. Suggested candidate rulers for examples (subject to holdout availability): Ashurbanipal, Sennacherib, Esarhaddon, Sargon II, Tiglath-pileser III. (Nebuchadnezzar II, Nabonidus, Sin-sarru-iskun as alternates if the above are unavailable at right fragment IDs.)
5. Fragment text in examples should be truncated at 150 tokens if the original exceeds that, to keep total prompt length manageable.

**Token budget estimate:** 5 examples × ~100 tokens average = ~500 example tokens + ~50 boilerplate + target fragment (median 208 tokens per pipeline log) ≈ ~760 tokens total. Well within Qwen 2.5-7B's context window.

## Few-shot examples

The actual example texts are filled at inference time from the held-out pool. The example TEXTS below are illustrative only — do NOT use them directly; replace with real corpus fragments confirmed out of the eval set.

Illustrative ruler coverage (not literal text):
- Example 1: Ashurbanipal (~645 BCE) — titulary + campaign formula
- Example 2: Sennacherib (~700 BCE) — building inscription formula
- Example 3: Esarhaddon (~675 BCE) — restoration inscription
- Example 4: Sargon II (~715 BCE) — annals-style
- Example 5: Tiglath-pileser III (~740 BCE) — tribute formula

## Parse instructions

Same as pv1:
1. Extract first `{...}` JSON block; `json.loads()`.
2. Normalize `ruler` via canonical name table.
3. `year_bce` as positive integer.
4. `confidence` clipped to [0,1].
5. Fallback to regex on parse failure.
6. Log raw output for every fragment.
