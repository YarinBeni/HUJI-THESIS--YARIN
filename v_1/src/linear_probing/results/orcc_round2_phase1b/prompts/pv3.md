---
variant: pv3_cot
target: ruler, year
expected_output_schema: {"ruler": str, "year_bce": int, "confidence": float}
parse_strategy: json (extract LAST JSON block; ignore preceding reasoning text)
fragment_span_delimiter: "literal delimiter <<FRAG>>...</FRAG>> in the user message"
---

## System prompt

```
You are an expert Assyriologist specializing in Akkadian royal inscriptions from the first millennium BCE. You read transliterated cuneiform texts in standard romanized notation. Your task is to identify the ruler who commissioned an inscription and estimate the approximate year BCE it was written, based on linguistic, onomastic, and formulaic evidence in the text.
```

## User prompt template

```
This is an Akkadian royal inscription in transliteration. Reason briefly about the text, then give your answer as a JSON object.

Inscription:
<<FRAG>>
{{fragment_text}}
<</FRAG>>

Reason step by step in a few sentences about:
(1) Any ruler name or royal titulary visible in the text.
(2) Geographic or place-name references that indicate period or empire.
(3) Lexical or orthographic features that mark the text as early, middle, or late first-millennium BCE.
(4) Genre formulae (building inscriptions, annals, votive texts) that constrain the date range.

After your reasoning, output a JSON object on its own line — nothing after it:
{"ruler": "<ruler name>", "year_bce": <integer year BCE as a positive number>, "confidence": <float 0.0–1.0>}

Use the ruler's standard English-language name (e.g. "Ashurbanipal", "Sennacherib", "Esarhaddon", "Sargon II", "Nebuchadnezzar II", "Tiglath-pileser III", "Nabonidus", "Sin-sarru-iskun"). If you cannot determine the ruler or year, use null for that field.
```

**Placeholder:** `{{fragment_text}}` is replaced verbatim with the tier-0-cleaned romanized Akkadian transliteration for the fragment.

## Fragment span convention

The fragment text is wrapped in `<<FRAG>>` and `<</FRAG>>`. The eval harness pools hidden states at the last token of the fragment span (immediately before `<</FRAG>>`). The model's reasoning text and the final JSON block are generated AFTER this token and are therefore NOT captured in the activation extraction — only the fragment representation is probed.

`span_end_token` = last token of `{{fragment_text}}` content, before `<</FRAG>>`.

## Few-shot examples

None (zero-shot CoT — adding examples would inflate token budget significantly given the reasoning step).

## Parse instructions

1. Find the **last** `{...}` JSON block in the model output (the one that follows the reasoning). Do NOT use the first JSON block if reasoning accidentally contains JSON.
   ```python
   import re, json
   matches = list(re.finditer(r'\{[^{}]+\}', output, re.DOTALL))
   if matches:
       candidate = matches[-1].group()
       parsed = json.loads(candidate)
   ```
2. Extract `ruler`, `year_bce`, `confidence` as in pv1.
3. Normalize ruler via canonical name table.
4. **Store the reasoning text separately** — everything before the final JSON block — as `reasoning_text` in the output record. This is logged but not used for scoring.
5. If JSON parse fails: try regex on full output (ruler name + year pattern) as fallback.
6. Log raw output and reasoning text for every fragment.

**Reasoning budget guidance:** The prompt says "a few sentences." At inference time, if the model produces more than ~200 tokens of reasoning before the JSON, it is still valid — do not truncate model output during generation. Just parse the last JSON block regardless of reasoning length.
