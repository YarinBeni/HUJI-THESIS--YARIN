---
variant: pv1_framed
target: ruler, year
expected_output_schema: {"ruler": str, "year_bce": int, "confidence": float}
parse_strategy: json
fragment_span_delimiter: "literal delimiter <<FRAG>>...</FRAG>> in the user message"
---

## System prompt

```
You are an expert Assyriologist specializing in Akkadian royal inscriptions from the first millennium BCE. You read transliterated cuneiform texts in standard romanized notation. Your task is to identify the ruler who commissioned an inscription and estimate the approximate year BCE it was written, based on linguistic, onomastic, and formulaic evidence in the text.
```

## User prompt template

```
This is an Akkadian royal inscription in transliteration. Identify the ruler who commissioned it and estimate the year BCE it was written.

Inscription:
<<FRAG>>
{{fragment_text}}
<</FRAG>>

Respond with a JSON object only — no prose before or after it:
{"ruler": "<ruler name>", "year_bce": <integer year BCE as a positive number>, "confidence": <float 0.0–1.0>}

Use the ruler's standard English-language name (e.g. "Ashurbanipal", "Sennacherib", "Esarhaddon", "Sargon II", "Nebuchadnezzar II", "Tiglath-pileser III", "Nabonidus", "Sin-sarru-iskun"). If you cannot determine the ruler or year, use null for that field.
```

**Placeholder:** `{{fragment_text}}` is replaced verbatim with the tier-0-cleaned romanized Akkadian transliteration for the fragment.

## Fragment span convention

The fragment text is wrapped in `<<FRAG>>` and `<</FRAG>>`. The eval harness locates these literal strings in the tokenized prompt, records the token index of the last token inside the span, and pools hidden states there for re-probing. The JSON response that follows is NOT part of the fragment span.

Token-level: `span_end_token` = index of last token of `{{fragment_text}}` content, immediately before `<</FRAG>>`.

## Few-shot examples

None (this is the zero-shot framed variant).

## Parse instructions

1. Find the first `{...}` JSON block in the model output.
2. `json.loads()` it. Extract:
   - `ruler` (str): normalize using the canonical name table from pv0.
   - `year_bce` (int): the corpus stores years as positive integers (year is the absolute year BCE). If the model outputs a negative number, take the absolute value.
   - `confidence` (float, 0.0–1.0): record as-is; clip to [0,1] if out of range.
3. If JSON parse fails, fall back to regex as in pv0.
4. Log raw output alongside parsed result for every fragment.

**Scoring note:** `year_bce` is compared against the corpus `year` field (positive integer = years before common era). Ruler is compared as a string against corpus `ruler` field after normalization.
