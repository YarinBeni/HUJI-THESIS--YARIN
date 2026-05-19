---
variant: pv0_bare
target: ruler, year
expected_output_schema: {"ruler": str, "year_bce": int}
parse_strategy: regex
fragment_span_delimiter: "positional — fragment text is the entire user message content before the literal string '\n\nWho wrote this and when?'"
---

## System prompt

(none — no system message is sent for this variant)

## User prompt template

```
<<FRAG>>
{{fragment_text}}
<</FRAG>>

Who wrote this and when?
```

**Placeholder:** `{{fragment_text}}` is replaced verbatim with the tier-0-cleaned romanized Akkadian transliteration text for the fragment.

## Fragment span convention

The fragment text is wrapped in literal XML-style delimiters `<<FRAG>>` and `<</FRAG>>`. The eval harness MUST find the token positions of `<<FRAG>>` and `<</FRAG>>` in the tokenized input, then pool hidden states at the last token **inside** that span (i.e., the token immediately preceding `<</FRAG>>`). This is the CLS token used for re-probing.

Regex to locate span in decoded string:
```
<<FRAG>>(.*?)<</FRAG>>
```
Token-level: record `span_start_token` = index of first token of `{{fragment_text}}` content, `span_end_token` = last token before `<</FRAG>>`. Pool at `span_end_token`.

## Few-shot examples

None (this variant is the no-framing baseline).

## Parse instructions

Apply in order until a match is found:

1. **JSON block** — if the model output contains `{`, attempt `json.loads()` on the first JSON object. Extract `ruler` (str) and `year_bce` (int or str→int).

2. **Regex fallback** — search for:
   - Ruler: one of the 8 canonical names (case-insensitive): `Ashurbanipal`, `Sennacherib`, `Esarhaddon`, `Sargon II`, `Nebuchadnezzar II`, `Tiglath-pileser III`, `Nabonidus`, `Sin-sarru-iskun` (also accept `Sin-šarru-iškun`).
   - Year: pattern `(\d{3,4})\s*BCE` or `(\d{3,4})\s*B\.C\.E\.` → negate (store as negative int to match corpus convention) or store as positive with BCE flag.

3. **Unmatched** — record `ruler=null`, `year_bce=null`, flag as `parse_failed=true`.

**Canonical ruler name normalization table** (map free-text → corpus label):
```
"Ashurbanipal", "Assurbanipal", "Assur-bani-apli"  → "Ashurbanipal"
"Sennacherib", "Sin-ahhe-eriba"                     → "Sennacherib"
"Esarhaddon", "Asarhaddon", "Esarra-haddon"         → "Esarhaddon"
"Sargon", "Sargon II"                               → "Sargon II"
"Nebuchadnezzar", "Nebuchadrezzar", "Nebuchadnezzar II" → "Nebuchadnezzar II"
"Tiglath-pileser", "Tiglathpileser", "Tiglath-pileser III" → "Tiglath-pileser III"
"Nabonidus", "Nabu-naid"                            → "Nabonidus"
"Sin-sarru-iskun", "Sin-šarru-iškun", "Sinsharishkun" → "Sin-sarru-iskun"
```

Log the raw model output string alongside the parsed result for every fragment.
