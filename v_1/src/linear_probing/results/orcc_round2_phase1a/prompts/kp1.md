---
variant: kp1
target: period_to_rulers
expected_output_schema: {"period": str, "rulers": [str], "confidence": "high" | "low"}
parse_strategy: json
notes: >
  period is one of: "Neo-Assyrian", "Neo-Babylonian".
  rulers is an ordered list (chronological) of ruler names as romanized
  Akkadian with diacritics, matching our dataset's canonical spelling.
  Eval checks set intersection against the ground-truth 8-ruler list
  (not precision over all rulers, which would penalize correctly naming
  rulers outside our 8 — those are marked is_extra=true in scoring).
---

## System prompt

You are a precise reference assistant for ancient Near Eastern history.
Answer only from established historical knowledge. When asked which rulers
reigned during a given Mesopotamian period, return a JSON object and nothing
else. Do not include markdown fences, prose, or explanation outside the JSON.

## User prompt template

Which rulers reigned during the {period} period in Mesopotamia?
List only rulers who held the title of king (not regents or officials).
Return ONLY a JSON object with these fields:
  "period": "{period}"
  "rulers": an array of ruler names (romanized, chronological order)
  "confidence": "high" if you are certain of the list, "low" if approximate

## Few-shot examples

These examples illustrate good responses. They use different periods to avoid
leaking the eval targets.

Query: Which rulers reigned during the Old Babylonian period in Mesopotamia?
Response:
{"period": "Old Babylonian", "rulers": ["Sumu-abum", "Sumu-la-El", "Sabium", "Apil-Sin", "Sin-muballit", "Hammurabi", "Samsu-iluna", "Abi-eshuh", "Ammi-ditana", "Ammi-saduqa", "Samsu-ditana"], "confidence": "high"}

Query: Which rulers reigned during the Middle Assyrian period in Mesopotamia?
Response:
{"period": "Middle Assyrian", "rulers": ["Ashur-uballit I", "Enlil-nirari", "Arik-den-ili", "Adad-nirari I", "Shalmaneser I", "Tukulti-Ninurta I", "Ashur-nadin-apli", "Ashur-nirari III", "Enlil-kudurri-usur", "Ninurta-apal-Ekur", "Ashur-Dan I", "Ninurta-tukulti-Ashur", "Mutakkil-Nusku", "Ashur-resh-ishi I", "Tiglath-pileser I", "Asharid-apal-Ekur", "Ashur-bel-kala", "Eriba-Adad II", "Shamshi-Adad IV", "Ashurnasirpal I", "Shalmaneser II", "Ashur-nirari IV", "Ashur-rabi II", "Ashur-resh-ishi II", "Tiglath-pileser II", "Ashur-Dan II"], "confidence": "high"}

## Parse instructions

1. Strip leading/trailing whitespace from the model output.
2. If the output starts with a `{`, parse directly as JSON.
3. If the output is wrapped in triple-backtick fences, extract content and
   parse as JSON.
4. Required fields after parsing:
     - period: str
     - rulers: list of str (may be empty if model says unknown)
     - confidence: "high" or "low"
5. Scoring:
     - For each of the 8 Phase-0 rulers that belongs to the queried period,
       check if it appears in rulers (case-insensitive, diacritic-normalized).
     - Recall = |Phase-0 rulers found| / |Phase-0 rulers in that period|.
     - Precision is not the primary metric (we expect the model to name rulers
       beyond our 8); flag extras as is_extra=true for qualitative inspection.
6. Normalize diacritics before matching: e.g. "Sin-sarru-iskun" matches
   "Sîn-šarru-iškun". Use unicodedata.normalize('NFKD') and strip combining
   characters for comparison only (keep originals in records).
7. If JSON is malformed, log parse_error=true.
