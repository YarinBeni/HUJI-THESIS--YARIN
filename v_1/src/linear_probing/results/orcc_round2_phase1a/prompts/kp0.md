---
variant: kp0
target: ruler_reign_dates
expected_output_schema: {"start_year": int, "end_year": int, "confidence": "high" | "low", "declined": bool}
parse_strategy: json
notes: >
  Years are expressed as positive integers representing BCE dates
  (e.g. 668 means 668 BCE). start_year >= end_year because we count
  backward (Ashurbanipal: start=668, end=627). The "declined" flag
  is false for real rulers; it is the hallucination-control field
  used by kp2 (same template).
---

## System prompt

You are a precise reference assistant for ancient Near Eastern history.
Answer only from established historical knowledge. When asked about a ruler's
reign dates, return a JSON object and nothing else. Do not include markdown
fences, prose, or explanation outside the JSON.

## User prompt template

When did the Akkadian/Assyrian/Babylonian ruler {ruler} reign?
Return ONLY a JSON object with these fields:
  "start_year": the year BCE the reign began (integer, e.g. 668)
  "end_year": the year BCE the reign ended (integer, e.g. 627)
  "confidence": "high" if you are certain, "low" if approximate
  "declined": false

If you do not know this ruler at all, return:
  {"start_year": null, "end_year": null, "confidence": "low", "declined": true}

## Few-shot examples

These examples illustrate good responses. They are held out from the eval set
(not among the 8 Phase-0 rulers); do not use them as eval cases.

Query: When did the Akkadian/Assyrian/Babylonian ruler Assurnasirpal II reign?
Response:
{"start_year": 883, "end_year": 859, "confidence": "high", "declined": false}

Query: When did the Akkadian/Assyrian/Babylonian ruler Shalmaneser III reign?
Response:
{"start_year": 858, "end_year": 824, "confidence": "high", "declined": false}

Query: When did the Akkadian/Assyrian/Babylonian ruler Nebuchadnezzar I reign?
Response:
{"start_year": 1125, "end_year": 1104, "confidence": "high", "declined": false}

## Parse instructions

1. Strip leading/trailing whitespace from the model output.
2. If the output starts with a `{`, parse directly as JSON.
3. If the output is wrapped in triple-backtick fences (```json ... ``` or
   ``` ... ```), extract the content between the fences and parse as JSON.
4. Required fields after parsing:
     - start_year: int or null
     - end_year: int or null
     - confidence: "high" or "low"
     - declined: bool
5. Scoring: a prediction is CORRECT if the true reign year (start or end)
   falls within [min(start_year, end_year) - 50, max(start_year, end_year) + 50].
   Use a ±50-year tolerance per Yarin's call (2026-05-19): we want to detect
   whether Qwen has any temporal anchor for these rulers, not pin down exact
   dates. A ±5 tolerance would fail on any minor confusion (e.g. Esarhaddon
   vs. Sennacherib, only ~20 yr apart). ±50 keeps the within-empire/-period
   answer "correct" and only penalizes wholly wrong-millennium answers.
6. If JSON is malformed or required fields are missing, log as parse_error=true
   and skip from scoring (count separately for error-rate reporting).
