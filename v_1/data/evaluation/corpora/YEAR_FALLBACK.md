# Year Fallback for NB Rulers — Justification

**Date:** 2026-05-19
**Patch:** `v_1/src/corpus/03_build_orcc_corpus.py` — `_RULER_YEAR_FALLBACK` dict + `extract_year_with_fallback()`

---

## Problem

The `year` column in `orcc_corpus.parquet` is derived from the `sub_period` column in
`v_1/data/raw/chungrong/orcc_round1/royal_inscriptions.csv`. For four rulers, `sub_period`
is absent from the ORACC source:

| Ruler | Fragments | sub_period in ORACC |
|---|---|---|
| Nebuchadnezzar II | 87 | NaN (all) |
| Nabonidus | 68 | NaN (all) |
| Sargon II | 144 | NaN for 141 of 144 |
| Tiglath-pileser III | 75 | NaN for 4 of 75 |

Without a fix, these fragments have `year = NaN`, making year regression impossible for
~300 fragments including the two NB rulers in the Phase 0 balanced subset.

---

## Authority

Email from **Chung-rong Ni** (corpus provider), subject **"Adding the corpora of Royal
Inscriptions"**, **2026-05-19**:

> "For sub_periods of Nebuchadnezzar II and Nabonidus, Oracc did not include these
> information, but it is clearly Nebuchadnezzar II (605-562 BCE) and Nabonidus
> (556-539 BCE). Do I need to add them, or you can simply add these meta.
> Oracc does not have more granular per-inscription dates, and we cannot even know
> specific dates in many cases."

Chung-rong confirmed: (1) ORACC lacks this data, (2) per-inscription dates do not exist,
(3) we should add the reign-span labels ourselves. **No updated CSV is expected.**

---

## Fix

Hardcoded fallback in `03_build_orcc_corpus.py`. Convention: reign **end-year** (BCE),
matching the `min(digits)` convention applied to sub_period strings for NA rulers
(e.g. `"ca. 668-631"` → 631 = Ashurbanipal's end-year).

| Ruler | Fallback year | Reign |
|---|---|---|
| Nebuchadnezzar II | 562 | 605–562 BCE |
| Nabonidus | 539 | 556–539 BCE |
| Sargon II | 705 | 722–705 BCE (consistent with the 3 frags that had sub_period) |
| Tiglath-pileser III | 727 | 745–727 BCE (consistent with the 71 frags that had sub_period) |

The fallback only fires when `sub_period` is NaN — real ORACC values always take
precedence. Re-running `03_build_orcc_corpus.py` regenerates the parquet.

---

## Limitation

All fragments for a given ruler now share the same year value (the reign end-year).
Within-ruler year variance = 0 for all rulers except Esarhaddon (which has genuine
per-inscription dates from ORACC: 11 distinct values across 669–681 BCE).

This means **year regression on the balanced subset is chronologically shallow** —
the model effectively learns ruler identity, not intra-reign chronology. This limitation
is documented and accepted; per-inscription dating is unavailable from the source.
