# Phase 1a — Knowledge-Probe Aggregated Report

## VERDICT: Phase 1a is a candidate bottleneck (one or more sub-checks failed).

Failed sub-checks:
  - kp2 hallucination_rate 0.500 >= 0.30

### Headline metrics
  - kp0 accuracy:            1.000 (threshold >= 0.625, PASS)
  - kp1 aggregate_recall:    0.750 (threshold >= 0.50, PASS)
  - kp2 hallucination_rate:  0.500 (threshold < 0.30, FAIL)


## kp0 — "When did ruler X reign?" (8 real rulers)

- Tolerance: +/- 50 years
- Total: 8, Correct: 8, Parse errors: 0
- Accuracy (overall): 1.000
- Accuracy (scoreable only): 1.000
- Gate: accuracy >= 0.625? PASS

| ruler | true range | predicted range | hit/miss | parse_error |
|---|---|---|---|---|
| Ashurbanipal | 668-631 BCE (alt end 627) | 668-627 BCE | HIT | no |
| Sennacherib | 705-681 BCE | 705-681 BCE | HIT | no |
| Esarhaddon | 681-669 BCE | 681-669 BCE | HIT | no |
| Sargon II | 722-705 BCE | 722-705 BCE | HIT | no |
| Nebuchadnezzar II | 605-562 BCE | 605-562 BCE | HIT | no |
| Tiglath-pileser III | 745-727 BCE | 745-727 BCE | HIT | no |
| Nabonidus | 556-539 BCE | 556-539 BCE | HIT | no |
| Sîn-šarru-iškun | 627-612 BCE | 626-609 BCE | HIT | no |

## kp1 — "Which rulers reigned during period Y?" (2 periods)

- Total periods: 2, Parse errors: 0
- Aggregate recall over Phase-0 rulers: 0.750 (6/8)
- Gate: aggregate_recall >= 0.50? PASS

| period | expected Phase-0 rulers | model-returned set | recall |
|---|---|---|---|
| Neo-Assyrian | Sennacherib; Esarhaddon; Sargon II; Tiglath-pileser III; Ashurbanipal; Sîn-šarru-iškun | Ashurnasirpal II; Shalmaneser III; Assurdayan III; Tiglath-Pileser III; Shalmaneser IV; Adad-nirari III; Sargon II; Sennacherib; Esarhaddon; Assurbanipal | 0.667 |
| Neo-Babylonian | Nebuchadnezzar II; Nabonidus | Nabopolassar; Nebuchadnezzar II; Evil-Merodach; Nergal-sharezer; Nabonidus | 1.000 |

### kp1 extras (informational — rulers returned outside Phase-0 set)

| period | extras |
|---|---|
| Neo-Assyrian | Ashurnasirpal II; Shalmaneser III; Assurdayan III; Shalmaneser IV; Adad-nirari III; Assurbanipal |
| Neo-Babylonian | Nabopolassar; Evil-Merodach; Nergal-sharezer |

## kp2 — Hallucination probe (8 fake names)

- Total: 8, Scoreable: 8, Parse errors: 0
- Declined correctly: 4, Hallucinations: 4
- Hallucination rate (over scoreable): 0.500
- Gate (< 0.3): FAIL

| fake_name | declined | hallucinated_year_if_any | status |
|---|---|---|---|
| Nabu-šarru-iddina the Lesser | no | 626-605 | hallucination |
| Esarhaddon III | no | 681-669 | hallucination |
| Nergal-nasir-apli | yes |  | declined_correctly |
| Marduk-mukin-šarru | yes |  | declined_correctly |
| Sîn-eriba-bel | yes |  | declined_correctly |
| Assur-bel-mukin the Younger | no | 648-645 | hallucination |
| Tukulti-apil-iddina II | no | 801-793 | hallucination |
| Nabû-šumu-eriba the Elder | yes |  | declined_correctly |
