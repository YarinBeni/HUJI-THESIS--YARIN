# Class-imbalance / sample-size diagnosis of the balanced-MC king probe

Corpus: 1202 fragments, 41 rulers, 6 periods, 74 provenances.

Balanced-MC subset: **8 rulers × k=21 = 168 fragments/draw**, 200 draws. k is capped by the SMALLEST of the 8 classes (undersampling to the min).

## 1. Full-corpus fragments per ruler (top 15) + king-name coverage

| ruler | n frags | median year (BCE) | king cov (tier0) | in balanced-8 |
|---|---|---|---|---|
| Ashurbanipal | 268 | 631 | 0.47 | ✅ |
| Sennacherib | 237 | 681 | 0.67 | ✅ |
| Esarhaddon | 176 | 669 | 0.46 | ✅ |
| Sargon II | 144 | 705 | 0.53 | ✅ |
| Nebuchadnezzar II | 87 | 562 | 0.00 | ✅ |
| Tiglath-pileser III | 75 | 727 | 0.20 | ✅ |
| Nabonidus | 68 | 539 | 0.18 | ✅ |
| Sîn-šarru-iškun | 21 | 612 | 0.43 | ✅ |
| Nabopolassar | 15 | 605 | 0.00 |  |
| Shalmaneser V | 12 | 722 | 1.00 |  |
| Nebuchadnezzar I | 10 | 1104 | 0.00 |  |
| ribo | 9 |  | 0.00 |  |
| Neriglissar | 7 | 556 | 0.00 |  |
| Adad-apla-iddina | 7 | 1047 | 0.00 |  |
| Šamaš-šuma-ukin | 6 | 648 | 0.17 |  |

*33 of 41 rulers have < k=21 fragments and are excluded from balanced-MC entirely; the 33 excluded rulers hold 126 fragments (10% of the corpus).*

## 2. The 8 balanced rulers — the probe only ever sees these

| ruler | full n | median year (BCE) | king cov (tier0) | E[king-found in a k=21 draw] |
|---|---|---|---|---|
| Ashurbanipal | 268 | 631 | 0.47 | 9.9 / 21 |
| Sennacherib | 237 | 681 | 0.67 | 14.1 / 21 |
| Esarhaddon | 176 | 669 | 0.46 | 9.7 / 21 |
| Sargon II | 144 | 705 | 0.53 | 11.1 / 21 |
| Nebuchadnezzar II | 87 | 562 | 0.00 | 0.0 / 21 |
| Tiglath-pileser III | 75 | 727 | 0.20 | 4.2 / 21 |
| Nabonidus | 68 | 539 | 0.18 | 3.7 / 21 |
| Sîn-šarru-iškun | 21 | 612 | 0.43 | 9.0 / 21 |

Distinct median-year values among the 8 rulers: **8** — i.e. the regression target `year` is essentially an **8-level step function of ruler identity**.

## 3. Effective per-draw sample: `mean` vs `king_last`/`king_mean`

- **mean pool:** all 168 fragments/draw, all 8 ruler-groups, balanced 21/ruler.
- **king pool:** only name-found fragments survive → ~**62 fragments/draw** (≈37% of mean), and only ~**7/8 ruler-groups** contribute ≥1 point on average.
- Rulers nearly absent from the king pool (E[found] < 3 per draw): Nebuchadnezzar II — mostly Neo-Babylonian admin that never name the king.

With GroupKFold-by-ruler (n_splits=5) over ~7 surviving groups and only a handful of distinct year values, each test fold holds 1–2 rulers = 1–2 distinct years. Spearman on a fold with one year is undefined (the `ConstantInputWarning` in J6/J3r logs); with two it collapses to 'are the two groups separated?'. That is why king_last is **high AND high-variance** (±0.3–0.4) and why an **untrained/random** model scores ~0.64: the name token is a near one-hot ruler id, and year is a function of ruler, so any pooling that reads the name token trivially recovers year — no learned chronology needed.

## 4. Period & provenance imbalance (whole corpus)

### period
| period | n | share |
|---|---|---|
| Neo-Assyrian | 939 | 78.1% |
| Neo-Babylonian | 217 | 18.1% |
| Middle Babylonian | 29 | 2.4% |
| nan | 9 | 0.7% |
| wall slab (with reliefs) | 4 | 0.3% |
| Achaemenid | 3 | 0.2% |
| Hellenistic | 1 | 0.1% |

### provenance (top 12)
| provenance | n | share |
|---|---|---|
| Kuyunjik (Nineveh) | 497 | 41.3% |
| Babylon | 121 | 10.1% |
| Qalat Sherqat (Assur) | 99 | 8.2% |
| Nimrud (Kalhu) | 98 | 8.2% |
| Khorsabad (Dur-Šarrukin) | 59 | 4.9% |
| Babylonia | 30 | 2.5% |
| Sippar | 29 | 2.4% |
| Luristan | 28 | 2.3% |
| Borsippa | 24 | 2.0% |
| Ur | 21 | 1.7% |
| Uruk | 18 | 1.5% |
| Babylon (Bābili) | 14 | 1.2% |

*40 provenances have a single fragment; the top-3 sites cover 60% of the corpus.*
