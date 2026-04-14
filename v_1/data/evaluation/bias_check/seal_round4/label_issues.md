# SEAL Confusion Matrix Analysis — Label Quality Report

Generated: 2026-04-09
Purpose: identify systematic misclassifications that may indicate label quality
issues, to be reviewed with Chungrong before requesting CSV corrections.

How to read this: Each confusion pair shows how many fragments with true label A
were predicted as B. Fragment IDs are the actual IDs from the source CSVs.
Where the same fragments are confused in both tier0 and maximal, the confusion
is robust (not a cleaning artifact). Where it only appears in maximal, stripping
writing conventions is causing the confusion.

---

## Task: `period`

Classes: 6 | N: 383 | k: 2 | Acc (tier0/maximal): 0.838/0.770 | Macro-F1 (tier0/maximal): 0.608/0.464

### Confusion matrix (tier0)

| True/Pred | `Archaic/Old Akkadian/Ebla` | `Late Babylonian` | `Middle Babylonian/Assyrian` | `Neo-Assyrian and Late Babylonian` | `Old Assyrian` | `Old Babylonian` |
|---|---|---|---|---|---|---|
| `Archaic/Old Akkadian/Ebla` | 0 | 0 | 0 | 0 | 1 | 1 |
| `Late Babylonian` | 0 | 38 | 0 | 0 | 0 | 0 |
| `Middle Babylonian/Assyrian` | 0 | 11 | 30 | 1 | 0 | 23 |
| `Neo-Assyrian and Late Babylonian` | 0 | 3 | 2 | 39 | 0 | 0 |
| `Old Assyrian` | 0 | 0 | 0 | 0 | 2 | 3 |
| `Old Babylonian` | 0 | 2 | 14 | 0 | 1 | 212 |

### Confusion matrix (maximal)

| True/Pred | `Archaic/Old Akkadian/Ebla` | `Late Babylonian` | `Middle Babylonian/Assyrian` | `Neo-Assyrian and Late Babylonian` | `Old Assyrian` | `Old Babylonian` |
|---|---|---|---|---|---|---|
| `Archaic/Old Akkadian/Ebla` | 0 | 0 | 0 | 0 | 0 | 2 |
| `Late Babylonian` | 0 | 30 | 1 | 7 | 0 | 0 |
| `Middle Babylonian/Assyrian` | 0 | 6 | 21 | 1 | 0 | 37 |
| `Neo-Assyrian and Late Babylonian` | 0 | 6 | 3 | 34 | 0 | 1 |
| `Old Assyrian` | 0 | 0 | 0 | 0 | 0 | 5 |
| `Old Babylonian` | 0 | 1 | 17 | 1 | 0 | 210 |

### Top confusions with fragment IDs

#### `Middle Babylonian/Assyrian` predicted as `Old Babylonian`

- tier0: 23x  |  maximal: 37x
- New in maximal (cleaning removes discriminating signal): [1561, 1563, 1577, 1598, 1642, 7234, 7237, 7243, 7251, 7254, 7264, 7265, 7450, 7451, 7547, 27122, 27124]
- Resolved by maximal cleaning (writing conventions were the cue): [7233, 7242, 7580]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1520 | seal | Middle Babylonian/Assyrian | epics and myths | Unknown | 199 |
| 1557 | seal | Middle Babylonian/Assyrian | epics and myths | Akhetaten | 369 |
| 1559 | seal | Middle Babylonian/Assyrian | epics and myths | Hattuša | 27 |
| 1562 | seal | Middle Babylonian/Assyrian | epics and myths | Nippur | 48 |
| 1567 | seal | Middle Babylonian/Assyrian | epics and myths | Hattuša | 151 |
| 1582 | seal | Middle Babylonian/Assyrian | epics and myths | Akhetaten | 336 |
| 1597 | seal | Middle Babylonian/Assyrian | epics and myths | Nippur | 194 |
| 1631 | seal | Middle Babylonian/Assyrian | love literature | Unknown | 141 |
| 1635 | seal | Middle Babylonian/Assyrian | love literature | Nippur | 144 |
| 1638 | seal | Middle Babylonian/Assyrian | love literature | Nippur | 225 |
| 1641 | seal | Middle Babylonian/Assyrian | catalogues | Sippar | 24 |
| 1644 | seal | Middle Babylonian/Assyrian | literary letters | Unknown | 62 |
| 1780 | seal | Middle Babylonian/Assyrian | wisdom literature | Nippur | 134 |
| 1784 | seal | Middle Babylonian/Assyrian | wisdom literature | Babylon | 32 |
| 7233 | seal | Middle Babylonian/Assyrian | incantations | Hattuša | 23 |
| 7242 | seal | Middle Babylonian/Assyrian | incantations | Assur | 19 |
| 7263 | seal | Middle Babylonian/Assyrian | incantations | Hattuša | 2 |
| 7280 | seal | Middle Babylonian/Assyrian | incantations | Babylon;Sippar | 213 |
| 7284 | seal | Middle Babylonian/Assyrian | incantations | Hattuša | 24 |
| 7295 | seal | Middle Babylonian/Assyrian | incantations | Hattuša | 22 |
| 7296 | seal | Middle Babylonian/Assyrian | incantations | Hattuša | 10 |
| 7580 | seal | Middle Babylonian/Assyrian | hymns and prayers | Babylon | 12 |
| 7590 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 103 |

#### `Old Babylonian` predicted as `Middle Babylonian/Assyrian`

- tier0: 14x  |  maximal: 17x
- New in maximal (cleaning removes discriminating signal): [1542, 1626, 7052, 7101, 7121, 7123, 7126, 7175, 7202, 7214, 7494, 7500]
- Resolved by maximal cleaning (writing conventions were the cue): [1549, 1625, 1630, 1765, 7082, 7189, 7199, 7211, 30351]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1549 | seal | Old Babylonian | epics and myths | Unknown | 15 |
| 1625 | seal | Old Babylonian | love literature | Kiš | 12 |
| 1630 | seal | Old Babylonian | love literature | Kiš | 8 |
| 1765 | seal | Old Babylonian | wisdom literature | Unknown | 12 |
| 7082 | seal | Old Babylonian | incantations | Unknown | 42 |
| 7089 | seal | Old Babylonian | incantations | Unknown | 4 |
| 7119 | seal | Old Babylonian | incantations | Unknown | 9 |
| 7189 | seal | Old Babylonian | incantations | Unknown | 15 |
| 7194 | seal | Old Babylonian | incantations | Unknown | 4 |
| 7199 | seal | Old Babylonian | incantations | Unknown | 1 |
| 7208 | seal | Old Babylonian | incantations | Larsa area | 3 |
| 7211 | seal | Old Babylonian | incantations | Larsa area | 10 |
| 26862 | seal | Old Babylonian | incantations | Larsa area | 4 |
| 30351 | seal | Old Babylonian | incantations | Unknown | 6 |

#### `Middle Babylonian/Assyrian` predicted as `Late Babylonian`

- tier0: 11x  |  maximal: 6x
- New in maximal (cleaning removes discriminating signal): [7235, 7482, 7562]
- Resolved by maximal cleaning (writing conventions were the cue): [1561, 1563, 1598, 1600, 1637, 1642, 7232, 7547]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1561 | seal | Middle Babylonian/Assyrian | epics and myths | Ugarit | 60 |
| 1563 | seal | Middle Babylonian/Assyrian | wisdom literature | Emar;Ugarit | 149 |
| 1598 | seal | Middle Babylonian/Assyrian | epics and myths | Unknown | 127 |
| 1600 | seal | Middle Babylonian/Assyrian | epics and myths | Assur;Nineveh | 294 |
| 1637 | seal | Middle Babylonian/Assyrian | love literature | Assur | 68 |
| 1642 | seal | Middle Babylonian/Assyrian | catalogues | Assur | 774 |
| 7232 | seal | Middle Babylonian/Assyrian | incantations | Unknown | 65 |
| 7257 | seal | Middle Babylonian/Assyrian | incantations | Ugarit | 60 |
| 7262 | seal | Middle Babylonian/Assyrian | incantations | Emar | 78 |
| 7545 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 4 |
| 7547 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 21 |

#### `Neo-Assyrian and Late Babylonian` predicted as `Late Babylonian`

- tier0: 3x  |  maximal: 6x
- New in maximal (cleaning removes discriminating signal): [31697, 31976, 32592, 32978, 33124]
- Resolved by maximal cleaning (writing conventions were the cue): [31727, 33562]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 31727 | dll | Neo-Assyrian and Late Babylonian | rituals | Babylon | 56 |
| 33322 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 50 |
| 33562 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 7 |

#### `Old Assyrian` predicted as `Old Babylonian`

- tier0: 3x  |  maximal: 5x
- New in maximal (cleaning removes discriminating signal): [7216, 7223]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7217 | seal | Old Assyrian | incantations | Kaniš | 19 |
| 7222 | seal | Old Assyrian | incantations | Kaniš | 29 |
| 7226 | seal | Old Assyrian | incantations | Kaniš | 9 |

#### `Neo-Assyrian and Late Babylonian` predicted as `Middle Babylonian/Assyrian`

- tier0: 2x  |  maximal: 3x
- New in maximal (cleaning removes discriminating signal): [31687, 32774]
- Resolved by maximal cleaning (writing conventions were the cue): [31713]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 31713 | dll | Neo-Assyrian and Late Babylonian | commentary | Nineveh | 15 |
| 33543 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 5 |

#### `Old Babylonian` predicted as `Late Babylonian`

- tier0: 2x  |  maximal: 1x
- New in maximal (cleaning removes discriminating signal): [1629]
- Resolved by maximal cleaning (writing conventions were the cue): [1542, 7520]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1542 | seal | Old Babylonian | epics and myths | Unknown | 130 |
| 7520 | seal | Old Babylonian | hymns and prayers | Sippar | 1382 |

#### `Archaic/Old Akkadian/Ebla` predicted as `Old Assyrian`

- tier0: 1x  |  maximal: 0x
- Resolved by maximal cleaning (writing conventions were the cue): [7026]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7026 | seal | Archaic/Old Akkadian/Ebla | incantations | Kiš | 88 |

#### `Archaic/Old Akkadian/Ebla` predicted as `Old Babylonian`

- tier0: 1x  |  maximal: 2x
- New in maximal (cleaning removes discriminating signal): [7026]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7042 | seal | Archaic/Old Akkadian/Ebla | incantations | Nippur | 13 |

#### `Middle Babylonian/Assyrian` predicted as `Neo-Assyrian and Late Babylonian`

- tier0: 1x  |  maximal: 1x
- New in maximal (cleaning removes discriminating signal): [1637]
- Resolved by maximal cleaning (writing conventions were the cue): [1577]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1577 | seal | Middle Babylonian/Assyrian | epics and myths | Ugarit | 178 |

### New confusions under maximal cleaning only

These pairs are clean under tier0 but emerge after aggressive cleaning,
suggesting writing conventions were the only thing separating them.

- `Late Babylonian` → `Neo-Assyrian and Late Babylonian`: 7x  IDs: [34571, 35344, 35514, 36163, 36280, 38658, 39654]

---

## Task: `genre`

Classes: 16 | N: 384 | k: 2 | Acc (tier0/maximal): 0.599/0.365 | Macro-F1 (tier0/maximal): 0.361/0.269

### Confusion matrix (tier0)

| True/Pred | `catalogues` | `chronicles` | `commentary` | `epics` | `epics and myths` | `funerary texts` | `hymns and prayers` | `incantations` | `lamentations` | `literary letters` | `love literature` | `lyrics` | `miscellaneous` | `prophecies` | `rituals` | `wisdom literature` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `catalogues` | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `chronicles` | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| `commentary` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |
| `epics` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 4 | 0 |
| `epics and myths` | 0 | 0 | 0 | 1 | 34 | 0 | 2 | 3 | 3 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| `funerary texts` | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hymns and prayers` | 0 | 1 | 0 | 0 | 7 | 0 | 12 | 5 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| `incantations` | 0 | 0 | 0 | 1 | 4 | 2 | 11 | 114 | 9 | 0 | 3 | 0 | 6 | 2 | 2 | 7 |
| `lamentations` | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `literary letters` | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 2 | 5 | 0 | 0 | 0 | 0 | 4 | 1 |
| `love literature` | 0 | 0 | 0 | 1 | 5 | 1 | 2 | 3 | 4 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| `lyrics` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 0 | 2 | 0 |
| `miscellaneous` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `prophecies` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| `rituals` | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 18 | 0 |
| `wisdom literature` | 1 | 0 | 0 | 0 | 8 | 0 | 0 | 4 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 1 |

### Confusion matrix (maximal)

| True/Pred | `catalogues` | `chronicles` | `commentary` | `epics` | `epics and myths` | `funerary texts` | `hymns and prayers` | `incantations` | `lamentations` | `literary letters` | `love literature` | `lyrics` | `miscellaneous` | `prophecies` | `rituals` | `wisdom literature` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `catalogues` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `chronicles` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 0 |
| `commentary` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 |
| `epics` | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 | 0 |
| `epics and myths` | 0 | 0 | 1 | 0 | 21 | 1 | 1 | 3 | 7 | 2 | 1 | 2 | 2 | 1 | 0 | 6 |
| `funerary texts` | 0 | 0 | 1 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hymns and prayers` | 0 | 0 | 1 | 0 | 2 | 1 | 12 | 2 | 2 | 0 | 3 | 0 | 2 | 2 | 0 | 1 |
| `incantations` | 8 | 3 | 0 | 0 | 15 | 6 | 16 | 44 | 11 | 7 | 9 | 3 | 8 | 4 | 6 | 21 |
| `lamentations` | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `literary letters` | 0 | 2 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 2 | 0 |
| `love literature` | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 1 | 8 | 1 | 0 | 0 | 0 | 5 |
| `lyrics` | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 22 | 0 | 0 | 8 | 0 |
| `miscellaneous` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `prophecies` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `rituals` | 0 | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 11 | 0 |
| `wisdom literature` | 0 | 0 | 0 | 1 | 2 | 0 | 1 | 2 | 2 | 2 | 2 | 0 | 1 | 0 | 0 | 4 |

### Top confusions with fragment IDs

#### `incantations` predicted as `hymns and prayers`

- tier0: 11x  |  maximal: 16x
- New in maximal (cleaning removes discriminating signal): [7042, 7056, 7089, 7128, 7135, 7145, 7160, 7177, 7210, 7234, 7280, 7450]
- Resolved by maximal cleaning (writing conventions were the cue): [7082, 7109, 7120, 7242, 7248, 7264, 7451]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7082 | seal | Old Babylonian | incantations | Unknown | 42 |
| 7109 | seal | Old Babylonian | incantations | Larsa area | 29 |
| 7120 | seal | Old Babylonian | incantations | Unknown | 28 |
| 7123 | seal | Old Babylonian | incantations | Unknown | 10 |
| 7193 | seal | Old Babylonian | incantations | Larsa area | 59 |
| 7214 | seal | Old Babylonian | incantations | Isin | 3 |
| 7242 | seal | Middle Babylonian/Assyrian | incantations | Assur | 19 |
| 7248 | seal | Middle Babylonian/Assyrian | incantations | Dūr-Kurigalzu | 4 |
| 7250 | seal | Middle Babylonian/Assyrian | incantations | Ur | 18 |
| 7264 | seal | Middle Babylonian/Assyrian | incantations | Ugarit | 21 |
| 7451 | seal | Middle Babylonian/Assyrian | incantations | Hattuša | 8 |

#### `incantations` predicted as `lamentations`

- tier0: 9x  |  maximal: 11x
- New in maximal (cleaning removes discriminating signal): [7099, 7153, 7170, 7175, 7176, 7284, 7604, 13431]
- Resolved by maximal cleaning (writing conventions were the cue): [7046, 7056, 7057, 7112, 7171, 7207]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7046 | seal | Old Babylonian | incantations | Ur | 36 |
| 7056 | seal | Old Babylonian | incantations | Unknown | 50 |
| 7057 | seal | Old Babylonian | incantations | Unknown | 23 |
| 7063 | seal | Old Babylonian | incantations | Unknown | 39 |
| 7112 | seal | Old Babylonian | incantations | Unknown | 41 |
| 7147 | seal | Old Babylonian | incantations | Isin | 35 |
| 7148 | seal | Old Babylonian | incantations | Isin | 21 |
| 7171 | seal | Old Babylonian | incantations | Larsa area | 22 |
| 7207 | seal | Old Babylonian | incantations | Larsa area | 10 |

#### `wisdom literature` predicted as `epics and myths`

- tier0: 8x  |  maximal: 2x
- New in maximal (cleaning removes discriminating signal): [1669]
- Resolved by maximal cleaning (writing conventions were the cue): [1563, 1655, 1738, 1751, 1754, 1768, 1780]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1563 | seal | Middle Babylonian/Assyrian | wisdom literature | Emar;Ugarit | 149 |
| 1655 | seal | Old Babylonian | wisdom literature | Ur | 182 |
| 1736 | seal | Old Babylonian | wisdom literature | Sippar | 57 |
| 1738 | seal | Old Babylonian | wisdom literature | Unknown;Larsa | 1045 |
| 1751 | seal | Old Babylonian | wisdom literature | Larsa | 395 |
| 1754 | seal | Old Babylonian | wisdom literature | Unknown | 120 |
| 1768 | seal | Old Babylonian | wisdom literature | Uruk | 110 |
| 1780 | seal | Middle Babylonian/Assyrian | wisdom literature | Nippur | 134 |

#### `hymns and prayers` predicted as `epics and myths`

- tier0: 7x  |  maximal: 2x
- New in maximal (cleaning removes discriminating signal): [7494, 26698]
- Resolved by maximal cleaning (writing conventions were the cue): [7500, 7501, 7502, 7508, 7520, 7590, 26600]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7500 | seal | Old Babylonian | hymns and prayers | Mari | 119 |
| 7501 | seal | Old Babylonian | hymns and prayers | Unknown | 131 |
| 7502 | seal | Old Babylonian | hymns and prayers | Unknown | 148 |
| 7508 | seal | Old Babylonian | hymns and prayers | Larsa | 81 |
| 7520 | seal | Old Babylonian | hymns and prayers | Sippar | 1382 |
| 7590 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 103 |
| 26600 | seal | Old Babylonian | hymns and prayers | Unknown | 293 |

#### `incantations` predicted as `wisdom literature`

- tier0: 7x  |  maximal: 21x
- New in maximal (cleaning removes discriminating signal): [7053, 7077, 7081, 7082, 7097, 7140, 7158, 7159, 7171, 7186, 7198, 7201, 7204, 7222, 7295]
- Resolved by maximal cleaning (writing conventions were the cue): [7192]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7067 | seal | Old Babylonian | incantations | Unknown | 42 |
| 7094 | seal | Old Babylonian | incantations | Unknown | 25 |
| 7142 | seal | Old Babylonian | incantations | Larsa area | 32 |
| 7165 | seal | Old Babylonian | incantations | Unknown | 22 |
| 7168 | seal | Old Babylonian | incantations | Mari | 40 |
| 7172 | seal | Old Babylonian | incantations | Larsa area | 37 |
| 7192 | seal | Old Babylonian | incantations | Unknown | 47 |

#### `incantations` predicted as `miscellaneous`

- tier0: 6x  |  maximal: 8x
- New in maximal (cleaning removes discriminating signal): [7155, 7205, 7264]
- Resolved by maximal cleaning (writing conventions were the cue): [7089]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7062 | seal | Old Babylonian | incantations | Unknown | 17 |
| 7089 | seal | Old Babylonian | incantations | Unknown | 4 |
| 7150 | seal | Old Babylonian | incantations | Isin | 23 |
| 7154 | seal | Old Babylonian | incantations | Adab | 33 |
| 7211 | seal | Old Babylonian | incantations | Larsa area | 10 |
| 7263 | seal | Middle Babylonian/Assyrian | incantations | Hattuša | 2 |

#### `hymns and prayers` predicted as `incantations`

- tier0: 5x  |  maximal: 2x
- New in maximal (cleaning removes discriminating signal): [7528, 26699]
- Resolved by maximal cleaning (writing conventions were the cue): [7488, 7504, 7529, 7562, 26698]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7488 | seal | Old Babylonian | hymns and prayers | Uruk | 11 |
| 7504 | seal | Old Babylonian | hymns and prayers | Ur | 32 |
| 7529 | seal | Old Babylonian | hymns and prayers | Girsu | 15 |
| 7562 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 7 |
| 26698 | seal | Old Babylonian | hymns and prayers | Unknown | 22 |

#### `love literature` predicted as `epics and myths`

- tier0: 5x  |  maximal: 2x
- New in maximal (cleaning removes discriminating signal): [1626]
- Resolved by maximal cleaning (writing conventions were the cue): [1617, 1631, 1635, 1638]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1617 | seal | Old Babylonian | love literature | Babylon | 153 |
| 1631 | seal | Middle Babylonian/Assyrian | love literature | Unknown | 141 |
| 1633 | seal | Middle Babylonian/Assyrian | love literature | Assur | 106 |
| 1635 | seal | Middle Babylonian/Assyrian | love literature | Nippur | 144 |
| 1638 | seal | Middle Babylonian/Assyrian | love literature | Nippur | 225 |

#### `epics` predicted as `rituals`

- tier0: 4x  |  maximal: 3x
- Resolved by maximal cleaning (writing conventions were the cue): [35576]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 35447 | lbpl | Late Babylonian | epics | Babylon | 100 |
| 35514 | lbpl | Late Babylonian | epics | Babylon | 93 |
| 35576 | lbpl | Late Babylonian | epics | Babylon | 669 |
| 35952 | lbpl | Late Babylonian | epics | Babylon | 323 |

#### `incantations` predicted as `epics and myths`

- tier0: 4x  |  maximal: 15x
- New in maximal (cleaning removes discriminating signal): [7046, 7051, 7054, 7057, 7069, 7109, 7122, 7146, 7180, 7200, 7203, 7217, 7233, 7265]
- Resolved by maximal cleaning (writing conventions were the cue): [7074, 7174, 7280]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7061 | seal | Old Babylonian | incantations | Larsa area | 64 |
| 7074 | seal | Old Babylonian | incantations | Larsa | 138 |
| 7174 | seal | Old Babylonian | incantations | Unknown | 61 |
| 7280 | seal | Middle Babylonian/Assyrian | incantations | Babylon;Sippar | 213 |

### New confusions under maximal cleaning only

These pairs are clean under tier0 but emerge after aggressive cleaning,
suggesting writing conventions were the only thing separating them.

- `incantations` → `catalogues`: 8x  IDs: [7052, 7071, 7084, 7126, 7237, 7254, 26519, 30351]
- `incantations` → `literary letters`: 7x  IDs: [7120, 7194, 7202, 7207, 7236, 7296, 7451]
- `love literature` → `wisdom literature`: 5x  IDs: [1619, 1620, 1622, 1630, 1631]
- `incantations` → `chronicles`: 3x  IDs: [7208, 7244, 7262]
- `incantations` → `lyrics`: 3x  IDs: [7248, 26862, 27670]

---

## Task: `sub_genre`

Classes: 43 | N: 246 | k: 2 | Acc (tier0/maximal): 0.333/0.305 | Macro-F1 (tier0/maximal): 0.286/0.267

### Confusion matrix (tier0)

| True/Pred | `anger` | `anzu` | `atraḫasis` | `baby crying` | `birth` | `dialogues` | `diseases (various) / evil (general)` | `diseases/demons (various)` | `dogs` | `etana` | `evil eye` | `fever` | `flies and wasps` | `foodstuff and drink` | `gastrointestinal problems` | `gilgameš` | `gula` | `hammurāpi` | `ištar` | `jaundice` | `lamaštu` | `legitimation` | `love` | `marduk` | `maškadu` | `miscellaneous` | `miscellaneous mb/ma epics` | `miscellaneous mb/ma incs.` | `miscellaneous ob epics` | `miscellaneous ob hymns` | `miscellaneous ob love lit.` | `narām-sin` | `nineanna` | `personal complaints` | `proverbs` | `riddles` | `samsuiluna` | `sargon` | `scorpions` | `snakes and reptiles` | `to named gods` | `toothache` | `worms and leeches` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `anger` | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `anzu` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `atraḫasis` | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `baby crying` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `birth` | 1 | 0 | 1 | 0 | 3 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `dialogues` | 0 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `diseases (various) / evil (general)` | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `diseases/demons (various)` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `dogs` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `etana` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `evil eye` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `fever` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `flies and wasps` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `foodstuff and drink` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gastrointestinal problems` | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| `gilgameš` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gula` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hammurāpi` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ištar` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `jaundice` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `lamaštu` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `legitimation` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `love` | 2 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| `marduk` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `maškadu` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous` | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous mb/ma epics` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous mb/ma incs.` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous ob epics` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous ob hymns` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous ob love lit.` | 0 | 0 | 0 | 0 | 0 | 4 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `narām-sin` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `nineanna` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `personal complaints` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `proverbs` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| `riddles` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `samsuiluna` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `sargon` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `scorpions` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 8 | 1 | 0 | 0 | 0 |
| `snakes and reptiles` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `to named gods` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `toothache` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `worms and leeches` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

### Confusion matrix (maximal)

| True/Pred | `anger` | `anzu` | `atraḫasis` | `baby crying` | `birth` | `dialogues` | `diseases (various) / evil (general)` | `diseases/demons (various)` | `dogs` | `etana` | `evil eye` | `fever` | `flies and wasps` | `foodstuff and drink` | `gastrointestinal problems` | `gilgameš` | `gula` | `hammurāpi` | `ištar` | `jaundice` | `lamaštu` | `legitimation` | `love` | `marduk` | `maškadu` | `miscellaneous` | `miscellaneous mb/ma epics` | `miscellaneous mb/ma incs.` | `miscellaneous ob epics` | `miscellaneous ob hymns` | `miscellaneous ob love lit.` | `narām-sin` | `nineanna` | `personal complaints` | `proverbs` | `riddles` | `samsuiluna` | `sargon` | `scorpions` | `snakes and reptiles` | `to named gods` | `toothache` | `worms and leeches` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `anger` | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `anzu` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `atraḫasis` | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `baby crying` | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `birth` | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `dialogues` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `diseases (various) / evil (general)` | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `diseases/demons (various)` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 1 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `dogs` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `etana` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `evil eye` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `fever` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `flies and wasps` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `foodstuff and drink` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gastrointestinal problems` | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| `gilgameš` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gula` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hammurāpi` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ištar` | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `jaundice` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `lamaštu` | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 |
| `legitimation` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `love` | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `marduk` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `maškadu` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous` | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous mb/ma epics` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous mb/ma incs.` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous ob epics` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous ob hymns` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `miscellaneous ob love lit.` | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `narām-sin` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `nineanna` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `personal complaints` | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `proverbs` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| `riddles` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `samsuiluna` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `sargon` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `scorpions` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 8 | 1 | 0 | 1 | 0 |
| `snakes and reptiles` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `to named gods` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `toothache` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `worms and leeches` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

### Top confusions with fragment IDs

#### `miscellaneous ob love lit.` predicted as `dialogues`

- tier0: 4x  |  maximal: 1x
- Resolved by maximal cleaning (writing conventions were the cue): [1619, 1621, 1622]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1619 | seal | Old Babylonian | love literature | Unknown | 111 |
| 1620 | seal | Old Babylonian | love literature | Unknown | 143 |
| 1621 | seal | Old Babylonian | love literature | Kiš | 89 |
| 1622 | seal | Old Babylonian | love literature | Sippar;Nippur | 246 |

#### `gilgameš` predicted as `anzu`

- tier0: 3x  |  maximal: 0x
- Resolved by maximal cleaning (writing conventions were the cue): [1532, 1534, 1830]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1532 | seal | Old Babylonian | epics and myths | Unknown | 323 |
| 1534 | seal | Old Babylonian | epics and myths | Unknown | 353 |
| 1830 | seal | Old Babylonian | epics and myths | Larsa | 577 |

#### `love` predicted as `baby crying`

- tier0: 3x  |  maximal: 2x
- Resolved by maximal cleaning (writing conventions were the cue): [7150]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7147 | seal | Old Babylonian | incantations | Isin | 35 |
| 7148 | seal | Old Babylonian | incantations | Isin | 21 |
| 7150 | seal | Old Babylonian | incantations | Isin | 23 |

#### `atraḫasis` predicted as `dialogues`

- tier0: 2x  |  maximal: 0x
- Resolved by maximal cleaning (writing conventions were the cue): [1517, 1520]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1517 | seal | Old Babylonian | epics and myths | Sippar | 731 |
| 1520 | seal | Middle Babylonian/Assyrian | epics and myths | Unknown | 199 |

#### `birth` predicted as `flies and wasps`

- tier0: 2x  |  maximal: 0x
- Resolved by maximal cleaning (writing conventions were the cue): [7055, 7056]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7055 | seal | Old Babylonian | incantations | Unknown | 31 |
| 7056 | seal | Old Babylonian | incantations | Unknown | 50 |

#### `dialogues` predicted as `anzu`

- tier0: 2x  |  maximal: 0x
- Resolved by maximal cleaning (writing conventions were the cue): [1638, 1751]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1638 | seal | Middle Babylonian/Assyrian | love literature | Nippur | 225 |
| 1751 | seal | Old Babylonian | wisdom literature | Larsa | 395 |

#### `dialogues` predicted as `etana`

- tier0: 2x  |  maximal: 0x
- Resolved by maximal cleaning (writing conventions were the cue): [1615, 1738]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1615 | seal | Old Babylonian | love literature | Sippar | 307 |
| 1738 | seal | Old Babylonian | wisdom literature | Unknown;Larsa | 1045 |

#### `dogs` predicted as `miscellaneous`

- tier0: 2x  |  maximal: 2x

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7089 | seal | Old Babylonian | incantations | Unknown | 4 |
| 7094 | seal | Old Babylonian | incantations | Unknown | 25 |

#### `gastrointestinal problems` predicted as `proverbs`

- tier0: 2x  |  maximal: 0x
- Resolved by maximal cleaning (writing conventions were the cue): [7111, 7122]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7111 | seal | Old Babylonian | incantations | Unknown | 16 |
| 7122 | seal | Old Babylonian | incantations | Unknown | 12 |

#### `ištar` predicted as `dialogues`

- tier0: 2x  |  maximal: 1x
- Resolved by maximal cleaning (writing conventions were the cue): [7494]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1631 | seal | Middle Babylonian/Assyrian | love literature | Unknown | 141 |
| 7494 | seal | Old Babylonian | hymns and prayers | Unknown | 240 |

### New confusions under maximal cleaning only

These pairs are clean under tier0 but emerge after aggressive cleaning,
suggesting writing conventions were the only thing separating them.

- `miscellaneous ob love lit.` → `proverbs`: 3x  IDs: [1619, 1622, 1630]
- `birth` → `gilgameš`: 2x  IDs: [7063, 7229]
- `dialogues` → `diseases/demons (various)`: 2x  IDs: [1638, 1738]
- `dialogues` → `miscellaneous`: 2x  IDs: [1637, 1736]
- `dialogues` → `proverbs`: 2x  IDs: [1615, 1751]

---

## Task: `provenance`

Classes: 25 | N: 374 | k: 2 | Acc (tier0/maximal): 0.259/0.219 | Macro-F1 (tier0/maximal): 0.171/0.127

### Confusion matrix (tier0)

| True/Pred | `Adab` | `Akhetaten` | `Assur` | `Babylon` | `Dūr-Kurigalzu` | `Emar` | `Girsu` | `Hattuša` | `Isin` | `Kaniš` | `Kiš` | `Larsa` | `Larsa area` | `Mari` | `Nerebtum` | `Nineveh` | `Nippur` | `Sippar` | `Susa` | `Tell Duweihes` | `Ugarit` | `Unknown` | `Ur` | `Uruk` | `Šaduppûm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Adab` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Akhetaten` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Assur` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Babylon` | 0 | 0 | 0 | 44 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Dūr-Kurigalzu` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Emar` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Girsu` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `Hattuša` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 3 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| `Isin` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `Kaniš` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Kiš` | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 4 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Larsa` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 1 | 2 | 0 | 0 |
| `Larsa area` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 | 0 | 3 | 2 | 9 | 4 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 4 | 1 | 0 | 0 |
| `Mari` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1 |
| `Nerebtum` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| `Nineveh` | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Nippur` | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 2 | 1 | 1 | 0 | 3 | 1 | 0 | 0 | 0 | 1 | 2 | 0 | 0 |
| `Sippar` | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 2 | 0 | 0 | 4 | 2 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Susa` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Tell Duweihes` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Ugarit` | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 0 |
| `Unknown` | 1 | 0 | 1 | 7 | 0 | 0 | 0 | 16 | 3 | 1 | 3 | 15 | 24 | 6 | 2 | 2 | 2 | 3 | 7 | 0 | 3 | 9 | 3 | 0 | 4 |
| `Ur` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 2 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 2 | 0 | 0 | 0 |
| `Uruk` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `Šaduppûm` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

### Confusion matrix (maximal)

| True/Pred | `Adab` | `Akhetaten` | `Assur` | `Babylon` | `Dūr-Kurigalzu` | `Emar` | `Girsu` | `Hattuša` | `Isin` | `Kaniš` | `Kiš` | `Larsa` | `Larsa area` | `Mari` | `Nerebtum` | `Nineveh` | `Nippur` | `Sippar` | `Susa` | `Tell Duweihes` | `Ugarit` | `Unknown` | `Ur` | `Uruk` | `Šaduppûm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Adab` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `Akhetaten` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Assur` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `Babylon` | 1 | 0 | 0 | 42 | 0 | 3 | 0 | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| `Dūr-Kurigalzu` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Emar` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `Girsu` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `Hattuša` | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| `Isin` | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Kaniš` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Kiš` | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 2 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Larsa` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| `Larsa area` | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 4 | 1 | 2 | 3 | 3 | 4 | 2 | 1 | 0 | 0 | 0 | 2 | 2 | 1 | 0 | 1 | 1 |
| `Mari` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| `Nerebtum` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| `Nineveh` | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| `Nippur` | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 2 | 0 | 0 | 1 | 1 | 1 | 0 | 4 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| `Sippar` | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 4 | 0 | 2 | 3 | 1 | 0 | 1 | 0 | 0 | 0 |
| `Susa` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 2 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 |
| `Tell Duweihes` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `Ugarit` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 4 | 0 | 0 | 0 | 0 |
| `Unknown` | 3 | 4 | 1 | 7 | 5 | 1 | 1 | 10 | 8 | 5 | 4 | 11 | 8 | 4 | 5 | 1 | 1 | 2 | 5 | 4 | 6 | 7 | 3 | 1 | 5 |
| `Ur` | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| `Uruk` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Šaduppûm` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |

### Top confusions with fragment IDs

#### `Unknown` predicted as `Larsa area`

- tier0: 24x  |  maximal: 8x
- New in maximal (cleaning removes discriminating signal): [7056]
- Resolved by maximal cleaning (writing conventions were the cue): [1546, 1664, 7052, 7066, 7070, 7073, 7101, 7103, 7119, 7122, 7123, 7130, 7153, 7201, 7604, 13484, 27670]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1546 | seal | Old Babylonian | epics and myths | Unknown | 9 |
| 1664 | seal | Old Babylonian | miscellaneous | Unknown | 9 |
| 7052 | seal | Old Babylonian | incantations | Unknown | 23 |
| 7054 | seal | Old Babylonian | incantations | Unknown | 40 |
| 7066 | seal | Old Babylonian | incantations | Unknown | 10 |
| 7067 | seal | Old Babylonian | incantations | Unknown | 42 |
| 7070 | seal | Old Babylonian | incantations | Unknown | 38 |
| 7073 | seal | Old Babylonian | incantations | Unknown | 22 |
| 7101 | seal | Old Babylonian | incantations | Unknown | 6 |
| 7103 | seal | Old Babylonian | incantations | Unknown | 29 |
| 7107 | seal | Old Babylonian | incantations | Unknown | 12 |
| 7111 | seal | Old Babylonian | incantations | Unknown | 16 |
| 7119 | seal | Old Babylonian | incantations | Unknown | 9 |
| 7122 | seal | Old Babylonian | incantations | Unknown | 12 |
| 7123 | seal | Old Babylonian | incantations | Unknown | 10 |
| 7130 | seal | Old Babylonian | incantations | Unknown | 14 |
| 7153 | seal | Old Babylonian | incantations | Unknown | 29 |
| 7164 | seal | Old Babylonian | incantations | Unknown | 18 |
| 7201 | seal | Old Babylonian | incantations | Unknown | 3 |
| 7604 | seal | Old Babylonian | incantations | Unknown | 25 |
| 13429 | seal | Old Babylonian | incantations | Unknown | 62 |
| 13484 | seal | Old Babylonian | incantations | Unknown | 10 |
| 27670 | seal | Old Babylonian | incantations | Unknown | 8 |
| 30351 | seal | Old Babylonian | incantations | Unknown | 6 |

#### `Unknown` predicted as `Hattuša`

- tier0: 16x  |  maximal: 10x
- New in maximal (cleaning removes discriminating signal): [7066, 7120, 7174, 26600]
- Resolved by maximal cleaning (writing conventions were the cue): [1755, 7062, 7089, 7108, 7189, 7558, 7559, 7560, 7561, 13431]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1755 | seal | Old Babylonian | wisdom literature | Unknown | 8 |
| 7062 | seal | Old Babylonian | incantations | Unknown | 17 |
| 7089 | seal | Old Babylonian | incantations | Unknown | 4 |
| 7108 | seal | Old Babylonian | incantations | Unknown | 35 |
| 7121 | seal | Old Babylonian | incantations | Unknown | 34 |
| 7133 | seal | Old Babylonian | incantations | Unknown | 6 |
| 7177 | seal | Old Babylonian | incantations | Unknown | 8 |
| 7189 | seal | Old Babylonian | incantations | Unknown | 15 |
| 7192 | seal | Old Babylonian | incantations | Unknown | 47 |
| 7194 | seal | Old Babylonian | incantations | Unknown | 4 |
| 7244 | seal | Middle Babylonian/Assyrian | incantations | Unknown | 16 |
| 7558 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 12 |
| 7559 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 7 |
| 7560 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 9 |
| 7561 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 9 |
| 13431 | seal | Old Babylonian | incantations | Unknown | 5 |

#### `Unknown` predicted as `Larsa`

- tier0: 15x  |  maximal: 11x
- New in maximal (cleaning removes discriminating signal): [1644, 1649, 1670, 1754, 1808, 7502, 26698, 26699]
- Resolved by maximal cleaning (writing conventions were the cue): [1531, 1539, 1613, 1614, 1619, 7053, 7056, 7057, 7063, 7174, 7494, 26600]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1531 | seal | Old Babylonian | epics and myths | Unknown | 49 |
| 1532 | seal | Old Babylonian | epics and myths | Unknown | 323 |
| 1539 | seal | Old Babylonian | epics and myths | Unknown | 246 |
| 1613 | seal | Old Babylonian | catalogues | Unknown | 143 |
| 1614 | seal | Old Babylonian | love literature | Unknown | 55 |
| 1619 | seal | Old Babylonian | love literature | Unknown | 111 |
| 1812 | seal | Old Babylonian | lamentations | Unknown | 28 |
| 7053 | seal | Old Babylonian | incantations | Unknown | 32 |
| 7056 | seal | Old Babylonian | incantations | Unknown | 50 |
| 7057 | seal | Old Babylonian | incantations | Unknown | 23 |
| 7063 | seal | Old Babylonian | incantations | Unknown | 39 |
| 7071 | seal | Old Babylonian | incantations | Unknown | 111 |
| 7174 | seal | Old Babylonian | incantations | Unknown | 61 |
| 7494 | seal | Old Babylonian | hymns and prayers | Unknown | 240 |
| 26600 | seal | Old Babylonian | hymns and prayers | Unknown | 293 |

#### `Babylon` predicted as `Nineveh`

- tier0: 11x  |  maximal: 5x
- New in maximal (cleaning removes discriminating signal): [33292]
- Resolved by maximal cleaning (writing conventions were the cue): [32857, 32919, 33065, 33124, 33208, 33353, 33385]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 32264 | dll | Neo-Assyrian and Late Babylonian | rituals | Babylon | 785 |
| 32857 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 117 |
| 32919 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 152 |
| 32978 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 30 |
| 33065 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 101 |
| 33100 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 52 |
| 33124 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 226 |
| 33208 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 175 |
| 33353 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 52 |
| 33385 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 85 |
| 34229 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 87 |

#### `Nineveh` predicted as `Babylon`

- tier0: 8x  |  maximal: 6x
- New in maximal (cleaning removes discriminating signal): [31697, 33549, 33621]
- Resolved by maximal cleaning (writing conventions were the cue): [27128, 31713, 33520, 33746, 33837]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 27128 | seal | Later Periods (SB, NA, LB) | epics and myths | Nineveh | 137 |
| 31713 | dll | Neo-Assyrian and Late Babylonian | commentary | Nineveh | 15 |
| 33520 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 24 |
| 33745 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 82 |
| 33746 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 146 |
| 33837 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 187 |
| 33936 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 230 |
| 34139 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 60 |

#### `Unknown` predicted as `Babylon`

- tier0: 7x  |  maximal: 7x
- New in maximal (cleaning removes discriminating signal): [7123, 7558, 7559, 7560, 7561]
- Resolved by maximal cleaning (writing conventions were the cue): [1598, 7082, 7547, 7551, 26699]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1598 | seal | Middle Babylonian/Assyrian | epics and myths | Unknown | 127 |
| 7082 | seal | Old Babylonian | incantations | Unknown | 42 |
| 7545 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 4 |
| 7547 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 21 |
| 7551 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 15 |
| 26699 | seal | Old Babylonian | hymns and prayers | Unknown | 20 |
| 38285 | lbpl | Late Babylonian | rituals | Unknown | 41 |

#### `Unknown` predicted as `Susa`

- tier0: 7x  |  maximal: 5x
- New in maximal (cleaning removes discriminating signal): [1539, 1766, 7200]
- Resolved by maximal cleaning (writing conventions were the cue): [6, 1536, 1537, 1644, 7502]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 6 | seal | Old Babylonian | epics and myths | Unknown | 136 |
| 1524 | seal | Old Babylonian | epics and myths | Unknown | 289 |
| 1534 | seal | Old Babylonian | epics and myths | Unknown | 353 |
| 1536 | seal | Old Babylonian | epics and myths | Unknown | 280 |
| 1537 | seal | Old Babylonian | epics and myths | Unknown | 159 |
| 1644 | seal | Middle Babylonian/Assyrian | literary letters | Unknown | 62 |
| 7502 | seal | Old Babylonian | hymns and prayers | Unknown | 148 |

#### `Unknown` predicted as `Mari`

- tier0: 6x  |  maximal: 4x
- New in maximal (cleaning removes discriminating signal): [1537, 7528, 7604, 27670]
- Resolved by maximal cleaning (writing conventions were the cue): [1542, 1548, 1620, 1670, 1754, 1808]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1542 | seal | Old Babylonian | epics and myths | Unknown | 130 |
| 1548 | seal | Old Babylonian | epics and myths | Unknown | 285 |
| 1620 | seal | Old Babylonian | love literature | Unknown | 143 |
| 1670 | seal | Old Babylonian | lamentations | Unknown | 113 |
| 1754 | seal | Old Babylonian | wisdom literature | Unknown | 120 |
| 1808 | seal | Old Babylonian | lamentations | Unknown | 270 |

#### `Larsa area` predicted as `Mari`

- tier0: 4x  |  maximal: 4x
- New in maximal (cleaning removes discriminating signal): [7172, 7210]
- Resolved by maximal cleaning (writing conventions were the cue): [7109, 7193]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7109 | seal | Old Babylonian | incantations | Larsa area | 29 |
| 7160 | seal | Old Babylonian | incantations | Larsa area | 39 |
| 7193 | seal | Old Babylonian | incantations | Larsa area | 59 |
| 27569 | seal | Old Babylonian | incantations | Larsa area | 28 |

#### `Larsa area` predicted as `Unknown`

- tier0: 4x  |  maximal: 1x
- New in maximal (cleaning removes discriminating signal): [7205]
- Resolved by maximal cleaning (writing conventions were the cue): [7126, 7185, 7209, 7210]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7126 | seal | Old Babylonian | incantations | Larsa area | 36 |
| 7185 | seal | Old Babylonian | incantations | Larsa area | 26 |
| 7209 | seal | Old Babylonian | incantations | Larsa area | 26 |
| 7210 | seal | Old Babylonian | incantations | Larsa area | 21 |

### New confusions under maximal cleaning only

These pairs are clean under tier0 but emerge after aggressive cleaning,
suggesting writing conventions were the only thing separating them.

- `Unknown` → `Dūr-Kurigalzu`: 5x  IDs: [1536, 7057, 7122, 7547, 13484]
- `Unknown` → `Akhetaten`: 4x  IDs: [1631, 1765, 7070, 7165]
- `Unknown` → `Tell Duweihes`: 4x  IDs: [1664, 7053, 7103, 7110]
- `Babylon` → `Emar`: 3x  IDs: [33065, 37180, 39031]
- `Hattuša` → `Sippar`: 2x  IDs: [7295, 7450]

---

## Task: `sub_provenance`

Classes: 25 | N: 374 | k: 2 | Acc (tier0/maximal): 0.259/0.219 | Macro-F1 (tier0/maximal): 0.171/0.127

### Confusion matrix (tier0)

| True/Pred | `Larsa area` | `Unknown` | `mod. Boghazköy` | `mod. Ishan Baḥriyat` | `mod. Ishchali` | `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | `mod. Kouyunjik, Tell Nabi Yunus` | `mod. Kültepe` | `mod. Nuffar` | `mod. Qalʿat Sharqat` | `mod. Ras Shamrah` | `mod. Shush` | `mod. Tell Abu Ḥabbah` | `mod. Tell Bismaya` | `mod. Tell Meskene` | `mod. Tell el-Amarna` | `mod. Tell el-Muqayyar` | `mod. Tell el-Uhaymir` | `mod. Tell es-Senkereh` | `mod. Tell Ḥariri` | `mod. Tell Ḥarmal` | `mod. Telloh` | `mod. Warka` | `mod. ʿAqar Quf` | `vicinity of Nippur` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Larsa area` | 9 | 4 | 2 | 3 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 3 | 2 | 4 | 0 | 0 | 0 | 0 | 0 |
| `Unknown` | 24 | 9 | 16 | 3 | 2 | 7 | 2 | 1 | 2 | 1 | 3 | 7 | 3 | 1 | 0 | 0 | 3 | 3 | 15 | 6 | 4 | 0 | 0 | 0 | 0 |
| `mod. Boghazköy` | 3 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ishan Baḥriyat` | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ishchali` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | 0 | 0 | 3 | 0 | 0 | 44 | 11 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Kouyunjik, Tell Nabi Yunus` | 0 | 0 | 0 | 0 | 0 | 8 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Kültepe` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Nuffar` | 2 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 1 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| `mod. Qalʿat Sharqat` | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ras Shamrah` | 0 | 0 | 2 | 0 | 0 | 4 | 3 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Shush` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Abu Ḥabbah` | 2 | 1 | 0 | 1 | 0 | 3 | 4 | 0 | 2 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Bismaya` | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Meskene` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Amarna` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Muqayyar` | 2 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Uhaymir` | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell es-Senkereh` | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 2 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Ḥariri` | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| `mod. Tell Ḥarmal` | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 2 | 0 | 0 | 0 | 0 |
| `mod. Telloh` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `mod. Warka` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. ʿAqar Quf` | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `vicinity of Nippur` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

### Confusion matrix (maximal)

| True/Pred | `Larsa area` | `Unknown` | `mod. Boghazköy` | `mod. Ishan Baḥriyat` | `mod. Ishchali` | `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | `mod. Kouyunjik, Tell Nabi Yunus` | `mod. Kültepe` | `mod. Nuffar` | `mod. Qalʿat Sharqat` | `mod. Ras Shamrah` | `mod. Shush` | `mod. Tell Abu Ḥabbah` | `mod. Tell Bismaya` | `mod. Tell Meskene` | `mod. Tell el-Amarna` | `mod. Tell el-Muqayyar` | `mod. Tell el-Uhaymir` | `mod. Tell es-Senkereh` | `mod. Tell Ḥariri` | `mod. Tell Ḥarmal` | `mod. Telloh` | `mod. Warka` | `mod. ʿAqar Quf` | `vicinity of Nippur` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Larsa area` | 3 | 1 | 3 | 4 | 2 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 3 | 4 | 1 | 0 | 1 | 1 | 2 |
| `Unknown` | 8 | 7 | 10 | 8 | 5 | 7 | 1 | 5 | 1 | 1 | 6 | 5 | 2 | 3 | 1 | 4 | 3 | 4 | 11 | 4 | 5 | 1 | 1 | 5 | 4 |
| `mod. Boghazköy` | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| `mod. Ishan Baḥriyat` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ishchali` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` | 0 | 0 | 3 | 0 | 0 | 42 | 5 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 3 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
| `mod. Kouyunjik, Tell Nabi Yunus` | 0 | 0 | 0 | 1 | 0 | 6 | 8 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `mod. Kültepe` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `mod. Nuffar` | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 2 | 4 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| `mod. Qalʿat Sharqat` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Ras Shamrah` | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 4 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Shush` | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell Abu Ḥabbah` | 0 | 1 | 0 | 1 | 0 | 4 | 4 | 0 | 0 | 1 | 0 | 3 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `mod. Tell Bismaya` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `mod. Tell Meskene` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Amarna` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell el-Muqayyar` | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 |
| `mod. Tell el-Uhaymir` | 1 | 1 | 2 | 1 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. Tell es-Senkereh` | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| `mod. Tell Ḥariri` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 1 |
| `mod. Tell Ḥarmal` | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 |
| `mod. Telloh` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `mod. Warka` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mod. ʿAqar Quf` | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `vicinity of Nippur` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |

### Top confusions with fragment IDs

#### `Unknown` predicted as `Larsa area`

- tier0: 24x  |  maximal: 8x
- New in maximal (cleaning removes discriminating signal): [7056]
- Resolved by maximal cleaning (writing conventions were the cue): [1546, 1664, 7052, 7066, 7070, 7073, 7101, 7103, 7119, 7122, 7123, 7130, 7153, 7201, 7604, 13484, 27670]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1546 | seal | Old Babylonian | epics and myths | Unknown | 9 |
| 1664 | seal | Old Babylonian | miscellaneous | Unknown | 9 |
| 7052 | seal | Old Babylonian | incantations | Unknown | 23 |
| 7054 | seal | Old Babylonian | incantations | Unknown | 40 |
| 7066 | seal | Old Babylonian | incantations | Unknown | 10 |
| 7067 | seal | Old Babylonian | incantations | Unknown | 42 |
| 7070 | seal | Old Babylonian | incantations | Unknown | 38 |
| 7073 | seal | Old Babylonian | incantations | Unknown | 22 |
| 7101 | seal | Old Babylonian | incantations | Unknown | 6 |
| 7103 | seal | Old Babylonian | incantations | Unknown | 29 |
| 7107 | seal | Old Babylonian | incantations | Unknown | 12 |
| 7111 | seal | Old Babylonian | incantations | Unknown | 16 |
| 7119 | seal | Old Babylonian | incantations | Unknown | 9 |
| 7122 | seal | Old Babylonian | incantations | Unknown | 12 |
| 7123 | seal | Old Babylonian | incantations | Unknown | 10 |
| 7130 | seal | Old Babylonian | incantations | Unknown | 14 |
| 7153 | seal | Old Babylonian | incantations | Unknown | 29 |
| 7164 | seal | Old Babylonian | incantations | Unknown | 18 |
| 7201 | seal | Old Babylonian | incantations | Unknown | 3 |
| 7604 | seal | Old Babylonian | incantations | Unknown | 25 |
| 13429 | seal | Old Babylonian | incantations | Unknown | 62 |
| 13484 | seal | Old Babylonian | incantations | Unknown | 10 |
| 27670 | seal | Old Babylonian | incantations | Unknown | 8 |
| 30351 | seal | Old Babylonian | incantations | Unknown | 6 |

#### `Unknown` predicted as `mod. Boghazköy`

- tier0: 16x  |  maximal: 10x
- New in maximal (cleaning removes discriminating signal): [7066, 7120, 7174, 26600]
- Resolved by maximal cleaning (writing conventions were the cue): [1755, 7062, 7089, 7108, 7189, 7558, 7559, 7560, 7561, 13431]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1755 | seal | Old Babylonian | wisdom literature | Unknown | 8 |
| 7062 | seal | Old Babylonian | incantations | Unknown | 17 |
| 7089 | seal | Old Babylonian | incantations | Unknown | 4 |
| 7108 | seal | Old Babylonian | incantations | Unknown | 35 |
| 7121 | seal | Old Babylonian | incantations | Unknown | 34 |
| 7133 | seal | Old Babylonian | incantations | Unknown | 6 |
| 7177 | seal | Old Babylonian | incantations | Unknown | 8 |
| 7189 | seal | Old Babylonian | incantations | Unknown | 15 |
| 7192 | seal | Old Babylonian | incantations | Unknown | 47 |
| 7194 | seal | Old Babylonian | incantations | Unknown | 4 |
| 7244 | seal | Middle Babylonian/Assyrian | incantations | Unknown | 16 |
| 7558 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 12 |
| 7559 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 7 |
| 7560 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 9 |
| 7561 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 9 |
| 13431 | seal | Old Babylonian | incantations | Unknown | 5 |

#### `Unknown` predicted as `mod. Tell es-Senkereh`

- tier0: 15x  |  maximal: 11x
- New in maximal (cleaning removes discriminating signal): [1644, 1649, 1670, 1754, 1808, 7502, 26698, 26699]
- Resolved by maximal cleaning (writing conventions were the cue): [1531, 1539, 1613, 1614, 1619, 7053, 7056, 7057, 7063, 7174, 7494, 26600]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1531 | seal | Old Babylonian | epics and myths | Unknown | 49 |
| 1532 | seal | Old Babylonian | epics and myths | Unknown | 323 |
| 1539 | seal | Old Babylonian | epics and myths | Unknown | 246 |
| 1613 | seal | Old Babylonian | catalogues | Unknown | 143 |
| 1614 | seal | Old Babylonian | love literature | Unknown | 55 |
| 1619 | seal | Old Babylonian | love literature | Unknown | 111 |
| 1812 | seal | Old Babylonian | lamentations | Unknown | 28 |
| 7053 | seal | Old Babylonian | incantations | Unknown | 32 |
| 7056 | seal | Old Babylonian | incantations | Unknown | 50 |
| 7057 | seal | Old Babylonian | incantations | Unknown | 23 |
| 7063 | seal | Old Babylonian | incantations | Unknown | 39 |
| 7071 | seal | Old Babylonian | incantations | Unknown | 111 |
| 7174 | seal | Old Babylonian | incantations | Unknown | 61 |
| 7494 | seal | Old Babylonian | hymns and prayers | Unknown | 240 |
| 26600 | seal | Old Babylonian | hymns and prayers | Unknown | 293 |

#### `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` predicted as `mod. Kouyunjik, Tell Nabi Yunus`

- tier0: 11x  |  maximal: 5x
- New in maximal (cleaning removes discriminating signal): [33292]
- Resolved by maximal cleaning (writing conventions were the cue): [32857, 32919, 33065, 33124, 33208, 33353, 33385]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 32264 | dll | Neo-Assyrian and Late Babylonian | rituals | Babylon | 785 |
| 32857 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 117 |
| 32919 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 152 |
| 32978 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 30 |
| 33065 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 101 |
| 33100 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 52 |
| 33124 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 226 |
| 33208 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 175 |
| 33353 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 52 |
| 33385 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 85 |
| 34229 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 87 |

#### `mod. Kouyunjik, Tell Nabi Yunus` predicted as `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes`

- tier0: 8x  |  maximal: 6x
- New in maximal (cleaning removes discriminating signal): [31697, 33549, 33621]
- Resolved by maximal cleaning (writing conventions were the cue): [27128, 31713, 33520, 33746, 33837]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 27128 | seal | Later Periods (SB, NA, LB) | epics and myths | Nineveh | 137 |
| 31713 | dll | Neo-Assyrian and Late Babylonian | commentary | Nineveh | 15 |
| 33520 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 24 |
| 33745 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 82 |
| 33746 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 146 |
| 33837 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 187 |
| 33936 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 230 |
| 34139 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 60 |

#### `Unknown` predicted as `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes`

- tier0: 7x  |  maximal: 7x
- New in maximal (cleaning removes discriminating signal): [7123, 7558, 7559, 7560, 7561]
- Resolved by maximal cleaning (writing conventions were the cue): [1598, 7082, 7547, 7551, 26699]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1598 | seal | Middle Babylonian/Assyrian | epics and myths | Unknown | 127 |
| 7082 | seal | Old Babylonian | incantations | Unknown | 42 |
| 7545 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 4 |
| 7547 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 21 |
| 7551 | seal | Middle Babylonian/Assyrian | hymns and prayers | Unknown | 15 |
| 26699 | seal | Old Babylonian | hymns and prayers | Unknown | 20 |
| 38285 | lbpl | Late Babylonian | rituals | Unknown | 41 |

#### `Unknown` predicted as `mod. Shush`

- tier0: 7x  |  maximal: 5x
- New in maximal (cleaning removes discriminating signal): [1539, 1766, 7200]
- Resolved by maximal cleaning (writing conventions were the cue): [6, 1536, 1537, 1644, 7502]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 6 | seal | Old Babylonian | epics and myths | Unknown | 136 |
| 1524 | seal | Old Babylonian | epics and myths | Unknown | 289 |
| 1534 | seal | Old Babylonian | epics and myths | Unknown | 353 |
| 1536 | seal | Old Babylonian | epics and myths | Unknown | 280 |
| 1537 | seal | Old Babylonian | epics and myths | Unknown | 159 |
| 1644 | seal | Middle Babylonian/Assyrian | literary letters | Unknown | 62 |
| 7502 | seal | Old Babylonian | hymns and prayers | Unknown | 148 |

#### `Unknown` predicted as `mod. Tell Ḥariri`

- tier0: 6x  |  maximal: 4x
- New in maximal (cleaning removes discriminating signal): [1537, 7528, 7604, 27670]
- Resolved by maximal cleaning (writing conventions were the cue): [1542, 1548, 1620, 1670, 1754, 1808]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1542 | seal | Old Babylonian | epics and myths | Unknown | 130 |
| 1548 | seal | Old Babylonian | epics and myths | Unknown | 285 |
| 1620 | seal | Old Babylonian | love literature | Unknown | 143 |
| 1670 | seal | Old Babylonian | lamentations | Unknown | 113 |
| 1754 | seal | Old Babylonian | wisdom literature | Unknown | 120 |
| 1808 | seal | Old Babylonian | lamentations | Unknown | 270 |

#### `Larsa area` predicted as `Unknown`

- tier0: 4x  |  maximal: 1x
- New in maximal (cleaning removes discriminating signal): [7205]
- Resolved by maximal cleaning (writing conventions were the cue): [7126, 7185, 7209, 7210]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7126 | seal | Old Babylonian | incantations | Larsa area | 36 |
| 7185 | seal | Old Babylonian | incantations | Larsa area | 26 |
| 7209 | seal | Old Babylonian | incantations | Larsa area | 26 |
| 7210 | seal | Old Babylonian | incantations | Larsa area | 21 |

#### `Larsa area` predicted as `mod. Tell Ḥariri`

- tier0: 4x  |  maximal: 4x
- New in maximal (cleaning removes discriminating signal): [7172, 7210]
- Resolved by maximal cleaning (writing conventions were the cue): [7109, 7193]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 7109 | seal | Old Babylonian | incantations | Larsa area | 29 |
| 7160 | seal | Old Babylonian | incantations | Larsa area | 39 |
| 7193 | seal | Old Babylonian | incantations | Larsa area | 59 |
| 27569 | seal | Old Babylonian | incantations | Larsa area | 28 |

### New confusions under maximal cleaning only

These pairs are clean under tier0 but emerge after aggressive cleaning,
suggesting writing conventions were the only thing separating them.

- `Unknown` → `mod. ʿAqar Quf`: 5x  IDs: [1536, 7057, 7122, 7547, 13484]
- `Unknown` → `mod. Tell el-Amarna`: 4x  IDs: [1631, 1765, 7070, 7165]
- `Unknown` → `vicinity of Nippur`: 4x  IDs: [1664, 7053, 7103, 7110]
- `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes` → `mod. Tell Meskene`: 3x  IDs: [33065, 37180, 39031]
- `Larsa area` → `mod. Ishchali`: 2x  IDs: [7109, 7209]

---

## Task: `domain`

Classes: 3 | N: 384 | k: 5 | Acc (tier0/maximal): 0.979/0.953 | Macro-F1 (tier0/maximal): 0.952/0.889

### Confusion matrix (tier0)

| True/Pred | `DLL` | `LBPL` | `SEAL` |
|---|---|---|---|
| `DLL` | 40 | 1 | 3 |
| `LBPL` | 2 | 35 | 1 |
| `SEAL` | 0 | 1 | 301 |

### Confusion matrix (maximal)

| True/Pred | `DLL` | `LBPL` | `SEAL` |
|---|---|---|---|
| `DLL` | 36 | 2 | 6 |
| `LBPL` | 5 | 31 | 2 |
| `SEAL` | 0 | 3 | 299 |

### Top confusions with fragment IDs

#### `DLL` predicted as `SEAL`

- tier0: 3x  |  maximal: 6x
- New in maximal (cleaning removes discriminating signal): [31697, 32774, 32813, 33562]
- Resolved by maximal cleaning (writing conventions were the cue): [31687]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 31687 | dll | Neo-Assyrian and Late Babylonian | commentary | Sippar | 18 |
| 31713 | dll | Neo-Assyrian and Late Babylonian | commentary | Nineveh | 15 |
| 33543 | dll | Neo-Assyrian and Late Babylonian | lyrics | Nineveh | 5 |

#### `LBPL` predicted as `DLL`

- tier0: 2x  |  maximal: 5x
- New in maximal (cleaning removes discriminating signal): [35151, 35344, 35514, 36280, 38658]
- Resolved by maximal cleaning (writing conventions were the cue): [35238, 38285]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 35238 | lbpl | Late Babylonian | epics | Babylon | 16 |
| 38285 | lbpl | Late Babylonian | rituals | Unknown | 41 |

#### `DLL` predicted as `LBPL`

- tier0: 1x  |  maximal: 2x
- New in maximal (cleaning removes discriminating signal): [33124]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 33322 | dll | Neo-Assyrian and Late Babylonian | lyrics | Babylon | 50 |

#### `LBPL` predicted as `SEAL`

- tier0: 1x  |  maximal: 2x
- New in maximal (cleaning removes discriminating signal): [35576, 36163]
- Resolved by maximal cleaning (writing conventions were the cue): [34808]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 34808 | lbpl | Late Babylonian | chronicles | Babylon | 47 |

#### `SEAL` predicted as `LBPL`

- tier0: 1x  |  maximal: 3x
- New in maximal (cleaning removes discriminating signal): [7235, 7257, 7262]
- Resolved by maximal cleaning (writing conventions were the cue): [1600]

| fragment_id | corpus | period | genre | provenance | word_count |
|---|---|---|---|---|---|
| 1600 | seal | Middle Babylonian/Assyrian | epics and myths | Assur;Nineveh | 294 |

---

## Summary: label issues to raise with Chungrong

| Task | CSV | Issue | Evidence | Status |
|------|-----|-------|----------|--------|
| period | seal.csv | `Middle Babylonian/Assyrian` is a compound label. Confused heavily with both OB and LB. | MB/MA->OB: 23x(t0)/37x(mx); MB/MA->LB: 11x(t0)/6x(mx) | **Partially resolved (2026-04-14)**: split into `Middle Babylonian` (24) + `Middle Assyrian` (6); 35 frags remain ambiguous |
| period | dll.csv | `Neo-Assyrian and Late Babylonian` is a compound label covering all 44 DLL fragments. | LB->NA+LB: 7x(mx); NA+LB->LB: 6x(mx) | **Partially resolved (2026-04-14)**: split into `Neo-Assyrian` (18) + `Neo or Late Babylonian` (26 still compound) |
| period | seal.csv | `Old Assyrian` has only N=5 fragments — 100% error rate under maximal. | OA->OB: 3x(t0)/5x(mx) | **Not addressed** in re-delivery |
| provenance / sub_provenance | all | 1:1 parallel columns, identical results at both cleanings. Are these distinct tasks? | F1m tier0: 0.171/0.171; maximal: 0.127/0.127 | **Not addressed** — awaiting clarification |

## Round 5 re-delivery (2026-04-14)

New CSVs at `yarin/emails_phase/round4/{seal,dll,lbpl}.csv` — not yet copied into
`seal_round4/`. Phases 0→A→B→C must be re-run on the corrected data.

**Key notes from Chunrong's reply:**
- Some tablets genuinely cannot be dated more precisely — `Neo or Late Babylonian` and
  remaining `Middle Babylonian/Assyrian` are honest "uncertain" labels, not oversights.
- Chunrong is still checking the period of some tablets — further corrections may follow.

**Key note from Nathan Wasserman's reply (2026-04-13):**
- Period labels (OB, LB, etc.) are 500-year buckets based on expert judgment, not hard science.
- The **real research goal** is fine-grained chronological ordering *within* a period (e.g.,
  ordering 229 OB texts among themselves), not just cross-period classification.
- Cross-period separation (OB vs LB) is something Nathan can already do by hand — the machine
  needs to do something harder and more precise.
- This reframes the thesis: the SEAL multi-task experiment is a stepping stone; the deeper
  question is within-OB ordering once Chunrong provides finer-grained period sub-labels.

