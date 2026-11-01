# SSL corpus census — 2026-09-02

rows before dedupe 43,283 · content duplicates removed 871 (cross-source duplicate groups: 46) · min length 8 words · **final 31,905 texts, 2,195,941 words**

| source | texts | words | median words | with period | with year |
|---|---|---|---|---|---|
| archibab | 1,522 | 79,733 | 39 | 1,522 | 0 |
| ebl | 16,552 | 966,964 | 26 | 0 | 0 |
| lbl_letters | 1,020 | 62,571 | 53 | 1,020 | 0 |
| oracc | 11,268 | 800,754 | 28 | 3,431 | 0 |
| orcc | 1,185 | 245,555 | 49 | 1,172 | 1,176 |
| seal | 358 | 40,364 | 42 | 355 | 0 |

## period labels (harmonised)

| period | texts | sources |
|---|---|---|
| Achaemenid | 15 | oracc:12, orcc:3 |
| Hellenistic | 401 | oracc:400, orcc:1 |
| Late Babylonian | 1,084 | lbl_letters:1020, seal:64 |
| Middle Assyrian | 6 | seal:6 |
| Middle Babylonian | 80 | seal:52, orcc:28 |
| Neo-Assyrian | 3,954 | oracc:3014, orcc:924, seal:16 |
| Neo-Babylonian | 221 | orcc:216, oracc:5 |
| Old Assyrian | 5 | seal:5 |
| Old Babylonian | 1,734 | archibab:1522, seal:212 |

unmapped period strings (top): `wall slab (with reliefs)`×4; `Archaic/Old Akkadian/Ebla`×2; `Later Periods (SB, NA, LB)`×1

## splits

| split | archibab | ebl | lbl_letters | oracc | orcc | seal |
|---|---|---|---|---|---|---|
| dated | 0 | 0 | 0 | 0 | 1176 | 0 |
| test | 152 | 1655 | 102 | 1126 | 0 | 35 |
| train | 1218 | 13242 | 816 | 9016 | 9 | 288 |
| val | 152 | 1655 | 102 | 1126 | 0 | 35 |
