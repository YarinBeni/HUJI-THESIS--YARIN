# SSL corpus census — 2026-09-02

rows before dedupe 47,001 · content duplicates removed 1,012 (cross-source duplicate groups: 47) · min length 8 words · **final 35,379 texts, 2,392,997 words**

| source | texts | words | median words | with period | with year |
|---|---|---|---|---|---|
| archibab | 2,693 | 143,342 | 40 | 1,522 | 0 |
| ebl | 16,552 | 966,964 | 26 | 0 | 0 |
| lbl_letters | 1,020 | 62,571 | 53 | 1,020 | 0 |
| oracc | 13,571 | 934,201 | 31 | 5,716 | 0 |
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
| Neo-Assyrian | 6,239 | oracc:5299, orcc:924, seal:16 |
| Neo-Babylonian | 221 | orcc:216, oracc:5 |
| Old Assyrian | 5 | seal:5 |
| Old Babylonian | 1,734 | archibab:1522, seal:212 |

unmapped period strings (top): `wall slab (with reliefs)`×4; `Archaic/Old Akkadian/Ebla`×2; `Later Periods (SB, NA, LB)`×1

## splits

| split   |   archibab |   ebl |   lbl_letters |   oracc |   orcc |   seal |
|:--------|-----------:|------:|--------------:|--------:|-------:|-------:|
| dated   |          0 |     0 |             0 |       0 |   1176 |      0 |
| test    |        269 |  1655 |           102 |    1357 |      0 |     35 |
| train   |       2155 | 13242 |           816 |   10857 |      8 |    288 |
| val     |        269 |  1655 |           102 |    1357 |      1 |     35 |
