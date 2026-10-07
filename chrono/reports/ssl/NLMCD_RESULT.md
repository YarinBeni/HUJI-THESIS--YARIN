# NLMCD-style concept manifolds

UMAP->8d, HDBSCAN(min_cluster_size=80) over 30,729 undated documents; clusters labelled by majority metadata. Within source-dominated clusters: period silhouette (full-dim) vs a 100-permutation null. Cross-representation alignment: adjusted Rand on shared non-noise documents (CBA-inspired).

## `Thalesian/cuneiformBase-400m::L12::mean` — 22 manifolds, 3,596 noise docs

| cluster | n | majority source (purity) | majority period (purity) | majority genre (purity) | period silhouette inside | p |
|---|---|---|---|---|---|---|
| 0 | 418 | oracc (1.00) | — |  (1.00) | — | — |
| 1 | 1,036 | oracc (1.00) | — |  (1.00) | — | — |
| 2 | 1,701 | archibab (0.89) | Old Babylonian (1.00) | lettre administrative (0.57) | — | — |
| 3 | 1,147 | lbl_letters (0.89) | Late Babylonian (0.96) | letter (0.89) | — | — |
| 4 | 304 | oracc (0.99) | Hellenistic (1.00) | Legal (0.98) | — | — |
| 5 | 271 | seal (0.82) | Old Babylonian (0.84) | incantations (0.43) | +0.047 (null +0.003±0.027) | 0.08 |
| 6 | 2,497 | oracc (0.89) | Neo-Assyrian (1.00) | Letter (0.82) | — | — |
| 7 | 1,096 | oracc (0.97) | Neo-Assyrian (1.00) | Legal (0.35) | — | — |
| 8 | 99 | ebl (0.96) | — | ['ARCHIVAL'] (0.53) | — | — |
| 9 | 532 | oracc (0.61) | Neo-Assyrian (1.00) |  (0.61) | — | — |
| 10 | 186 | ebl (1.00) | — | see genres.json file (0.60) | — | — |
| 11 | 1,580 | ebl (0.82) | Hellenistic (0.70) | see genres.json file (0.36) | +0.152 (null -0.001±0.026) | 0.00 |
| 12 | 263 | oracc (0.92) | Neo-Assyrian (1.00) |  (0.90) | — | — |
| 13 | 157 | oracc (0.65) | — |  (0.65) | — | — |
| 14 | 174 | ebl (0.55) | Neo-Assyrian (1.00) |  (0.42) | — | — |
| 15 | 116 | ebl (0.91) | — | [] (0.22) | — | — |
| 16 | 144 | oracc (1.00) | — |  (1.00) | — | — |
| 17 | 221 | oracc (1.00) | — |  (1.00) | — | — |
| 18 | 12,788 | ebl (0.86) | Neo-Assyrian (0.79) | [] (0.33) | — | — |
| 19 | 900 | ebl (0.99) | Neo-Assyrian (1.00) | ['CANONICAL ➝ Technical ➝ Astronomy ➝ Astronomical Diaries'] (0.80) | — | — |
| 20 | 1,135 | oracc (0.96) | Neo-Assyrian (1.00) |  (0.96) | — | — |
| 21 | 368 | oracc (0.99) | — |  (0.99) | — | — |

## `Qwen/Qwen3-8B::L27::mean` — 8 manifolds, 2,287 noise docs

| cluster | n | majority source (purity) | majority period (purity) | majority genre (purity) | period silhouette inside | p |
|---|---|---|---|---|---|---|
| 0 | 1,457 | archibab (0.97) | Old Babylonian (1.00) | lettre administrative (0.62) | — | — |
| 1 | 85 | oracc (1.00) | — |  (1.00) | — | — |
| 2 | 443 | oracc (0.95) | Neo-Assyrian (1.00) |  (0.93) | — | — |
| 3 | 887 | oracc (1.00) | — |  (1.00) | — | — |
| 4 | 102 | oracc (0.99) | — |  (0.99) | — | — |
| 5 | 24,058 | ebl (0.66) | Neo-Assyrian (0.67) | [] (0.22) | -0.084 (null -0.075±0.046) | 0.62 |
| 6 | 316 | oracc (0.93) | Hellenistic (0.94) | Legal (0.61) | — | — |
| 7 | 1,094 | oracc (0.98) | — |  (0.98) | — | — |

## `ssl::ssl_byol_cunei400m_wdated-s0::L0::h` — 3 manifolds, 31 noise docs

| cluster | n | majority source (purity) | majority period (purity) | majority genre (purity) | period silhouette inside | p |
|---|---|---|---|---|---|---|
| 0 | 1,741 | archibab (0.84) | Old Babylonian (0.98) | lettre administrative (0.53) | +0.205 (null -0.006±0.056) | 0.00 |
| 1 | 27,628 | ebl (0.59) | Neo-Assyrian (0.87) |  (0.28) | -0.048 (null -0.060±0.052) | 0.53 |
| 2 | 1,329 | lbl_letters (0.76) | Late Babylonian (0.91) | letter (0.76) | +0.444 (null +0.002±0.036) | 0.00 |

## `ssl_e2e::e2e_barlow_S-s0::L0::h` — 154 manifolds, 3,027 noise docs

| cluster | n | majority source (purity) | majority period (purity) | majority genre (purity) | period silhouette inside | p |
|---|---|---|---|---|---|---|
| 0 | 1,012 | lbl_letters (1.00) | Late Babylonian (1.00) | letter (1.00) | — | — |
| 1 | 122 | oracc (0.70) | Hellenistic (0.99) | Legal (0.60) | — | — |
| 2 | 110 | ebl (1.00) | — | [] (0.73) | — | — |
| 3 | 80 | ebl (0.96) | — | [] (0.44) | — | — |
| 4 | 82 | oracc (1.00) | — |  (1.00) | — | — |
| 5 | 92 | oracc (1.00) | — |  (1.00) | — | — |
| 6 | 173 | oracc (0.60) | Hellenistic (1.00) |  (0.60) | — | — |
| 7 | 132 | ebl (0.99) | — | [] (0.45) | — | — |
| 8 | 104 | ebl (0.97) | Neo-Assyrian (1.00) | [] (0.38) | — | — |
| 9 | 141 | ebl (0.95) | Hellenistic (0.50) | ['ARCHIVAL ➝ Administrative'] (0.24) | — | — |
| 10 | 128 | oracc (1.00) | — |  (1.00) | — | — |
| 11 | 82 | oracc (0.90) | — |  (0.90) | — | — |
| 12 | 181 | oracc (0.98) | Neo-Assyrian (1.00) | Letter (0.94) | — | — |
| 13 | 86 | archibab (1.00) | Old Babylonian (1.00) | lettre administrative (0.99) | — | — |
| 14 | 126 | archibab (1.00) | Old Babylonian (1.00) | lettre administrative (0.62) | — | — |
| 15 | 115 | oracc (0.71) | — |  (0.71) | — | — |
| 16 | 119 | oracc (0.98) | Neo-Assyrian (1.00) | Letter (0.92) | — | — |
| 17 | 143 | ebl (0.57) | Hellenistic (1.00) |  (0.34) | — | — |
| 18 | 98 | ebl (0.86) | — | [] (0.24) | — | — |
| 19 | 278 | ebl (0.98) | — | ['ARCHIVAL'] (0.42) | — | — |
| 20 | 339 | ebl (0.68) | Achaemenid (0.73) | see genres.json file (0.58) | — | — |
| 21 | 162 | ebl (0.76) | Hellenistic (1.00) | [] (0.48) | — | — |
| 22 | 143 | oracc (0.97) | Neo-Assyrian (1.00) |  (0.96) | — | — |
| 23 | 221 | ebl (0.94) | — | [] (0.45) | — | — |
| 24 | 109 | oracc (0.91) | — |  (0.91) | — | — |
| 25 | 131 | oracc (0.98) | — |  (0.98) | — | — |
| 26 | 140 | oracc (0.96) | Neo-Assyrian (1.00) |  (0.93) | — | — |
| 27 | 219 | oracc (0.99) | Neo-Assyrian (1.00) |  (0.95) | — | — |
| 28 | 205 | oracc (0.98) | Neo-Assyrian (1.00) |  (0.96) | — | — |
| 29 | 257 | oracc (0.95) | Neo-Assyrian (1.00) |  (0.93) | — | — |
| 30 | 89 | oracc (0.74) | Neo-Assyrian (1.00) |  (0.72) | — | — |
| 31 | 170 | ebl (0.81) | Old Babylonian (0.92) | [] (0.34) | — | — |
| 32 | 133 | oracc (0.55) | — |  (0.55) | — | — |
| 33 | 313 | oracc (0.94) | Neo-Assyrian (1.00) | Letter (0.88) | — | — |
| 34 | 178 | ebl (0.97) | — | see genres.json file (0.23) | — | — |
| 35 | 86 | oracc (0.88) | Hellenistic (0.96) | Legal (0.83) | — | — |
| 36 | 224 | ebl (0.91) | Neo-Assyrian (1.00) | [] (0.40) | — | — |
| 37 | 82 | oracc (0.82) | — |  (0.82) | — | — |
| 38 | 120 | ebl (0.95) | Old Babylonian (1.00) | see genres.json file (0.74) | — | — |
| 39 | 82 | ebl (0.99) | Neo-Assyrian (1.00) | [] (0.52) | — | — |
| 40 | 86 | ebl (0.94) | — | [] (0.43) | — | — |
| 41 | 102 | ebl (0.88) | — | see genres.json file (0.18) | — | — |
| 42 | 132 | ebl (0.98) | — | [] (0.28) | — | — |
| 43 | 90 | ebl (0.96) | Neo-Assyrian (1.00) | [] (0.42) | — | — |
| 44 | 153 | oracc (0.89) | — |  (0.89) | — | — |
| 45 | 204 | oracc (0.97) | — |  (0.97) | — | — |
| 46 | 145 | ebl (0.82) | — | [] (0.27) | — | — |
| 47 | 270 | ebl (0.86) | Old Babylonian (0.94) | [] (0.34) | — | — |
| 48 | 155 | ebl (0.91) | Old Babylonian (0.83) | [] (0.30) | — | — |
| 49 | 95 | ebl (1.00) | — | [] (0.41) | — | — |
| 50 | 231 | ebl (0.93) | Neo-Assyrian (1.00) | [] (0.27) | — | — |
| 51 | 99 | oracc (0.91) | Neo-Assyrian (1.00) |  (0.90) | — | — |
| 52 | 150 | oracc (0.95) | Neo-Assyrian (1.00) | Letter (0.91) | — | — |
| 53 | 122 | ebl (0.91) | Neo-Assyrian (1.00) | [] (0.38) | — | — |
| 54 | 98 | oracc (1.00) | — |  (1.00) | — | — |
| 55 | 324 | ebl (0.98) | — | see genres.json file (0.54) | — | — |
| 56 | 82 | ebl (0.65) | — |  (0.35) | — | — |
| 57 | 124 | ebl (0.99) | — | [] (0.41) | — | — |
| 58 | 118 | oracc (0.71) | — |  (0.71) | — | — |
| 59 | 223 | ebl (0.72) | — |  (0.28) | — | — |
| 60 | 164 | oracc (0.91) | Old Babylonian (0.67) |  (0.90) | — | — |
| 61 | 180 | ebl (0.97) | Neo-Assyrian (0.80) | ['ARCHIVAL'] (0.44) | — | — |
| 62 | 95 | ebl (0.99) | — | [] (0.37) | — | — |
| 63 | 102 | ebl (0.99) | — | [] (0.40) | — | — |
| 64 | 176 | ebl (0.99) | — | [] (0.62) | — | — |
| 65 | 287 | oracc (0.97) | Hellenistic (0.99) | Legal (0.78) | — | — |
| 66 | 148 | oracc (0.53) | Neo-Assyrian (1.00) |  (0.52) | — | — |
| 67 | 219 | oracc (0.93) | Neo-Assyrian (1.00) |  (0.92) | — | — |
| 68 | 164 | ebl (0.99) | — | ['CANONICAL ➝ Technical ➝ Astronomy'] (0.31) | — | — |
| 69 | 133 | archibab (1.00) | Old Babylonian (1.00) | lettre administrative (0.55) | — | — |
| 70 | 151 | oracc (0.99) | — |  (0.99) | — | — |
| 71 | 142 | archibab (1.00) | Old Babylonian (1.00) | lettre administrative (0.87) | — | — |
| 72 | 83 | seal (0.98) | Late Babylonian (0.79) | lyrics (0.36) | — | — |
| 73 | 297 | oracc (0.97) | Neo-Assyrian (1.00) |  (0.97) | — | — |
| 74 | 96 | ebl (0.88) | Neo-Assyrian (1.00) | [] (0.40) | — | — |
| 75 | 128 | oracc (0.98) | Neo-Assyrian (1.00) | Administrative (0.94) | — | — |
| 76 | 109 | ebl (0.97) | Neo-Assyrian (1.00) | [] (0.48) | — | — |
| 77 | 81 | oracc (0.56) | Neo-Assyrian (1.00) |  (0.46) | — | — |
| 78 | 351 | oracc (0.99) | Neo-Assyrian (1.00) | Legal (0.55) | — | — |
| 79 | 191 | oracc (1.00) | — |  (1.00) | — | — |
| 80 | 132 | ebl (0.77) | Hellenistic (0.25) | [['ARCHIVAL']] (0.34) | — | — |
| 81 | 107 | ebl (0.98) | — | ['CANONICAL ➝ Technical ➝ Astronomy ➝ Astronomical Diaries'] (0.30) | — | — |
| 82 | 85 | ebl (0.98) | — | [] (0.59) | — | — |
| 83 | 111 | oracc (0.96) | Neo-Assyrian (0.99) |  (0.35) | — | — |
| 84 | 89 | ebl (0.90) | Late Babylonian (1.00) | [] (0.40) | — | — |
| 85 | 90 | oracc (0.90) | — |  (0.90) | — | — |
| 86 | 106 | ebl (0.73) | Old Babylonian (1.00) | [] (0.43) | — | — |
| 87 | 142 | ebl (0.85) | Old Babylonian (1.00) | [] (0.40) | — | — |
| 88 | 172 | oracc (0.53) | Neo-Assyrian (1.00) |  (0.45) | — | — |
| 89 | 85 | ebl (0.93) | Neo-Assyrian (1.00) | [] (0.51) | — | — |
| 90 | 149 | ebl (0.72) | Neo-Assyrian (1.00) | [] (0.31) | — | — |
| 91 | 145 | ebl (0.88) | Neo-Assyrian (1.00) | [] (0.46) | — | — |
| 92 | 86 | oracc (1.00) | — |  (1.00) | — | — |
| 93 | 94 | ebl (0.86) | — | [] (0.54) | — | — |
| 94 | 342 | archibab (1.00) | Old Babylonian (1.00) | lettre administrative (0.58) | — | — |
| 95 | 241 | archibab (1.00) | Old Babylonian (1.00) | lettre administrative (0.67) | — | — |
| 96 | 98 | ebl (0.85) | Neo-Assyrian (1.00) | [] (0.46) | — | — |
| 97 | 116 | ebl (0.99) | — | [] (0.41) | — | — |
| 98 | 92 | archibab (1.00) | Old Babylonian (1.00) | lettre politique (0.38) | — | — |
| 99 | 82 | ebl (0.80) | — | [] (0.35) | — | — |
| 100 | 148 | oracc (0.91) | — |  (0.91) | — | — |
| 101 | 109 | ebl (0.99) | — | [['CANONICAL', 'Divination', 'Celestial']] (0.25) | — | — |
| 102 | 129 | ebl (0.99) | — | [] (0.46) | — | — |
| 103 | 236 | oracc (1.00) | Neo-Assyrian (1.00) | Letter (0.96) | — | — |
| 104 | 136 | ebl (0.85) | Neo-Assyrian (1.00) | [] (0.24) | — | — |
| 105 | 156 | ebl (0.99) | — | [] (0.28) | — | — |
| 106 | 309 | ebl (1.00) | — | [] (0.37) | — | — |
| 107 | 112 | ebl (0.94) | Late Babylonian (1.00) | [] (0.33) | — | — |
| 108 | 222 | archibab (0.64) | Old Babylonian (0.76) | lettre administrative (0.55) | +0.491 (null +0.001±0.011) | 0.00 |
| 109 | 387 | ebl (0.99) | — | [] (0.34) | — | — |
| 110 | 208 | ebl (0.99) | — | [] (0.42) | — | — |
| 111 | 113 | seal (0.97) | Old Babylonian (0.92) | incantations (0.32) | — | — |
| 112 | 159 | ebl (0.69) | — | ['ARCHIVAL'] (0.37) | — | — |
| 113 | 110 | ebl (0.95) | — | [] (0.40) | — | — |
| 114 | 137 | ebl (0.76) | Neo-Assyrian (0.67) | [] (0.31) | — | — |
| 115 | 87 | ebl (0.86) | Neo-Assyrian (1.00) | [] (0.53) | — | — |
| 116 | 134 | ebl (1.00) | — | [] (0.40) | — | — |
| 117 | 175 | ebl (0.98) | — | see genres.json file (0.43) | — | — |
| 118 | 106 | oracc (0.99) | — |  (0.99) | — | — |
| 119 | 211 | oracc (0.98) | Neo-Assyrian (1.00) | Legal (0.58) | — | — |
| 120 | 90 | oracc (0.79) | Neo-Assyrian (1.00) | Letter (0.77) | — | — |
| 121 | 211 | oracc (0.98) | Neo-Assyrian (1.00) | Administrative (0.56) | — | — |
| 122 | 312 | oracc (1.00) | Neo-Assyrian (1.00) | Letter (0.93) | — | — |
| 123 | 84 | ebl (0.96) | — | [] (0.50) | — | — |
| 124 | 104 | ebl (1.00) | — | see genres.json file (0.75) | — | — |
| 125 | 130 | oracc (0.98) | — |  (0.98) | — | — |
| 126 | 403 | oracc (0.97) | Neo-Assyrian (1.00) | Letter (0.92) | — | — |
| 127 | 326 | oracc (0.97) | Neo-Assyrian (1.00) | Letter (0.84) | — | — |
| 128 | 115 | oracc (1.00) | — |  (1.00) | — | — |
| 129 | 137 | ebl (0.91) | — | [] (0.28) | — | — |
| 130 | 106 | ebl (0.99) | — | [] (0.29) | — | — |
| 131 | 145 | ebl (1.00) | — | [['CANONICAL', 'Divination', 'Extispicy']] (0.34) | — | — |
| 132 | 292 | ebl (0.98) | — | [] (0.45) | — | — |
| 133 | 178 | ebl (0.99) | — | [] (0.38) | — | — |
| 134 | 156 | ebl (0.97) | — | [['CANONICAL', 'Divination', 'Celestial']] (0.29) | — | — |
| 135 | 161 | ebl (0.99) | — | [['CANONICAL', 'Divination', 'Celestial']] (0.24) | — | — |
| 136 | 353 | ebl (0.99) | — | [['CANONICAL', 'Divination', 'Celestial']] (0.28) | — | — |
| 137 | 168 | ebl (0.98) | — | [] (0.31) | — | — |
| 138 | 106 | ebl (0.94) | Old Babylonian (1.00) | [] (0.25) | — | — |
| 139 | 137 | ebl (1.00) | — | see genres.json file (0.41) | — | — |
| 140 | 153 | ebl (1.00) | — | [] (0.34) | — | — |
| 141 | 769 | oracc (0.94) | Neo-Assyrian (1.00) |  (0.94) | — | — |
| 142 | 285 | ebl (0.99) | — | [['CANONICAL', 'Divination', 'Extispicy']] (0.26) | — | — |
| 143 | 208 | ebl (0.99) | — | [] (0.32) | — | — |
| 144 | 374 | oracc (0.66) | Neo-Assyrian (0.80) |  (0.64) | — | — |
| 145 | 301 | ebl (0.97) | Neo-Assyrian (1.00) | [] (0.24) | — | — |
| 146 | 194 | ebl (1.00) | — | [['CANONICAL', 'Divination', 'Celestial']] (0.22) | — | — |
| 147 | 250 | ebl (0.99) | — | [] (0.36) | — | — |
| 148 | 102 | ebl (1.00) | — | [] (0.68) | — | — |
| 149 | 475 | ebl (1.00) | — | ['CANONICAL ➝ Technical ➝ Astronomy ➝ Astronomical Diaries'] (0.93) | — | — |
| 150 | 289 | ebl (1.00) | — | ['CANONICAL ➝ Technical ➝ Astronomy ➝ Astronomical Diaries'] (0.87) | — | — |
| 151 | 480 | ebl (0.99) | — | [] (0.38) | — | — |
| 152 | 500 | ebl (0.84) | Neo-Assyrian (0.86) | [] (0.35) | — | — |
| 153 | 553 | ebl (0.97) | Neo-Assyrian (1.00) | [] (0.33) | — | — |

## Cross-representation alignment (adjusted Rand, non-noise overlap)

| | `cuneiformBase-400m` | `Qwen3-8B` | `ssl` | `ssl_e2e` |
|---|---|---|---|---|
| `cuneiformBase-400m` | 1.000 | 0.240 | 0.152 | 0.052 |
| `Qwen3-8B` | 0.240 | 1.000 | 0.299 | 0.005 |
| `ssl` | 0.152 | 0.299 | 1.000 | 0.004 |
| `ssl_e2e` | 0.052 | 0.005 | 0.004 | 1.000 |
