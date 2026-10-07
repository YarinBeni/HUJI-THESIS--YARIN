# Harvester validation (E1–E4)

970 documents carry at least one [PN]; 24 kings got a chosen spelling.

**E1** `validate_histograms.png` — green bars are the harvester's choices; grey bars are what it rejected.

## E2 — coverage per king

| king | docs with PN | first PN is a chosen spelling |
|---|---|---|
| Ashurbanipal | 226 | 41% |
| Sennacherib | 196 | 75% |
| Esarhaddon | 147 | 28% |
| Sargon II | 99 | 62% |
| Nebuchadnezzar II | 82 | 48% |
| Tiglath-pileser III | 54 | 30% |
| Nabonidus | 53 | 30% |
| Sîn-šarru-iškun | 18 | 50% |
| Nabopolassar | 14 | 57% |
| Shalmaneser V | 12 | 75% |
| Nabû-mukin-apli | 5 | 40% |
| Marduk-nadin-ahhe | 3 | 67% |
| Marduk-zakir-šumi I | 3 | 67% |
| Nabû-šuma-iškun | 2 | 100% |
| Antiochus I | 1 | 100% |
| Itti-Marduk-balaṭu | 1 | 100% |
| Ninurta-nadin-šumi | 1 | 100% |
| Enlil-nadin-apli | 1 | 100% |
| Nabû-šumu-libur | 1 | 100% |
| Ninurta-kudurri-usur I | 1 | 100% |
| Nabû-šumu-lišir | 1 | 100% |
| Marduk-šakin-šumi | 1 | 100% |
| Nabû-naṣir | 1 | 100% |
| Bel-ibni | 1 | 100% |

## E3 — ruler identification from the chosen spellings

Documents whose first PN is a chosen spelling: 476. Predicted king == metadata king: **96.0%**.

Worst confusions (true -> predicted, count):

- Esarhaddon -> Sennacherib: 11
- Ashurbanipal -> Sennacherib: 3
- Nebuchadnezzar II -> Nabopolassar: 2
- Esarhaddon -> Sargon II: 1
- Sennacherib -> Tiglath-pileser III: 1
- Sîn-šarru-iškun -> Sennacherib: 1

## E4 — spelling specificity (appears anywhere in other kings' texts)

| spelling | king | his docs | other kings' docs |
|---|---|---|---|
| `m-an-ti-'u-ku-us` | Antiochus I | 1 | 0 |
| `m-AN-ŠAR2-DU3-A` | Ashurbanipal | 81 | 7 |
| `m-te-um-man` | Ashurbanipal | 12 | 0 |
| `m-d-AG-ga-mil` | Bel-ibni | 1 | 0 |
| `d-en-lil2-SUM-IBILA` | Enlil-nadin-apli | 1 | 0 |
| `m-aš-šur-PAP-AŠ` | Esarhaddon | 26 | 14 |
| `m-AN-ŠAR2-ŠEŠ-SUM-NA` | Esarhaddon | 15 | 24 |
| `d-AMAR-UTU-DUGUD-ŠEŠ-MEŠ-šu2` | Itti-Marduk-balaṭu | 1 | 0 |
| `d-AMAR-UTU-SUM-ŠEŠ-MEŠ` | Marduk-nadin-ahhe | 2 | 2 |
| `d-AMAR-UTU-MU-MU` | Marduk-zakir-šumi I | 2 | 0 |
| `m-d-AMAR-UTU-GAR-MU` | Marduk-šakin-šumi | 1 | 1 |
| `d-AG-na-'i-id` | Nabonidus | 8 | 0 |
| `d-na-bi-um-na-'i-id` | Nabonidus | 8 | 0 |
| `d-na-bi-um-IBILA-u2-ṣu-ur2` | Nabopolassar | 5 | 23 |
| `d-AG-IBILA-u2-ṣur` | Nabopolassar | 3 | 11 |
| `m-ri-mut-DINGIR-MEŠ` | Nabû-mukin-apli | 2 | 0 |
| `m-d-EN-ib-ni` | Nabû-naṣir | 1 | 0 |
| `m-d-AMAR-UTU-IBILA-URU3` | Nabû-šuma-iškun | 1 | 0 |
| `m-d-AG-MU-im-bi` | Nabû-šuma-iškun | 1 | 0 |
| `d-AG-MU-li-bur` | Nabû-šumu-libur | 1 | 0 |
| `m-d-PA-MU-SI-SA2` | Nabû-šumu-lišir | 1 | 0 |
| `d-na-bi-um-ku-du-ur2-ri-u2-ṣu-ur2` | Nebuchadnezzar II | 25 | 0 |
| `d-AG-NIG2-DU-URU3` | Nebuchadnezzar II | 14 | 2 |
| `d-MAŠ-NIG2-DU-PAP` | Ninurta-kudurri-usur I | 1 | 1 |
| `d-nin-urta-SUM-MU` | Ninurta-nadin-šumi | 1 | 1 |
| `m-MAN-GIN` | Sargon II | 31 | 26 |
| `m-LUGAL-GI-NA` | Sargon II | 30 | 29 |
| `m-d-30-PAP-MEŠ-SU` | Sennacherib | 125 | 90 |
| `m-d-EN-ZU-ŠEŠ-MEŠ-eri-ba` | Sennacherib | 22 | 4 |
| `m-d-SILIM-man-MAŠ` | Shalmaneser V | 7 | 0 |
| `m-SILIM-man-MAŠ` | Shalmaneser V | 2 | 0 |
| `m-d-30-LUGAL-GAR-un` | Sîn-šarru-iškun | 7 | 0 |
| `m-d-EN-ZU-LUGAL-GAR-un` | Sîn-šarru-iškun | 2 | 0 |
| `m-tukul-ti-A-e2-šar2-ra` | Tiglath-pileser III | 13 | 5 |
| `m-ka-ki-i` | Tiglath-pileser III | 3 | 0 |
