# SEAL Task Registry — Self-Test Verification

Generated: `2026-04-14T14:14:41Z`
Source: `v_1/src/bias_check/seal_tasks.py` (Phase B self-test)
Contract: `v_1/data/raw/chungrong/seal_round4/inspection_report.json`

This report confirms that `load_task_data()` reproduces the fragment and class counts predicted by the Phase 0 inspection script.  A mismatch here means the parquet or the registry has drifted from the agreed contract.

## Summary

| Task | Corpora | N in | After null | Classes | Singletons | Classes left | N left | k | Status |
|------|---------|-----:|-----------:|--------:|-----------:|-------------:|-------:|--:|--------|
| `period` | seal+dll+lbpl | 384 | 384 | 10 | 1 | 9 | 383 | 2 | ✓ PASS |
| `genre` | seal+dll+lbpl | 384 | 384 | 16 | 0 | 16 | 384 | 2 | ✓ PASS |
| `sub_genre` | seal | 302 | 281 | 78 | 35 | 43 | 246 | 2 | ✓ PASS |
| `provenance` | seal+dll+lbpl | 384 | 384 | 35 | 10 | 25 | 374 | 2 | ✓ PASS |
| `sub_provenance` | seal+dll+lbpl | 384 | 384 | 35 | 10 | 25 | 374 | 2 | ✓ PASS |
| `domain` | seal+dll+lbpl | 384 | 384 | 3 | 0 | 3 | 384 | 5 | ✓ PASS |

## Task: `period`

- label column: `period`
- corpora pooled: ['seal', 'dll', 'lbpl']
- N fragments (input): 384
- N fragments (after null filter): 384
- N classes (input): 10; singletons: 1
- N classes (after singleton drop): 9
- N fragments (after singleton drop): 383
- smallest surviving class size: 2
- effective k: 2

- singletons dropped: ['Later Periods (SB, NA, LB)']

Top 5 classes by N:

  - `Old Babylonian`: 229
  - `Late Babylonian`: 38
  - `Middle Babylonian/Assyrian`: 35
  - `Neo or Late Babylonian`: 26
  - `Middle Babylonian`: 24

All counts match the Phase 0 inspection contract. ✓

## Task: `genre`

- label column: `genre`
- corpora pooled: ['seal', 'dll', 'lbpl']
- N fragments (input): 384
- N fragments (after null filter): 384
- N classes (input): 16; singletons: 0
- N classes (after singleton drop): 16
- N fragments (after singleton drop): 384
- smallest surviving class size: 2
- effective k: 2

Top 5 classes by N:

  - `incantations`: 161
  - `epics and myths`: 48
  - `lyrics`: 33
  - `hymns and prayers`: 28
  - `rituals`: 23

All counts match the Phase 0 inspection contract. ✓

## Task: `sub_genre`

- label column: `sub_genre`
- corpora pooled: ['seal']
- N fragments (input): 302
- N fragments (after null filter): 281
- N classes (input): 78; singletons: 35
- N classes (after singleton drop): 43
- N fragments (after singleton drop): 246
- smallest surviving class size: 2
- effective k: 2

- singletons dropped: ['adad', 'adapa', 'amurru', 'baby disease', 'collapse', 'descent to the netherworld', 'diarrhea', 'eye (disease)', 'fables', 'gall', 'girra', 'goat', 'ipiq-ištar', 'itūrmēr', 'lamasaga/baba'] (+20 more)

Top 5 classes by N:

  - `scorpions`: 19
  - `gastrointestinal problems`: 16
  - `dogs`: 14
  - `love`: 13
  - `gilgameš`: 13

All counts match the Phase 0 inspection contract. ✓

## Task: `provenance`

- label column: `provenance`
- corpora pooled: ['seal', 'dll', 'lbpl']
- N fragments (input): 384
- N fragments (after null filter): 384
- N classes (input): 35; singletons: 10
- N classes (after singleton drop): 25
- N fragments (after singleton drop): 374
- smallest surviving class size: 2
- effective k: 2

- singletons dropped: ['Assur;Nineveh', 'Babylon;Borsippa', 'Babylon;Sippar', 'Emar;Ugarit', 'Ešnunna', 'Larsa;Lagaba', 'Malgium', 'Meturan', 'Sippar;Nippur', 'Unknown;Larsa']

Top 5 classes by N:

  - `Unknown`: 112
  - `Babylon`: 59
  - `Larsa area`: 32
  - `Sippar`: 19
  - `Nineveh`: 19

All counts match the Phase 0 inspection contract. ✓

## Task: `sub_provenance`

- label column: `sub_provenance`
- corpora pooled: ['seal', 'dll', 'lbpl']
- N fragments (input): 384
- N fragments (after null filter): 384
- N classes (input): 35; singletons: 10
- N classes (after singleton drop): 25
- N fragments (after singleton drop): 374
- smallest surviving class size: 2
- effective k: 2

- singletons dropped: ['Unknown;mod. Tell es-Senkereh', 'mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes;mod. Tell Abu Ḥabbah', 'mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes;modern Birs Nimrud', 'mod. Qalʿat Sharqat;mod. Kouyunjik, Tell Nabi Yunus', 'mod. Tell Abu Ḥabbah;mod. Nuffar', 'mod. Tell Asmar', 'mod. Tell Haddad', 'mod. Tell Meskene;mod. Ras Shamrah', 'mod. Tell Yassir', 'mod. Tell es-Senkereh;between Babylon and Kutha']

Top 5 classes by N:

  - `Unknown`: 112
  - `mod. Kasr, Amran Ibn Ali, Sahn, Ishin Aswad, Merkes`: 59
  - `Larsa area`: 32
  - `mod. Tell Abu Ḥabbah`: 19
  - `mod. Kouyunjik, Tell Nabi Yunus`: 19

All counts match the Phase 0 inspection contract. ✓

## Task: `domain`

- label column: `domain`
- corpora pooled: ['seal', 'dll', 'lbpl']
- N fragments (input): 384
- N fragments (after null filter): 384
- N classes (input): 3; singletons: 0
- N classes (after singleton drop): 3
- N fragments (after singleton drop): 384
- smallest surviving class size: 38
- effective k: 5

Top 5 classes by N:

  - `SEAL`: 302
  - `DLL`: 44
  - `LBPL`: 38

All counts match the Phase 0 inspection contract. ✓

