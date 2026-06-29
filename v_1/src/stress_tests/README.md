# Stress-testing the "LLM timeline" finding on low-resource Akkadian

Replication-under-stress of the spatiotemporal "world-model geometry" literature
(Gurnee–Tegmark, Godey, *A Matter of Time*, k-sparse probing) on a low-resource,
indirect (date-not-in-text), no-web-leakage setting. Thesis claim: **declarative
knowledge and linearly/geometrically recoverable representation are separable** —
the model *states* reign dates yet does not encode a recoverable timeline over the
text, and that geometry is not installed by scale, prompting, or NTP finetuning.

Full plan: approved design doc (see commit history / PR). Pillars: **P1** Gurnee–Tegmark
year probe, **P2** Godey geography, **P3** A Matter of Time anchors, **P7** k-sparse;
plus redo of **T9** (knowledge) and **T10** (prompt reprobe) on all Qwen3 + gpt-oss-120B,
and a GUI embedding update.

## Pooling sites (locked)
- `mean` — masked mean over all tokens (tier0 + maximal).
- `king_last` — last token of the located king-name span (tier0 only).
- `king_mean` — mean over the king-name span tokens (tier0 only).

Whole-sentence last-token is dropped. king_* are tier0-only because `maximal`
cleaning strips the logograms/determinatives that spell royal names.

## shared/ (J1 — DONE, validated locally)
| file | purpose |
|---|---|
| `ruler_spellings.csv` | ruler → Akkadian transliteration spelling variants (NEEDS EXPERT REVIEW) |
| `sites_gazetteer.csv` | provenance → lat/lon/region (P2); **97.5%** row coverage |
| `king_token.py` | locate king-name span (word-level + tokenizer offset→token span) |
| `probe_sites.py` | `pool_mean` / `pool_king_last` / `pool_king_mean` |
| `metrics.py` | reuse `pls_utils.compute_metrics` + `proximity_error` + great-circle |
| `anchors.py` | P3 ruler/year anchor prompts |
| `build_and_validate.py` | J1 sanity checks → `results/j1_harness_report.json` |

## Key finding from J1: the king-token is intrinsically sparse
King-name word-level coverage is **~44% within mapped rulers** (93% of corpus mapped).
This is **not** a bug — in ~56% of ORCC "royal inscriptions" the commissioning king is
not named in a recoverable titulary token:
- **Assyrian royal inscriptions** locate well: Sennacherib 0.66, Sargon II 0.53,
  Ashurbanipal 0.47, Esarhaddon ~0.39.
- **Neo-Babylonian texts** are largely administrative/legal — every personal name
  appears once (df=1), the king is never in titulary → Nebuchadnezzar II ≈ 0%.

Consequence: the P1 `king_*` arm is a **subset experiment** (mostly the big Assyrian
rulers, ~500 fragments). The sparsity is itself thesis-relevant (the date signal is
structurally indirect). `ruler_spellings.csv` should be reviewed/expanded by the
Assyriologist advisor to raise coverage on the Assyrian-royal subset before P1 results
are finalized; rows are marked `verified` / `review` / `low_coverage_expected`.

## Status
- **J1 shared harness — DONE** (this commit).
- J2–J11 (T9/T10 redo, P1/P2/P3/P7 probes, GUI, aggregate) — pending.
