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

## Cluster jobs — paste-ready (run in parallel)

All sbatch live in `sbatch/`. Each pulls THIS branch, runs, and pushes results
(big `*.npz` activations are gitignored / cluster-local; only JSON summaries +
coverage come back). Paste into the Schmidt web terminal:

```bash
cd ~/projects/HUJI-THESIS--YARIN && git pull origin claude/stress-test-timeline-analysis-9sh2vs
# --- Test 1: T9 direct knowledge (kp0/kp1/kp2) ---
sbatch v_1/src/stress_tests/sbatch/J2a_t9_qwen3.sbatch      # qwen3 x3 (array)
sbatch v_1/src/stress_tests/sbatch/J2b_t9_gptoss.sbatch     # gpt-oss-120B (gpu:4)
# --- Test 2: T10 prompt-reprobe (mean + king_last + king_mean, pv0-pv3) ---
sbatch v_1/src/stress_tests/sbatch/J3a_t10_qwen3.sbatch     # qwen3 x3 (array)
sbatch v_1/src/stress_tests/sbatch/J3b_t10_gptoss.sbatch    # gpt-oss-120B (gpu:4)
# --- Test 3: king-token activation extraction (tier0) ---
sbatch v_1/src/stress_tests/sbatch/J4_king_extract.sbatch   # 6 models (array)
sbatch v_1/src/stress_tests/sbatch/J4b_king_extract_gptoss.sbatch  # gpt-oss-120B (gpu:4)
```
All six are independent — fire them together; `squeue -u $USER` to watch.

## Status
- **J1 shared harness — DONE**, validated locally.
- **J2 (T9 knowledge), J3 (T10 prompt), J4 (king extraction) — BUILT**, scripts +
  sbatch committed; compile-checked + reprobe logic unit-tested locally.
  Awaiting cluster run (no GPU/models locally).
- J5 (P3 anchors), J6 (P1 probe), J7 (P2 geography), J9 (P7 k-sparse), J10 (GUI),
  J11 (aggregate) — pending.

## What still needs a human
- `ruler_spellings.csv` — expert review to lift king-token coverage (now ~44%).
- gpt-oss-120B GPU count: sbatch requests `gpu:4`; adjust if the cluster fits it in fewer.
