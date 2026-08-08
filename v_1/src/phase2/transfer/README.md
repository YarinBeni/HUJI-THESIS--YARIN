# E3 — frozen name-direction transfer + LEACE mediation

Phase-2 experiment E3 (see `../DECIDED_EXPERIMENTS.md`). The question: is the
document-side time axis the SAME axis cell A found for entity names, only weaker
— or a different axis entirely? And when the frozen direction does order
fragments, is that ordering mediated by ruler identity?

## How it works

1. **Freeze.** Take the cell-A ridge direction (`probe_wm.py` saved coef at its
   best layer, raw-activation coordinates; entity set `historical_figure`, the
   ρ≈.88 probe). No refitting, ever.
2. **Transfer.** Score every fragment: s = coef·x at the same residual depth.
   Zero document-side fitting means zero leakage — all 1,187 fragments are test
   data. Read-outs: Spearman(s, year) and the E1 pairwise evaluation with s as
   scorer (macro over ruler-pairs — directly comparable to E1's table).
3. **Mediate.** LEACE-erase one-hot ruler identity (rank ≤ 39 nick out of
   d=4096), re-apply the same frozen direction. Collapse ⇒ the transfer was an
   identity lookup; survival ⇒ a ruler-independent time component. (Per the
   ICC=1 degeneracy this is a mediation test, not a "does year survive" test.)
4. **Converge.** Cosine between the frozen cell-A direction and E1's pairwise
   direction (trained on relative order only, no absolute years), both moved to
   the standardized coordinates the pairwise probe lived in. Two independently
   derived time directions agreeing is convergent evidence neither could give
   alone.

Interpretation grid (transfer × erasure):

| | survives LEACE | collapses under LEACE |
|---|---|---|
| **transfer works** | ruler-independent document time, same axis as names | transfer = identity lookup → THE collapse mechanism |
| **transfer fails** | — | entity time and document time are different axes |

## Files

| file | what |
|---|---|
| `e3_transfer.py` | the whole experiment; `--acts-root/--dirs-root` overrides for testing |
| `sbatch/F5_e3_transfer.sbatch` | 4 arms × {akk_maximal, eng_tier0}; regenerates missing cell-A directions via probe_wm; `pip install concept-erasure` |

Needs (cluster-local): the akkadian npz store + `world_models/results/directions/`.

## Progress

- [x] `e3_transfer.py` written; synthetic smoke passes (planted direction: Spearman +.68, pairwise .93 — polarity verified; LEACE path importable-guarded)
- [x] F5 sbatch written (direction auto-regeneration fallback included)
- [x] F5 ran (job 22608, all 4 arms × 2 variants)
- [x] **Result: TRANSFER FAILS — bottom-left cell of the grid.** Frozen-direction
  Spearman spans −.17..+.07 (≈0), pairwise macro .44–.54 (≈chance), and the
  random twins are indistinguishable from the trained arms. LEACE moves nothing
  because there is nothing to mediate. Decisively: **cosine between the cell-A
  name direction and E1's pairwise document direction ≈ .01–.03 on every arm —
  chance-level orthogonality in d=4096 (null ≈ 1/√4096 ≈ .016)**. The entity
  time axis and the document time axis are DIFFERENT axes; the collapse is not
  dilution of one shared direction. This is a headline phase-2 finding: it rules
  out H-dilute and lands on H-rotate/H-absent.
- [x] Consequence for E2: steering uses the cell-A direction on cell-A/B prompts
  (still valid — same regime it was fitted in); using it as a document read-out
  is now known to be pointless.
