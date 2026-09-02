# P2 step 1 — HSIC deconfounding of the head's hidden layer vs provenance (C6) — 2026-09-02

Setup: Chrono-Barlow head on raw tier0 Akkadian features + λ·HSIC(h, provenance one-hot),
λ ∈ {1, 10}, 3 encoders × 3 seeds. Acceptance test: `probe_head_hidden --no-erase`
(can a linear / MLP probe read provenance from h? chance .07). Reference: unpenalised
head (E-MIN v2) and raw-feature readability.

| encoder | λ | provenance from raw X (lin) | **from h: lin / MLP** | mc ρ orig | crop16 |
|---|---|---|---|---|---|
| cunei400m | 0 (v2) | .44 | .06* / .17* | .61 ± .01 | .49 |
| cunei400m | 1 | .44 | .43 / .29 | .60 ± .01 | .49 |
| cunei400m | 10 | .44 | .43 / .30 | .60 ± .01 | .48 |
| Llama-2-7B | 1 | .42 | .41 / .32 | .53 ± .04 | .46 |
| Llama-2-7B | 10 | .42 | .41 / .34 | .53 ± .04 | .46 |
| Qwen3-8B | 1 | .41 | .28 / .20 | .43 ± .01 | .40 |
| Qwen3-8B | 10 | .41 | .28 / .20 | .44 ± .03 | .40 |

\* the v2 row's h-readability was measured on LEACE-erased input (head ladder); the
HSIC rows feed raw input, so the comparable reference is the raw-X column.

**Result: null.** With λ ≤ 10 the penalty changes neither the dating accuracy nor
the readability of provenance from the hidden layer (only Qwen moves, .41 → .28).
λ = 1 and λ = 10 give the same numbers, which says the penalty's gradient is
negligible against the Barlow/ordering terms — a scale problem, not a sign that
deconfounding is impossible: biased RBF-HSIC with a median-heuristic bandwidth on
a 256-row batch and a 21-class one-hot is O(10⁻²), while the Barlow loss is O(5).

**Next (C6b):** replace raw HSIC by kernel CKA = HSIC(h,Z)/√(HSIC(h,h)·HSIC(Z,Z)),
which is scale-free in [0, 1], log its per-epoch value, and sweep λ ∈ {1, 5}. Same
acceptance probe. If CKA at λ = 5 does not move readability either, the next
lever is an adversarial provenance classifier on h.
