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

## C6b — kernel CKA, λ ∈ {1, 5} (2026-09-02, evening)

| encoder | λ | logged CKA at end | provenance from h: lin / MLP (raw lin) | mc ρ orig |
|---|---|---|---|---|
| cunei400m | 1 | .06–.11 | .42 / .32 (.44) | .58 ± .02 |
| cunei400m | 5 | — | .42 / .34 (.44) | .54 ± .01 |
| Llama-2-7B | 1 | — | .40 / .31 (.42) | .51 ± .03 |
| Llama-2-7B | 5 | — | .39 / .30 (.42) | .51 ± .04 |
| Qwen3-8B | 1 | — | .29 / .20 (.41) | .43 ± .04 |
| Qwen3-8B | 5 | — | .27 / .18 (.41) | .43 ± .02 |

**Also null.** The penalty was active this time (batch CKA driven to ≈ .1) and
still a linear probe reads provenance from h at .4. Minimising a batch-level
dependence statistic does not prevent decodability. A gradient-reversal
adversary was prototyped and collapsed (the adversary reached CE ≈ 0, both
gradients vanished); the confusion-loss variant did not reach chance either
in a synthetic check. Not pursued further.

## Decision (Yarin, 2026-09-02 evening) — this line is closed

The method is augmentation invariance: one document, several *text* views
(clean / names masked / cropped / formulas stripped) must embed the same.
That is what was trained and what won (E-MIN v2). "Remove find-spot from the
head" was an *interpretation* question about the result, wrongly turned into a
training objective. It stays in the paper as a stated limitation — the head's
gain on LLM features co-varies with find-spot information — and as a failed side
experiment, not as part of the method. If the idea is ever revisited, the
SSL-consistent form is one more text view (`mask_place`: toponyms masked like
`mask_ruler`), not a penalty.
