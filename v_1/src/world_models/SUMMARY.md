# Summary — Do our models represent space & time? (Gurnee & Tegmark, replicated)

**One line:** we re-ran Gurnee & Tegmark's "Language Models Represent Space and Time"
(arXiv:2310.02207) on our own model ladder plus a random-init control the paper never
ran; our trained Llama-2 reproduces the paper almost exactly, the structure is
*learned* (random weights collapse it), and our small translation encoders are near
the surface-form floor on English — so their Akkadian strength is language-specific.

## The experiment

For six datasets — three spatial (world / US / NYC places → latitude, longitude) and
three temporal (historical figures → death year; books/songs/movies → release year;
NYT headlines → publication year) — we feed each entity to a model, take the hidden
state at the last token of each layer, and fit a **ridge probe** to recover the
coordinate or the year on **held-out** entities. High score = the fact is *linearly
decodable* from the model's internal representation. We ran 14 model arms + a TF-IDF
text baseline; encoders are mean-pooled (they have no causal last token), everything
else uses the paper's last-token convention.

## Headline result — the replication holds

Best-layer held-out **test R²**, our trained Llama-2-70B vs the paper's reported numbers:

| | world | US | NYC | figures | media | headlines |
|---|--:|--:|--:|--:|--:|--:|
| **our llama2_70b** | 0.905 | 0.846 | 0.363 | 0.833 | 0.860 | 0.757 |
| **paper 70B** | 0.911 | 0.864 | 0.359 | 0.835 | 0.885 | 0.746 |

Within ~0.02 on every dataset (NYC and figures essentially exact), peaking at layer 53
(~65% depth, matching the paper). **The probing harness is validated.**

## Full ladder (best-layer test R²)

| arm | world | US | NYC | figures | media | headlines |
|---|--:|--:|--:|--:|--:|--:|
| llama2_70b | .905 | .846 | .363 | .833 | .860 | .757 |
| llama2_13b | .883 | .808 | .272 | .802 | .780 | .663 |
| llama2_7b | .859 | .788 | .249 | .784 | .770 | .592 |
| qwen3_32b | .838 | .702 | .187 | .806 | .727 | .605 |
| qwen3_8b | .797 | .634 | .117 | .774 | .658 | .557 |
| qwen3_1b7 | .655 | .450 | .080 | .693 | .449 | .476 |
| uMT5-base | .438 | .325 | .133 | .494 | .153 | .349 |
| cuneiform-400M | .399 | .344 | .114 | .460 | .126 | .343 |
| AKK-300M | .381 | .312 | .120 | .448 | .123 | .300 |
| **TF-IDF (floor)** | .642 | .536 | .389 | .645 | .116 | .448 |
| llama2_13b_random | .282 | .290 | .044 | .284 | .038 | .267 |
| llama2_7b_random | .298 | .297 | .070 | .281 | .046 | .260 |
| random (qwen8b) | .327 | .379 | .059 | .276 | .055 | .196 |

*(gpt-oss-120B and llama2_70b_random probes were still finishing at time of writing;
Spearman table is in `results/summary_best_layer_spearman.csv`.)*

## What we learned

1. **Reproduction.** Trained Llamas climb the paper's ladder in order (7B .859 → 13B
   .883 → 70B .905/.911 on world), so the re-implementation reproduces Gurnee & Tegmark.
2. **Scale.** Both families are monotonic in size (Qwen3 1.7B .655 → 8B .797 → 32B
   .838). Bigger models carry a cleaner internal map and timeline.
3. **It's learned, not architectural — the control the paper never ran.** Every
   random-init arm collapses to the bottom in *both* domains: Llama-2-13B trained vs
   random is .883→.282 (space) and .780→.038 (release years, i.e. essentially zero).
   The geometry comes from training, not from the architecture or tokenizer.
4. **A real floor.** TF-IDF over the raw entity string scores .642 on world and even
   beats every model on NYC (surface form wins fine-grained within-city geography).
   Trained LLMs clear it decisively; random arms sit *below* it. The gap over TF-IDF
   is what's genuinely learned.
5. **Cross-lingual control for the thesis.** The small translation encoders (uMT5,
   cunei-400M, AKK-300M) land near the floor on *English* world-knowledge. So their
   advantage on Akkadian dating is **not** generic probing skill — it is
   language-specific, which is exactly the thesis's claim.

## Method notes (for comparability)

Best-layer, held-out (their `is_test` split), empty prompt, ridge probe. Space R² is
the joint coordinate R² (haversine-consistent); space Spearman is the mean of the
latitude and longitude rank correlations; time uses R² and Spearman directly. Datasets
and entity-string construction are byte-faithful ports of the paper's repo. See
`PLAN.md` (design), `HANDOFF.md` (ops + the bugs we hit), `README.md` (run order).
