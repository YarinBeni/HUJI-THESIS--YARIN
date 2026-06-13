# RESULTS — Akkadian NTP fine-tune + depth ablation (Task 5)

_Last updated: 2026-06-14. Source: `results/scoreboard_best.csv` (balanced 200-draw
year-PLS Spearman). COMPLETE — all four families (1.7B/8B/32B/gpt-oss-120b) probed._

## Headline

**Continued Akkadian NTP pretraining does not improve the dating signal on the
length-controlled (maximal) metric, at any model scale or unfreeze depth.** The
only place fine-tuning helps is gpt-oss-120b on **tier0** (full text) with
**full-depth** training — i.e. exactly where the length confound lives. This
*strengthens* the "length, not class" thesis: the gain appears on the metric
that leaks length and vanishes on the one that controls for it.

## Numbers — best-layer year-PLS Spearman (balanced)

### maximal (aggressive cleaning, ≤30 tokens — the honest metric)

| model | base | ft (best cut) | Δ |
|---|---|---|---|
| Qwen3-1.7B | 0.3549 (L9) | 0.3556 (ft09/19/25, L9) | +0.001 (noise) |
| Qwen3-8B | 0.3633 (L16) | 0.3646 (ft00, L14) | +0.001 (noise) |
| Qwen3-32B | 0.3398 (L6) | 0.3398 (ft21/43/58, L6) | 0.000 |
| gpt-oss-120b | 0.3301 (L5) | 0.3301 (ft12/24/32, L5) | 0.000 |

All flat — max Δ = +0.0013 (8B), well inside the ±0.08 draw-to-draw std.
**Full fine-tune is neutral-to-slightly-negative** (1.7B ft00 0.342, 32B ft00
0.337, gpt-oss ft00 0.327 — all ≤ base). The frozen-layer signature is starkest
at 32B: ft21/ft43/ft58 are **byte-identical to base (0.3398)** at L6.

### tier0 (full text — length/style leak in)

| model | base | ft (best cut) | Δ |
|---|---|---|---|
| Qwen3-1.7B | 0.397 (L2, σ=0.29 unreliable) | 0.372 (L9) | ~flat |
| Qwen3-8B | 0.420 (L3, σ=0.32 unreliable) | 0.376 (ft00, L15) | ~flat |
| Qwen3-32B | 0.433 (L5, σ=0.31 unreliable) | 0.399 (ft21/43/58, L9) | ~flat |
| **gpt-oss-120b** | **0.404 (L27)** | **0.451 (ft00, L17)** | **+0.048** |

(Base tier0 for the Qwen models has huge variance — the single-best-layer pick
is unstable; the FT arms have tight σ.)

## Mechanism — why partial-depth fine-tuning can't move it

"cut = k" freezes blocks below k. The maximal dating signal peaks **early**
(L5–L16 depending on model), at or below most cut points. So for any cut above
the peak, the winning layer's representation is **frozen → byte-identical to
base**. This is visible directly: gpt-oss ft12/ft24/ft32 all score **exactly
0.3301** (= base) at L5; Qwen3-8B ft24/ft32 both **exactly 0.3635** (= base) at
L16. The only arm that can change the signal is full-depth (cut0), and that is
neutral (Qwen) or helps only on tier0 (gpt-oss).

Interpretation: the year signal is an **early-layer, lexical/orthographic**
phenomenon (spelling drift over centuries), already present in the pretrained
lower layers. Continued NTP on the upper layers has nothing to add to it, and
retraining the lower layers (full FT) at best does nothing, at worst erodes it.

## gpt-oss-120b base — meeting Task 2

Untrained gpt-oss-120b: **0.404 tier0 / 0.330 maximal**. Strong on tier0 (best
*clean*-variance base model) but on maximal it drops *below* Qwen3-8B (0.363)
and 1.7B (0.355). **Scale does not win once length is controlled** — confirms
the core finding on the largest open model.

## Training sanity (NTP worked — it's the probe that's flat)

Val perplexity dropped cleanly everywhere (monotonic in unfrozen depth,
converged by epoch 2):

| model | base ppl | best ppl (cut0 / full-depth) |
|---|---|---|
| Qwen3-1.7B | 11.1 | 6.65 |
| Qwen3-8B | 8.1 | 5.39 |
| gpt-oss-120b (LoRA) | 14.0 | 4.73 |

So the models *did* learn Akkadian; lower LM loss simply does not translate to a
better dating probe — the decisive result.

## Status of jobs (see README run-log for IDs)

- ✅ ALL DONE: 1.7B (FT2/FT3), 8B (FT4/FT5), 32B (FT4b/FT5b), gpt-oss-120b
  (FT0/FT0b base; FT6/FT7 FT). Full ladder probed across maximal + tier0.
- gpt-oss-120b also folded into the canonical maximal panel set (fig1/2/4 +
  k-sweep, 9 models). Local comparison plot: `plot_maximal_pls.py --with-ft`
  → `results/figures/maximal_pls_{bestlayer,layerwise}.png`.
- Optional remaining polish: FT8/M6 to fold the best FT *cut* per family into
  the canonical fig1/2/4 (the local comparison plot already shows this).
