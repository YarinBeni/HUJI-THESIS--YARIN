# The gap audit (v2), the wave-4 fixes, and the wave-5 plan

A second, systematic pass over everything phase 2 ran — code and methodology —
beyond the four gaps the teaching guide already flagged. Every gap below is
either FIXED in code (wave 4 reruns it) or explicitly accepted with a reason.

## 1. Consolidated gap list

| # | gap | severity | status |
|---|---|---|---|
| G1 | **F8 firing measured at the last token only** — a year feature could fire mid-document and not survive to the end | HIGH — the headline mechanistic claim rests on it | FIXED: `sae/token_firing.py` encodes EVERY token (F11) |
| G2 | **Steering tested only late blocks (21–32) with fixed α** — the NAACL effects live in the first half, and late-layer residual norms dwarf a fixed push | HIGH — the causal null is not fair as run | FIXED: `--blocks`, `--alpha-mode rel` (push ∝ ‖h‖), chat option (F12) |
| G3 | **F2 behavioural ran without the chat template** on instruct-tuned Qwen3 — the No-bias may be a format artifact | MED | FIXED: `--chat` (+`enable_thinking=False`, no double-BOS) → `*.chat.json` (F10) |
| G4 | **E3 pooling mismatch**: direction fitted on `last` name tokens, applied to `mean` fragments | MED | FIXED: F13 reruns `--site last` |
| G5 | **E3 had no positive control** — "transfer fails" without showing the same code path succeeds on cell A could be a pipeline bug | MED | FIXED: `cellA_positive_control()` runs first, must give ρ≈.85+ (F13) |
| G6 | **LEACE hygiene**: fp32 whitening at d=4096≫n=1187 is ill-conditioned (qwen's 82% norm change smells numeric), and no check that erasure actually erased | MED | FIXED: fp64 fit + ruler-probe accuracy before/after (F13) |
| G7 | **Bradley–Terry scorer had an intercept**, breaking the antisymmetry P(a,b)+P(b,a)=1 | LOW — order randomization makes it ≈0; verified locally: floor unchanged (.643 vs .647 at 2 draws, within noise) | FIXED in code; F1 results stand, no rerun |
| G8 | **"Pairwise direction lenses to junk" had no baseline** — junk is what random directions produce too | LOW-MED | FIXED: 3 random-direction calibration controls added to the lens (F14) |
| G9 | **F1 layer selection used stride 2** — may have missed the Llama family's best eng layer (would explain "why is Llama flat on English?") | MED | FIXED: F14 reruns llama 7/13/70B eng at stride 1 |
| G10 | Fold sizes equal in ruler count, wildly unequal in fragments | LOW | ACCEPTED: macro metric + dyadic bootstrap already absorb it |
| G11 | Permutation floor .0066 (150 perms); layer fixed from F1 (mild anti-conservatism) | LOW | ACCEPTED + documented in RESULTS §4; 1000-perm rerun possible if a reviewer demands |
| G12 | The pairwise "time direction" exported from one draw's full fit (noisy) | LOW | ACCEPTED: used only for cosine, where ≈.01 vs threshold ≈.016 is not borderline |
| G13 | Multiple comparisons across arms/variants | MED, for writing | ACCEPTED for now: headline claims are the two pre-named arms; the wave-5 specification curve is the systematic answer |

## 2. What wave 4 runs (one shot, no interdependencies)

```
F10  behavioural + chat template     GPU ×3   → pairs/results/behavioral/*.chat.json
F11  SAE token-level firing audit    GPU ×1   → sae/results/token_firing.layer24.json
F12  steering v2 (blocks 4-20, rel α, chat)  GPU ×2 → steering/results/*.v2early.json
F13  E3 last-on-last + positive control + surgical LEACE  CPU ×4 → transfer/results/*.last.json
F14  llama eng stride-1  +  lens with random calibration  CPU ×6
```

**Decision rules, written before the results land:**
- F11: median fired-anywhere on documents stays < 2% → F8's claim survives its
  audit, quote it; jumps ≥ 10% → reframe as "fires locally, does not propagate"
  and add a propagation experiment to wave 5.
- F12: steer flip/Δlogit separates from the random control anywhere in blocks
  4–20 → the entity time direction is causally used and cell-C steering (wave 5)
  becomes the priority; still null → report "no causal use detected under both
  the paper's recipe and the norm-matched variant".
- F10: chat flips the yes-rate toward ~.5 and consistency ≥ .8 → v1's null was
  format; accuracy is then the real behavioural number. Consistency stays low →
  the representation↔behaviour dissociation is genuine.
- F13: positive control must be ρ≥.8; if last-on-last transfer is STILL null
  with the control passing, "different axes" is sealed. Ruler-probe after LEACE
  must be ≤ 2× chance for any mediation sentence to be quotable.
- F14: if a stride-1 layer lifts Llama eng above its stride-2 number by > 1 sd,
  the "Llama is flat on English" anomaly was a sweep artifact; else it is real
  and worth a sentence (tokenizer? gloss register?).

## 3. Wave 5 — the experiments not yet run (planned, not yet coded)

Ordered by information-per-GPU-hour:

**W5.1 — E6 Esarhaddon micro-study (CPU).** The only ruler with within-ruler
year variance (176 fragments, 11 years, 681–669 BCE) — the single place where
"document time free of identity" is defined. Run inside his fragments only:
ridge/PLS year probes vs TF-IDF floor vs twins; within-ruler pairwise ordering
(needs a small `--within-ruler` extension of `pairs_data.draw_pairs`); Fiedler
seriation restricted to his cloud. Inference: permute year-groups within
Esarhaddon. Honest floor: label sd ≈ 2.5y over a 12-year window — state the
detection limit up front.

**W5.2 — E7 spectral seriation (CPU).** Unsupervised: kNN graph on fragment
activations (reuse the p9 geodesic-graph code), Fiedler vector = the cloud's
natural 1-D order; labels used ONCE post hoc (Spearman vs chronology, vs genre,
vs provenance — whichever it matches is the answer). Runs per arm per variant;
also within-Esarhaddon and after E4 erasure. This is the only experiment whose
result cannot be blamed on labels at all.

**W5.3 — E4 confounder-erasure suite (CPU).** LEACE(genre ⊕ length-bins ⊕
provenance) per train fold → does the nested-PLS gap (.336 vs .243) survive?
With the surgical controls F13 just built, plus quadratic/RBF ruler probes
(reuse P10 kernels) as the nonlinear-leak battery. Decomposition caveat baked
in: provenance predicts ruler at 55.6%.

**W5.4 — cell-C steering (GPU; only if F12 finds a live direction).** Steer at
the ruler-name tokens inside Akkadian fragments (king_token span machinery,
maxking cleaning) while asking earlier/later — the direct causal test of "the
anchor never engages". If F12 is null, this is skipped: no point injecting a
direction that does nothing even at home.

**W5.5 — specification curve (CPU, closes G13).** One figure: every headline
number across layer × cleaning × pooling × probe × protocol, medians marked.
The systematic answer to "you picked the configuration that works".

**W5.6 — E1 attention-pooling arm (CPU+GPU-light, closes the mean-pooling
assumption).** Rerun the pairwise probe with last-token AND max-pooling over
tokens for the two headline arms; if ordering improves materially, the E1
numbers were pooling-limited.

## 4. Runbook

Wave 4 now:

```bash
cd ~/projects/HUJI-THESIS--YARIN && git pull origin main && bash v_1/src/phase2/submit_wave4.sh
```

Wave 5: to be coded after wave 4 lands (W5.4's go/no-go depends on F12).
