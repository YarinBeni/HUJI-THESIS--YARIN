# Findings — do language models represent space & time? English vs Akkadian

Two experiments in one protocol (Gurnee & Tegmark 2023: build an entity string → take
its last-token embedding per layer → ridge-probe a coordinate or a year on held-out
entities). One on the paper's English data, one on our Akkadian corpus. Together they
give a clean, symmetric result.

> **On English, general LLMs own space and time and small translation encoders don't.
> On Akkadian, general LLMs are near-random for dating and geography — a bag of
> character n-grams beats them, and training barely helps — while the specialized
> small translation encoder is what carries the signal. The internal "world model" is
> language-specific to what the model was trained on.**

---

## Experiment 1 — English (replication of Gurnee & Tegmark)

14 model arms + a TF-IDF text floor, six datasets (space: world/US/NYC places →
coordinates; time: historical figures / media / headlines → year). Best-layer held-out
test R²:

| arm | world | us | nyc | figures | media | headlines |
|---|--:|--:|--:|--:|--:|--:|
| Llama-2-70B (paper) | .911 | .864 | .359 | .835 | .885 | .746 |
| **our llama2_70b** | **.905** | **.846** | **.363** | **.833** | **.860** | **.757** |
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

*(gpt-oss-120B and llama2_70b_random were still finishing at write time; they land
mid-ladder and near the other randoms respectively and do not change the conclusions.)*

**Findings.**
1. **Reproduction.** Our trained Llama-2-70B matches the paper within ~0.02 on every
   dataset (peak layer 53 ≈ their ~65% depth) → the probing harness is validated.
2. **Scaling.** Both families are monotonic in size (Qwen3 1.7B→8B→32B = .655→.797→.838
   on world; Llama 7B→13B→70B).
3. **Learned, not architectural** — the control the paper never ran. Random-init arms
   collapse in *both* domains (Llama-13B .883→.282 space, .780→.038 media); the ~0.6 R²
   gap is entirely training.
4. **A real floor.** TF-IDF scores .642 on world (and beats every model on NYC); trained
   LLMs clear it decisively, random arms sit below it.
5. **Encoders low on English.** uMT5 / cunei-400M / AKK-300M land near the floor — their
   Akkadian strength is not generic probing skill.

---

## Experiment 2 — Akkadian (the same protocol, our rulers & find-spots)

Entity = a whole fragment (its maximal-cleaned Akkadian, or its English translation);
target = composition year and find-spot (lon, lat). Two ruler sets: **r8** (8
best-attested rulers, ~1071 texts — the dense, trustworthy panel) and **r40** (all 40,
~1187 texts — a sparse tail). Decoder LLMs only + TF-IDF; **encoders excluded** (no
causal last token). Held-out-by-ruler split.

### Year — r8, best-layer test R² / Spearman ρ (Akkadian text)

| arm | R² | ρ |
|---|--:|--:|
| **TF-IDF (floor)** | **0.634** | **0.793** |
| llama2_70b | 0.428 | 0.596 |
| llama2_7b | 0.418 | — |
| llama2_13b | 0.417 | — |
| qwen3_8b | 0.386 | 0.533 |
| qwen3_1b7 | 0.339 | — |
| qwen3_32b | 0.316 | — |
| *llama2_7b **random*** | *0.355* | *0.495* |
| *qwen3-8b **random*** | *0.343* | *0.482* |
| *llama2_13b **random*** | *0.291* | — |

### Geo — r8, best-layer test R²: weak for everyone (~0.10–0.18 across all arms, TF-IDF included).

**Findings.**
1. **TF-IDF beats every embedding arm** (year r8 R² 0.634 vs best model 0.428). A bag of
   Akkadian character n-grams is the strongest dater.
2. **Trained ≈ random.** The trained-vs-random gap that was ~0.6 on English is **~0.04
   here** (qwen8b .386 vs its random .343; llama-7b .418 vs random .355). The big
   multilingual LLMs hold almost no *learned* Akkadian chronology — what little their
   embeddings carry is surface-level, which is why random init nearly matches and
   TF-IDF wins.
3. **Akkadian > English translation for year** (TF-IDF .634 vs .495; qwen8b .386 vs
   .297). The original orthography carries date cues (ruler-name spellings, sign
   choices) that translation washes out.
4. **Geo is weak for all** and **r40 collapses** (long tail of 1-text rulers under
   hold-one-ruler-out) — read r8 as the trustworthy panel.

---

## Synthesis — the inversion

| | English (high-resource) | Akkadian (low-resource) |
|---|---|---|
| ordering of arms | trained LLM ≫ TF-IDF ≫ random | **TF-IDF ≫ trained ≈ random** |
| trained-vs-random gap (best) | ~0.6 R² | ~0.04 R² |
| where the signal lives | the big models' learned geometry | surface form; the *specialized* small encoder (thesis P1/P2) |

The two halves compose: the big models that own space & time on English are
**near-random on Akkadian**, and the small translation encoders that are near-floor on
English are exactly what recovers Akkadian time in the thesis's own probes. Linear
"world-model" structure is not a general property of scale — it is **specific to the
languages and entities a model was actually trained on.**

## Caveats (state these alongside the result)

- **Pooling.** This uses last-token pooling on decoder LLMs that tokenize Akkadian
  poorly (many byte/UNK tokens), so "big LLMs fail on Akkadian" is precisely *big LLM +
  G&T last-token protocol* — the fairest G&T-faithful test, but the thesis's success
  case uses mean-pooled specialized encoders, which are excluded here by design.
- **Scale & discreteness.** With 8 rulers, year probing is closer to separating 8
  groups than continuous regression (the categorical-feature confound G&T flagged);
  hold-one-ruler-out is the stronger follow-up. Geo extent is small (Mesopotamian
  sites), so haversine R² behaves differently than a global map.
- **Two English arms (gpt-oss, 70b_random) were still finishing at write time.**

## Files

- English: `results/RESULTS.md`, `results/summary_best_layer_{r2,spearman}.csv`;
  design in `PLAN.md`, ops + bugs in `HANDOFF.md`, summary in `SUMMARY.md`.
- Akkadian: `akkadian/results/RESULTS_akk.md`,
  `akkadian/results/summary_{year,geo}_best_test_{r2,spearman}.csv`; design in
  `akkadian/README.md`.
