# Experiment map — the {salient, obscure} × {high, low}-resource matrix

The paper (Gurnee & Tegmark) tests **one cell**. Everything we added is either a **new
cell**, a **new axis**, or a **fairness device**. Keeping those three apart is what makes
the story tellable.

## 1. The core matrix — what is the entity, what is the language

|                       | **High-resource language (English)** | **Low-resource language (Akkadian)** |
|-----------------------|--------------------------------------|--------------------------------------|
| **Salient entities**  | **CELL A** — the paper's cell. Famous places / figures / media / headlines. *Slides 25, 30, 31* | **CELL D — EMPTY.** No salient entities written in Akkadian. (Closest proxy: `r8` = the 8 best-attested rulers vs `r40` = the long tail.) |
| **Obscure entities**  | **CELL B** — Assyrian rulers & find-spots, *written in English* (`eng_tier0` gloss). *Slides 26, 27, 32, 33* | **CELL C** — the same entities in raw Akkadian (`akk_maximal`). *Slides 28, 29, 32, 33* |

**Why cell B is the important one we added.** It is the control that separates the two
confounded causes. Cell A → C changes *both* entity salience *and* language at once. Cell B
holds the language fixed at English and changes only the entity, so:

* A vs B  → isolates **entity obscurity** (same language, famous vs obscure entities)
* B vs C  → isolates **language resource** (same entities, English gloss vs raw Akkadian)

## 2. Axes we added *inside* a cell (variants, not new cells)

| Axis | Paper | Ours |
|---|---|---|
| **Pooling** | last token of the entity | `last` **and** `mean` over the entity/fragment |
| **Entity span** | short name / headline | whole fragment (also: king-name token only, slide 16) |
| **Probe** | ridge | ridge, **PLS (k = 1…64)**, kernel PLS, geodesic KPLS, supervision dial |
| **Model ladder** | Llama-2 7/13/70B | + Qwen3 1.7/8/32B, gpt-oss-120B, 3 encoders, TF-IDF |
| **Controls** | — | **random-init twins** of every Llama + Qwen |
| **Read-out** | R² | R², Spearman, great-circle km |

## 3. Fairness devices — NOT experiments

These exist only to make an unbalanced corpus comparable to the paper's balanced one.
They are protocol, not findings:

* `maximal` cleaning (Akkadian text normalisation)
* **balanced Monte-Carlo** (r8, cap 21, 200 draws) — removes ruler-frequency imbalance
* **by-site MC** (10 merged find-spots, cap 21) — the space analog
* **LORO / by-site hold-out** — generalisation stress, not a different question
* `r8` vs `r40` — attestation depth (doubles as a weak salience proxy inside Akkadian)

## 4. Slide → cell map

| # | Slide | Cell | Target | Pooling | Probe | Role |
|---|---|---|---|---|---|---|
| 0–3 | title / thesis / protocol / journey | — | — | — | — | narrative |
| 4 | PLS vs Ridge, all models | C | time | mean | PLS+ridge | **primary** (thesis headline) |
| 5 | signal deepens with layer | C | time | mean | PLS | layer variant |
| 6 | k = 3–5 components | C | time | mean | PLS-k | **superseded by 32** (k→64) |
| 7 | scale & finetune do nothing | C | time | mean | ridge | model-family ablation |
| 8 | tokenizer efficiency | C | — | — | — | analysis |
| 9 | translation finetune → deep repr | C | time | mean | PLS | layer variant |
| 10 | chronology entangled (embedding viz) | C | time | mean | PCA/UMAP | geometry viz |
| 11–12 | contributions / stress-test intro | — | — | — | — | narrative |
| 13 | T9 free-text generation | C | time | — | generation | non-probe |
| 14 | P2 find-spot | B+C | space | mean | PLS+ridge | **primary (km)** |
| 15 | P1 whole-text vs king-name token | C | time | mean vs **entity-token** | ridge | **pooling variant** |
| 16 | T10 prompting | C | time | — | prompt | non-probe |
| 17 | translation probe — year | B (vs C ref) | time | mean | PLS+ridge | **primary (B, time)** |
| 18 | translation probe — geo | B (vs C ref) | space | mean | PLS+ridge | **≈ duplicate of 14** |
| 19 | E5 shuffle words | C | time | mean | ridge | order ablation |
| 20 | P9 geodesic kernel PLS | C | time | mean | kernel | probe variant |
| 21 | P8 supervision dial | C | time | mean | kernel | probe variant |
| 22 | T12 ask the LLM | C | time | — | generation | non-probe |
| 23 | tier0 baseline vs controls | B | both | mean | ridge | control |
| 24 | **English G&T replication** | **A** | both | last | ridge | **primary (A)** |
| 25 | Exp1 year — English gloss | B | time | last | ridge, MC r8 | **primary (B, time, MC)** |
| 26 | Exp1 geo — English gloss | B | space | last+mean | ridge, by-site MC | **primary (B, space, R²)** |
| 27 | Exp2 year — raw Akkadian | C | time | last | ridge, MC r8 | **primary (C, time, MC)** |
| 28 | Exp2 geo — raw Akkadian | C | space | last+mean | ridge, by-site MC | **primary (C, space, R²)** |
| 29 | English layer sweep | A | both | last+mean | ridge | **primary (A, depth)** |
| 30 | English PLS k=1…64 | A | both | last+mean | PLS-k | **primary (A, dimensionality)** |
| 31 | Akkadian layer sweep | B+C | both | last+mean | ridge | **primary (B/C, depth)** |
| 32 | Akkadian PLS k=1…64 | B+C | both | last+mean | PLS-k | **primary (B/C, dimensionality)** |

*(Slide numbers are `data-index`; the on-screen counter is +1.)*

## 5. Coverage per cell

| Cell | time | space | layer | PLS-k | last | mean | random controls |
|---|---|---|---|---|---|---|---|
| **A** English / salient | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **B** English / obscure | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **C** Akkadian / obscure | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **D** Akkadian / salient | — | — | — | — | — | — | — |

**Complete except cell D**, which has no natural filler (no famous entities exist in
Akkadian outside these same royal names). The honest substitute is the `r8` vs `r40`
attestation contrast already in the data.

## 6. Redundancy to resolve

* **18 ≈ 14** — same P2 by-site km protocol; 18 only swaps the text variant and repeats
  14's Akkadian column as its reference. **Strongest candidate to cut.**
* **6 → 32** — old k = 1…5 sweep superseded by k = 1…64 across all arms.
* **14/18 (km) vs 26/28 (R²)** — same probe, different read-out. Keep one framing, or fold
  km in as a secondary column on 26/28.
* **17 vs 25** — both "does English surface the year?", but 17 is holdout+PLS+mean and 25
  is balanced-MC+ridge+last. Different protocol → keep both, 25 is the honest one.

## 7. The three-slide story the matrix supports

1. **A** — the paper reproduces on English, and our extra arms + random controls hold
   (trained ≫ random by ~0.73 R²).
2. **A → B** — swap famous entities for obscure ones, keep English: signal drops but
   survives; TF-IDF starts to lead on time.
3. **B → C** — swap English for Akkadian, keep the entities: trained collapses to its
   random twin. **The world model is a property of the language the model was trained on,
   not of scale.**
