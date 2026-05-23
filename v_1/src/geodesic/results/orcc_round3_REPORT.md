# Round 3 — Geodesic / Manifold Readout of Akkadian Temporal Structure

**Status:** Phase A NULL → one-page negative result.  Pivot to Phase E1 (Qwen3 scale sweep).

---

## Phase 0 — Activation Inventory (job 8529, 2026-05-24)

| Method | Cleaning | Pool | Layers | Shape | Path |
|---|---|---|---|---|---|
| qwen | tier0 | mean | 29 | (1202, 3584) | orcc_round1/activations/ |
| qwen | tier0 | last | 29 | (1202, 3584) | orcc_round1/activations/ |
| qwen | maximal | mean | 29 | (1202, 3584) | orcc_round1/activations/ |
| qwen | maximal | last | 29 | (1202, 3584) | orcc_round1/activations/ |
| mlm | tier0 | mean | — | MISSING | not found at any candidate path |
| thalesian_akk300m | all 4 | all | 9 | (1202, 512) | orcc__embed/activations/ |
| thalesian_cunei400m | all 4 | all | 13 | (1202, 768) | orcc__embed/activations/ |
| random | all 4 | all | 29 | (1202, 3584) | orcc_round1/activations/ |

**Parquet:** 1,193 / 1,202 year-labeled. **Phase A gate: PASS.**

**Random-Qwen disposition:** activations found at `orcc_round1/activations/` (fallback path). No re-extraction needed. Round 2 MC gap was a script path issue, not missing activations.

**MLM disposition:** not found; excluded from Phase B (which is moot given Phase A null).

---

## Phase A — Single-layer POC (job 8530, 2026-05-24)

**Configuration:** Thalesian cuneiBase-400m, layer 12, tier0/mean (Round 2 best).

| Metric | PLS (refit full-set) | Isomap (A1) | Earliest-bin (A2) |
|---|---|---|---|
| Spearman | 0.633 | **0.035** | 0.002 |
| Pairwise-order acc (±100yr) | 0.859 | **0.547** | 0.440 |
| Neighbor purity (k=10, ±100yr) | — | **0.796** (+13.4σ) | 0.778 (+6.8σ) |

**kNN graph connected at k=3** (very sparse graph).

### Gate evaluation

- Best geodesic Spearman = 0.035; delta vs PLS = **−0.432** (≤ −0.05 ✓)
- Best pairwise-order accuracy = **0.547** < 0.60 ✓
- **Verdict: NULL** → Round 3 geodesic path closed.

### Interpretation

Phase A matches the pre-committed prediction: *"PLS exploited supervised projection in a way unsupervised geometry cannot."* Thalesian's temporal signal is real (PLS Spearman 0.467 CV / 0.633 full-set) but is not organized as a globally monotone 1D manifold.

**Notable secondary finding:** neighbor purity = 0.796 at **+13.4σ above null**. Nearby texts in representation space systematically tend to be close in calendar time, confirming local temporal clustering. This is not a geodesic ordering failure — it is a finding that the temporal manifold is locally coherent but multi-dimensional: PLS finds the signal via supervised projection; Isomap's 1D readout cannot resolve it unsupervised.

This result strengthens, not weakens, the Round 2 headline: Thalesian's representations encode temporal information, but the encoding is not a simple 1D curve.

**Suggested thesis framing:** *"Thalesian's encoder embeds time as a locally-coherent but globally non-monotone structure. Linear probing (PLS) efficiently extracts the signal via supervised projection; unsupervised geodesic readout cannot. This indicates the temporal dimension is one of several competing axes in the representation space, consistent with the model having also absorbed ruler, genre, and archive information."*

### Phases B–D

Not run (Phase A gate null). No `geodesic_best_layers.json` or centroid plots produced.

---

## Pivot — Phase E1: Qwen3 Scale Sweep

Per the pre-committed plan: Round 3 pivot to Round 2 Phase 2 (scale) unchanged.

**Qwen-Scope SAE coverage (as of paper 2026-04-30):**

| Model | Layers | d_model | SAE width | Sizes |
|---|---|---|---|---|
| Qwen3-1.7B-Base | 1–28 | 2048 | 32K | L0=50,100 |
| Qwen3-8B-Base | 1–36 | 4096 | 64K | L0=50,100 |
| Qwen3.5-2B-Base | 1–24 | 2048 | 32K | L0=50,100 |
| Qwen3.5-9B-Base | 1–32 | 4096 | 64K | L0=50,100 |

**Selected for Phase E1:** Qwen3-1.7B and Qwen3-8B (dense, both have full layer SAE coverage). Qwen3-8B is the scale peer of Qwen2.5-7B from Round 2; Qwen3-1.7B enables a 1.7B vs 8B scale comparison.

Note: the Round 2/3 plan specified "4B, 14B, 32B" — those sizes do not have Qwen-Scope SAEs. Qwen3-8B is the closest available dense model.

**Next steps:**
1. Extract Qwen3-1.7B and Qwen3-8B activations on ORCC (4 combos each = 8 parallel GPU sbatch).
2. Run PLS + CLS probes on each layer (same pipeline as Round 2).
3. If Qwen3-8B shows temporal signal (PLS Spearman > 0.3), proceed to SAE encoding + attribution.
4. Report Qwen3 vs Qwen2.5-7B vs Thalesian comparison.

Extraction jobs submitted: see below.
