# SAE2 — the Neuronpedia-labeled dictionary (jobs F22 + F23)

Replication + interpretation + intervention on a SECOND, independently trained
SAE (Adam Karvonen's Qwen3-8B batch-TopK 65k release, Neuronpedia-labeled),
per the handoff plan. Feature indices do not transfer between dictionaries —
that is the point: replicating entity-gating in a second dictionary upgrades it
from "one SAE's view" to a robust finding, and this one comes with autointerp
labels, dashboards and a steering UI for free.

| file | step |
|---|---|
| `karvonen.py` | step 0 loader: release discovery (fails loudly if not found), tolerant key mapping, batch-TopK threshold inference with topk fallback |
| `run_pipeline.py` | steps 0–2 + 4: layer + empirical offset, FVU gate on FOUR populations (cell B now included), feature hunt with cross-population firing, token-level fired-anywhere + position thirds; explicit replicated/not verdicts vs F8/F11 |
| `fetch_labels.py` | step 3: Neuronpedia autointerp labels + taxonomy classification (temporal / entity-identity / numeric-year / historical-domain / style / other) |
| `feature_steer.py` | step 5: amplify/suppress + ablate with firing-rate-matched random-feature controls (Feldman non-surgicality), and THE BRIDGE — clamp temporal features on mid-text in English glosses, ask if the signal reaches the last token |
| `sbatch/F22_*.sbatch` | pipeline + labels (GPU ×1) |
| `sbatch/F23_*.sbatch` | interventions (GPU ×1, afterok F22) |

Pre-registered rules carried over: eng fired-anywhere ≥10% and akk <2%
replicate F11; gate FVU ≤ .35 on cell A or features are not interpreted.
Inherited caution, verbatim: direction-steering (F12) was null — a null in
step 5 WITH the control is a publishable mechanistic result (transient firing,
not causally recoverable at readout); report against the control, don't retry
until significant.

## Progress
- [x] Code written; audited (offset bug caught in review: step-4/steering now use the EMPIRICAL step-0 offset, not a hardcoded +1)
- [x] F22 run 1 (23753): repo found, arbitrary 16k config grabbed → gate correctly failed (.82). Fixed: step 0 FVU-scans every (layer, file, offset).
- [x] F22 run 2 (23760): **scan verdict — only layer 9 is usable.** All 8 layer-9 configs pass brilliantly (FVU .017–.023) while EVERY layer-18/27 config fails the gate (.50–127). Gate ×4: cellA .0171 / eng .011 / akk .0077 / **cellB 1.39 (anomaly — small-n low-variance population, flagged)**. Hunt: 2,615 candidates, top |ρ(year)|=.44, cos(dec,ridge)≤.10 (distributed-direction claim replicates). Token firing: **eng .853 fired-anywhere → mid-text-firing REPLICATES; akk .441 → non-engagement does NOT replicate at layer 9** — early-layer features do engage on Akkadian (1.3% of tokens), so F11's non-engagement is a deep-layer phenomenon, not universal. Labels: all 404 — TWO root causes: the picked instrument was the 16k trainer (indices outside Neuronpedia's 65k space) and the guessed source id was unverified.
- [x] Fix committed: step 0 now prefers the 65k (labeled) width among near-ties (FVU tol .02 → picks trainer_2, .0179); `fetch_labels --source auto` probes a (model, source) grid on Neuronpedia and records what answered; trainer config.json recorded for source matching.
- [ ] F23 (23761) running on the 16k pick — interventions are still valid science (labels not needed for causality); rerun of F22+F23 on the labeled 65k queued behind it.
- [ ] RESULTS.md: FVU table ×4, labeled feature table, decomposition summary, intervention curves, replicated/not verdict per claim
