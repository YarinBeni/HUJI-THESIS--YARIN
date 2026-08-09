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
- [ ] F22 submitted (step 0 is blocking: if the release repo isn't found under the candidate names, the job fails with instructions — check Neuronpedia for the real repo and add it to `karvonen.py CANDIDATE_REPOS`)
- [ ] F23 submitted (afterok F22)
- [ ] RESULTS.md: FVU table ×4, labeled feature table, decomposition summary, intervention curves, replicated/not verdict per claim
