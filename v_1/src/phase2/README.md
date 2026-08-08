# Phase 2 — status board

Program: `RESEARCH_PROGRAM.md` (hypotheses) → `DECIDED_EXPERIMENTS.md` (the
decided list E1–E8, post-verification). One folder per experiment family; every
folder's README carries its own progress detail. This page is the wave tracker.

## Waves

**Wave 1 — pairs (E1) + its inference (E8)** — `pairs/`
- [x] F1 probe, 13 arms × 2 variants (job 22587) → `pairs/RESULTS.md`
- [x] F2 behavioural Yes/No (22588) — degenerate No-bias, documented
- [x] F3 robustness m=100 (22589) — picture unchanged
- [x] F4 done (both tasks). akk: floor p=.013, olmo p=.007, contrasts null — orderable by surface features, LLM adds nothing. eng: THE DISSOCIATION — trained models significant (olmo & qwen p=.0066) while the floor is NOT (p=.11); vs-twin contrasts marginal (p=.08–.10, n=40 rulers). Full tables in pairs/RESULTS.md §4.

**Wave 2 — transfer (E3)** — `transfer/`
- [x] F5 done — TRANSFER FAILS; name-time and document-time are orthogonal axes (cos ≈ .01, chance level). See transfer/README.md

**Wave 3 — traces (E4.4), SAE (E5), steering (E2)** — `traces/`, `sae/`, `steering/`
submitted as one block by `submit_all.sh` (F8 chained afterok F7):
- [x] F6 done, all 3 models — **cell-A year direction is genuinely temporal**: its early end reads BC/BCE/ancient/Athen (and in Qwen even 公元前 'BCE', 战国 'Warring States') in every model; OLMo's late end contains literal year fragments (187/188). **E1's pairwise document direction lenses to junk in all models** — no temporal vocab, no royal names: consistent with E3's orthogonality, the document axis is not a vocabulary-aligned time axis.
- [ ] F7 SAE FVU gate: Qwen-Scope vs our acts, incl. Akkadian OOD (1 CPU task)
- [ ] F8 SAE feature hunt at the gated layer (afterok F7)
- [x] F9 done (3 tasks) — **null at the tested band**: flip rates ≈ random control, logit shifts ≈ 0 at α≤24, blocks 21–32. Caveat before concluding 'not causal': our ridge direction lives at LATE layers (26/29) while the NAACL effects concentrated in the FIRST half of the stack, and α may be small vs late-layer residual norms. A follow-up sweep (early-mid blocks, α scaled to residual norm) is the fair test; as run, no causal use detected.

Deferred by design: cell-C steering (needs king-token span integration),
E6 Esarhaddon micro-study, E7 seriation, E4 confounder-erasure suite.

## Submitting

```bash
bash v_1/src/phase2/submit_all.sh      # wave 3, with dependencies
```

Earlier waves' jobs live in `pairs/sbatch/` and `transfer/sbatch/` and can be
resubmitted individually; every job syncs main first and commits its results.
