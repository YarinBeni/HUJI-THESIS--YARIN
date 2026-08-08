# Phase 2 — status board

Program: `RESEARCH_PROGRAM.md` (hypotheses) → `DECIDED_EXPERIMENTS.md` (the
decided list E1–E8, post-verification). One folder per experiment family; every
folder's README carries its own progress detail. This page is the wave tracker.

## Waves

**Wave 1 — pairs (E1) + its inference (E8)** — `pairs/`
- [x] F1 probe, 13 arms × 2 variants (job 22587) → `pairs/RESULTS.md`
- [x] F2 behavioural Yes/No (22588) — degenerate No-bias, documented
- [x] F3 robustness m=100 (22589) — picture unchanged
- [x] F4 task 1 (akk): permutation says the ordering signal is REAL (floor p=.013, olmo p=.007 vs shuffled-chronology null ≈ .498) while the contrasts confirm NO separation (olmo−floor p=.36, olmo−twin p=.97) — orderable by surface features, LLM adds nothing. Task 0 (eng) still running

**Wave 2 — transfer (E3)** — `transfer/`
- [x] F5 done — TRANSFER FAILS; name-time and document-time are orthogonal axes (cos ≈ .01, chance level). See transfer/README.md

**Wave 3 — traces (E4.4), SAE (E5), steering (E2)** — `traces/`, `sae/`, `steering/`
submitted as one block by `submit_all.sh` (F8 chained afterok F7):
- [ ] F6 logit-lens of the year directions (3 CPU tasks)
- [ ] F7 SAE FVU gate: Qwen-Scope vs our acts, incl. Akkadian OOD (1 CPU task)
- [ ] F8 SAE feature hunt at the gated layer (afterok F7)
- [ ] F9 steering, NAACL recipe: qwen3_8b cells A+B, olmo2_7b cell A (3 GPU tasks)

Deferred by design: cell-C steering (needs king-token span integration),
E6 Esarhaddon micro-study, E7 seriation, E4 confounder-erasure suite.

## Submitting

```bash
bash v_1/src/phase2/submit_all.sh      # wave 3, with dependencies
```

Earlier waves' jobs live in `pairs/sbatch/` and `transfer/sbatch/` and can be
resubmitted individually; every job syncs main first and commits its results.
