# Onboarding prompt — Akkadian dating stress-tests

Paste the block below to a fresh / advisor agent so it reads the right docs and knows
the current state. (Update the LIVE STATE section as jobs change.)

---

You are taking over the "Akkadian dating stress-test" thesis project. Get up to speed by
reading these files IN THIS ORDER before doing anything:

1. `v_1/src/stress_tests/HANDOFF.md` — session handoff: the thesis question,
   data/protocol, code map, cluster+git conventions, and running NEXT-STEPS.
2. `v_1/src/stress_tests/ADVISOR_WALKTHROUGH.md` — the narrative tour: each experiment,
   the published claim it mirrors, the config, the result table, and the "so what?"
   cross-cutting patterns. Best single read for an advisor.
3. `v_1/src/stress_tests/results/RESULTS_stress_tests_explained.md` — the full briefing:
   every experiment (T9, P2, P1 a/b/c, P3, P7, T10, EDA), the paper it mirrors, exact
   config, models run, WHERE each result file lives, headline numbers, conclusions,
   caveats, a config matrix (§11), and the one-paragraph conclusion (§12). Primary doc.
3. Result tables:
   - `v_1/src/stress_tests/results/RESULTS_stress_tests.md` (P1 balanced-MC + P2)
   - `v_1/src/stress_tests/p1_gurnee_tegmark/results/maxking/RESULTS_maxking.md`
   - `v_1/src/stress_tests/results/csv/*.csv` (one CSV per experiment)
   - `v_1/src/stress_tests/results/eda/*.png` (figures)

Conventions: feature branch `claude/handoff-stress-test-docs-2bbunu`; the CLUSTER runs from
`main` (land code by fast-forwarding the feature branch onto main on the login node). The
user pastes every sbatch; the agent NEVER SSHes. Only result JSON/CSV/MD + balanced-subset
draws are committed (`*.npz`/logs gitignored). The stop-hook "Unverified commits" warning
is cosmetic — ignore it, do NOT rewrite the cluster-authored history on main.

One-line finding: models KNOW the dates (T9) and the pipeline is valid (P2 geography
decodes), but the date is NOT recoverable structure over text — every positive is token
identity (mostly the king name) and a RANDOM-init model matches/beats trained on the
decisive sites, with no scale effect up to gpt-oss-120B. Only cuneiform-domain Thalesian
adds a modest real increment.

LIVE STATE (2026-07-02): **the full ladder is complete.** All experiments have run for
all intended models. The two stragglers finished successfully:
- qwen3_32b T10 balanced-MC — done (12h rerun; mean stays flat ~0.38–0.42 across pv0–3).
- gpt-oss-120B maximal-with-kings — done (re-extracted on gpu:8; mean ruler-F1 = 0.750
  ≈ random 0.741, king sites ~0.98 → confirms token-identity, no scale effect).
No jobs are pending. Remaining optional TODOs: extend MLM to P2/P3/P7/maxking (needs its
own extractor); ruler_spellings.csv expert review to raise king coverage.
