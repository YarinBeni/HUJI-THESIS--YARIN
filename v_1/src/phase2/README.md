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
- [x] F7 done — **the gate passes wide open at layers 16–28** (FVU .08–.20; layer 8 fails at .82). Crucially Akkadian FVU (.11–.19) ≈ English: the representations are NOT broken by the script — **H-OOD killed at the reconstruction level**. Empirical layer offset = 1 (our file L+1 = SAE block L).
- [x] F8 done (layer 24) — **the collapse reaches the feature basis**: feature 38678 fires on 62% of cell-A entities with ρ=.57 vs death year, but the top-50 year features have median firing 11.7% on entities vs 0.08% on ENGLISH glosses and ~0 on Akkadian (31/50 exactly zero). The time features are gated on entity-mention contexts — they never engage on document text even in English. cos(W_dec, ridge) ≤ .12: the ridge direction is distributed over many features.
- [x] F9 done (3 tasks) — **null at the tested band**: flip rates ≈ random control, logit shifts ≈ 0 at α≤24, blocks 21–32. Caveat before concluding 'not causal': our ridge direction lives at LATE layers (26/29) while the NAACL effects concentrated in the FIRST half of the stack, and α may be small vs late-layer residual norms. A follow-up sweep (early-mid blocks, α scaled to residual norm) is the fair test; as run, no causal use detected.

**Wave 4 — the gap-fix wave** (`submit_wave4.sh`, jobs 22908–22912, all landed;
decision rules pre-registered in GAPS_AND_WAVE5.md §2):
- [x] F13 e3 last-on-last: **positive control PASSES** (same code path reproduces cell-A ρ=.87–.89, twin .57) and transfer is STILL null (ρ −.14..−.01, pairwise ≈ chance) → **"different axes" is sealed**. LEACE surgical check now verified (ruler-probe .41–.59 → .07–.14 ≈ chance) and fp64 fixed the qwen numeric anomaly (rel-change .82 → .46/.09).
- [x] F11 token firing: **THE ONE STORY CHANGE.** Median fired-anywhere: entities 17.9%, **English gloss docs 14.9%** (≥10% rule → reframe), Akkadian 1.0% (<2% rule → survives). New two-part mechanism: on English documents the year features DO fire mid-text but the signal **does not propagate** to the document-level readout; on Akkadian they **never engage at all**.
- [x] F12 steering v2 (blocks 4–20, rel-α, chat): **still null** — flips/Δlogit ≈ random control everywhere. Per the rule: "no causal use detected under both the paper's recipe and the norm-matched variant"; **wave-5 cell-C steering is skipped**.
- [x] F10 behavioral+chat: yes-rates moved (0 → .10–.76) but order-consistency stays ≤ .43 and macro ≤ chance → **the representation↔behaviour dissociation is genuine**, tested under both formats.
- [x] F14: stride-1 leaves llama eng ≈ unchanged (7B .584→.584, 13B .586→.606 [+0.5 sd], 70B best-layer noise) → **the "Llama flat on English" anomaly is real**, not a sweep artifact. Lens random-controls lens to junk — so cell-A's temporal tokens are meaningful, and the honest phrasing for the pairwise direction is "indistinguishable from a random direction's projection".

**Wave 5 — the remaining decided experiments** (`submit_wave5.sh`; audited
before submission — 6 bugs found and fixed in review+smoke, see the wave-5
section of GAPS_AND_WAVE5.md):
- [x] F15 done + rerun with length controls — **VERDICT: STRUCTURAL.** The length-only baseline (ρ=.355) equals or exceeds every arm's raw probe (.28–.42); partialling length out collapses everything to .03–.22 with the twin (.11/.19) indistinguishable from the trained arms. The within-Esarhaddon 'identity-free chronology' is length encoding, full stop — no residual learned-chronology signal survives the control.
- [x] F16 done — the cloud's natural 1-D order tracks LENGTH (ρ up to .94) and provenance, not year: third independent confirmation that the dominant document axis is stylistic. Within-Esarhaddon year matches co-occur with huge length correlations.
- [x] F17 done — **much of the eng document-time signal rides on provenance+length**: after LEACE, olmo eng pairwise .623→.558 and qwen .628→.582 vs the erased floor .533 (residual gaps +.025/+.049, sharply reduced); grouped ridge goes ≤0. Provenance probes .58–.66→.18–.27 (erasure verified). Genre constant — not a confounder at all.
- [x] F18 done — **the F11 mid-text firings are temporal NOISE, not recoverable chronology**: max-pooled year-features score .562 on eng, BELOW the .586 floor (rule: no propagation-failure claim; the entity time features do not compute document dates anywhere). On akk they merely match the surface floor (.656 ≈ .658).
- [x] F19 done — **the specification curve catches a real sensitivity**: at site=last on eng the RANDOM TWIN reaches .623, erasing the trained-vs-twin gap that exists at mean (.615 vs .553). The eng "trained models beat twins" claim is pooling-dependent and must be reported as such.

Cell-C steering SKIPPED per F12's pre-registered rule.

**Wave 6 — the chat-mode audit** (user's question: were chat models probed the
way they were trained to be addressed?):
- [x] Extraction-chain audit: padding/last-token/mean-mask logic VERIFIED correct in wm_lib, extract_akk and stress_tests extract_lib; T11/T12 generation experiments already used apply_chat_template + enable_thinking=False. Base-model arms (Llama-2, OLMo-2, twins, encoders) are correctly probed on bare text — that IS their training-time operation.
- [ ] F21 lens spectroscopy (E4.4b, user-proposed): whole-vocabulary rank-decile composition with per-bucket random-direction nulls + the year-token order test rho(year value, l_t) + cosine de-confounding of loud unembedding rows. Selftest passes (planted year axis: rho=1.00, year-bucket z=17). Decision rules: cellA monotone gradient + ordered year tokens → "semantic axis" upgrades to "calibrated axis"; any doc-direction bucket z-significant → first positive evidence it means something; flat → "indistinguishable from random" now spectrum-wide, with the one-directional-instrument caveat attached verbatim.
- [x] F15 rerun landed (23722) — verdict folded into the wave-5 F15 line: STRUCTURAL.
- [x] F20 done — **THE DECK STANDS.** qwen3_8b_chat (template + dating question): eng mean .593 / last .605 vs bare .636; akk mean .642 vs bare .649 — chat-wrapping is the same or slightly WORSE. Probing chat models on bare text did not understate them; the deck's probing tables need no flags.

**Wave 7 — SAE2: the labeled dictionary** (`sae2/`, user's handoff plan; F15-name in the plan re-numbered F22/F23):
- [~] F22 first run (23753): step 0 FOUND the release (adamkarvonen/qwen3-8b-saes, layers 9/18/27, 38 files) but the FVU gate CORRECTLY failed (.82 on cell A) — discovery had grabbed an arbitrary 16k config among several per layer. Fixed: step 0 now FVU-scans every (layer, file, offset) candidate and the gate itself picks the instrument. **Resubmit F22 (then F23 afterok).**
- [ ] F23 interventions (afterok F22): amplify/ablate with rate-matched random controls + THE BRIDGE (clamp temporal features mid-text on eng — does the signal reach the last token?). F12-null caution attached: null-with-control is the publishable outcome.

## Submitting

```bash
bash v_1/src/phase2/submit_all.sh      # wave 3, with dependencies
```

Earlier waves' jobs live in `pairs/sbatch/` and `transfer/sbatch/` and can be
resubmitted individually; every job syncs main first and commits its results.
