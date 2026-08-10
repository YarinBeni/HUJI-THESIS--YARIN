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
- [x] F21 lens spectroscopy done (23942, after 3 crash-fix rounds: superscript digits, None vocab ids) — **the spectrum seals both claims.** Cell-A direction: bucket-1 (early end) is significantly enriched in temporal_ancient vocabulary in ALL THREE models (z = 3.8/6.0/4.8 raw; 3.9/6.8/5.2 under the cos de-confounding — Bonferroni threshold 3.35) → the entity year axis is semantically temporal across the whole vocabulary, not just top-k. Pairwise doc direction: olmo & qwen (the models that carry the eng doc signal) show NOTHING above threshold in either variant → **"indistinguishable from random" is now spectrum-wide**; llama alone shows one marginal cell (bucket-9 temporal_modern, cos z=3.5) — per the pre-registered rule this is "first positive evidence", flagged as likely multiplicity (1 cell of 540 at z=3.5, and in the model whose eng behavioral signal is absent). The year-token order test is UNRUNNABLE: all three tokenizers split 4-digit numbers (<10 whole-year tokens) — "calibrated axis" upgrade untestable by this instrument; the one-directional-instrument caveat attaches verbatim.
- [x] F15 rerun landed (23722) — verdict folded into the wave-5 F15 line: STRUCTURAL.
- [x] F20 done — **THE DECK STANDS.** qwen3_8b_chat (template + dating question): eng mean .593 / last .605 vs bare .636; akk mean .642 vs bare .649 — chat-wrapping is the same or slightly WORSE. Probing chat models on bare text did not understate them; the deck's probing tables need no flags.

**Wave 7 — SAE2: the labeled dictionary** (`sae2/`, user's handoff plan; F15-name in the plan re-numbered F22/F23):
- [~] F22 run 1 (23753): gate correctly failed on an arbitrary 16k config → step 0 now FVU-scans all 38×2 candidates. Run 2 (23760) landed — **only layer 9 usable** (all configs .017–.023; layers 18/27 fail at .50–127). Gate ×4: cellA .017 / eng .011 / akk .008 / cellB 1.39 (small-n variance artifact, flagged). Hunt: top |ρ|=.44, cos(dec,ridge)≤.10 → **F8's distributed-direction claim replicates**. Run 3 (23898, **the pre-specified labeled 65k, k=80 — PRIMARY**): 776 candidates, top |ρ|=.42, cos≤.08; token firing eng .355 / akk .020 → **BOTH F11 verdicts REPLICATE** (eng mid-text firing ≥10%; akk non-engagement <2%, at the boundary). The 16k run's akk .441 stands as a dictionary-sensitivity note: non-engagement is defined relative to a sufficiently sparse feature basis. Labels STILL blocked — the full (model × source) probe grid 404'd; needs a one-minute browser lookup of the exact Neuronpedia source id. Full tables in sae2/RESULTS.md.
- [x] Labels: TERMINAL — Neuronpedia hosts ONLY layer 18 of the release (API-confirmed: L18 answers JSON, L9 404s) and L18 fails our gate (.56). No autointerp labels for the usable instrument.
- [x] F24 (23937) — decoder-row lens self-labels: layer-9 year-ρ features read as WHO-features (German-surname morphology, Chinese surnames, Germany, classical-Chinese dynastic register 帝王/天下/子孙, honorifics), no calendar vocab. **DOWNGRADED to suggestive** per the user's (correct) objection: logit lens at layer 9/36 is unreliable. Primary interpretation → F25.
- [ ] F25 — feature interpretation without the lens (standard practice): max-activating contexts across the three populations + Golden-Gate-style clamped generation (chat template, α·act95, what the model starts talking about is the label). GPU, `sbatch/F25_feature_interp.sbatch`.
- [x] F23 done (23948, after the empty-pool + bridge-padding fixes) — **three verdicts:** (1) AMPLIFY is the program's first positive causal result: pushing an onomastic feature moves the frozen year read-out monotonically in the sign of its ρ (Δ up to ±0.9 sd) while rate-matched controls stay flat — the name-culture features causally feed the entity time axis (direction-steering F9/F12 was null; feature-steering works). (2) ABLATE single features: nothing — the code is distributed, no single feature load-bearing. (3) THE BRIDGE: **null-with-control, the pre-registered publishable outcome** — clamping the features mid-text on glosses leaves last-token firing at exactly 0 and probe shifts inside the control band: mid-text firing stays local; no propagation channel exists even when the signal is forced in. F11/F18's "firing without chronology" is now causal. Tables in sae2/RESULTS.md.

**Wave 7 complete.**

**Wave 8 — the last two planned experiments** (user request: run everything
still on the table; both audited pre-submission — block-28 DIR arm removed
as causally unreachable, per-fragment years in the pair judge, y
standardized for the MLP, spans smoke-tested at 547/1187):
- [ ] F26 anchor ignition (`steering/ignite_anchor.py`, GPU): the original
  "cell-C steering", re-justified by F23's positive amplify. FEAT arm
  (top-5 onomastic features at the ruler-NAME token span of eng glosses /
  all-but-last of akk, vs rate-matched controls) + DIR arm (rel-α ridge
  direction at blocks 8/16/24 vs random direction). Read-out: frozen cell-A
  probe at last token. Pre-registered rules in the docstring;
  null-with-control publishable.
- [ ] F27 nonlinear probes (`erasure/e4_nonlinear.py`, CPU array ×8):
  kernel-RBF (median-heuristic γ, inner-CV) + MLP(256,128) under
  GroupKFold-by-ruler, pairs judged only with BOTH rulers held out;
  arms olmo/qwen/twin/tfidf × eng/akk. Retires the "nonlinear code would
  not be caught" caveat, one way or the other.
- [x] F25 feature interp done (23940) — **the layer-9 year-ρ features are ONOMASTIC detectors**, confirmed by the reliable instrument: max-activating contexts show German surnames (44713: Kienzle/Rusch/Riedel — and its clamped generation drifts into German, a clean Golden-Gate corroboration), Chinese name pieces (56768/26073), Chinese imperial names (9763), Byzantine/Korean/Liao monarchs (57332), "X of PLACE" nobility (17433), German noble houses (2343), 19th-c famous names (50848), and an ancient-genealogy feature firing on the GLOSSES themselves (53704: "son Cambyses I, father of", 17/20 examples from eng_tier0). The year correlation rides on naming culture ↔ era — the entity-mediated-time mechanism made concrete at the feature level. (α=10 clamps mostly degenerate generation — over-clamping; the labels rest on the contexts.) Full table in sae2/RESULTS.md.

## Submitting

```bash
bash v_1/src/phase2/submit_all.sh      # wave 3, with dependencies
```

Earlier waves' jobs live in `pairs/sbatch/` and `transfer/sbatch/` and can be
resubmitted individually; every job syncs main first and commits its results.
