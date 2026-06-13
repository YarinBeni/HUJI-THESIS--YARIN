# Round 4 — paper review (plain language, with links)

Grouped by what they're for. Each entry: **the idea** → **how we use it** → **which pillar** → link.
"Read first" = the 5 most load-bearing for understanding the round.

---

## A. Ancient-text AI — the domain and the systems we're imitating

### 1. Translating Akkadian to English with NMT (Gutherz et al., PNAS Nexus 2023) ★ read first
**Idea:** A neural translation model takes Akkadian (cuneiform glyphs *or* Latin transliteration)
straight to English, reaching BLEU ≈ 36–37 — surprisingly good for such a low-resource language.
**How we use it:** This is what makes Pillar 1's translation arm (1c) and the English-content
diagnostic (1d) realistic — there *is* usable Akkadian→English signal. It's also the source we'd
ask for parallel data. **Caveat:** translation erases orthography/morphology (the dating signal),
so we use it as a *diagnostic*, never as the main input.
- https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349

### 2. Thalesian `cuneiformBase-400m` model card ★ read first
**Idea:** Our winning model is **not** from-scratch and not big — it's a **finetune of Google's
`umt5-base`** (an encoder-decoder) on cuneiform translation / transliteration / script-conversion
tasks. Encoder-decoder, multilingual, sign-aware tokenizer.
**How we use it:** This reframes the whole thesis question from "why does a big cuneiform model
win" to "which ingredient — tokenizer, encoder-decoder architecture, seq2seq objective, or the
finetune itself — carries the win." That is exactly the factor ladder in **Pillar 1**.
- https://huggingface.co/Thalesian/cuneiformBase-400m

### 3. mT5 / uMT5 (Xue et al. 2021; Chung et al. "UniMax" 2023)
**Idea:** mT5 is a multilingual encoder-decoder trained with **span-corruption** (mask spans, the
bidirectional encoder reconstructs them) across 100+ languages; uMT5 improves the language
sampling. The key contrast with Qwen/gpt-oss: **bidirectional encoder + seq2seq**, not decoder-only
next-token prediction.
**How we use it:** It's the *base* of Thalesian, so probing **vanilla `google/umt5-base`** (Pillar
1b) isolates "what the architecture/pretraining already gives" from "what the cuneiform finetune
adds." This is the cheap-but-decisive experiment.
- mT5: https://arxiv.org/abs/2010.11934 · uMT5/UniMax: https://arxiv.org/abs/2304.09151

### 4. Ithaca — restoring & attributing Greek inscriptions (Assael et al., Nature 2022)
**Idea:** A deep net that does three historian tasks at once — text **restoration**, **geographic**
attribution, and **chronological** attribution (dates to within ~30 years) — and crucially is built
to *assist* historians (their accuracy jumped 25%→72% with it), not replace them.
**How we use it:** It's the template for framing dating as an *attribution* task with interpretability,
and the model for the output contract in **Pillar 6** (ChronoAtlas).
- https://www.nature.com/articles/s41586-022-04448-z

### 5. Aeneas — contextualizing ancient texts (Nature 2025) ★ read first
**Idea:** Successor to Ithaca. Adds a **retrieval** mechanism: for a fragment it returns
historically grounded **parallel texts** as evidence. In evaluation with 23 epigraphers, the
retrieved parallels cut "find analogues" time from days to minutes.
**How we use it:** This *is* the design of **Pillar 6** — our output is an interval + confidence +
**nearest dated parallels** + earlier/later evidence, not just a year. The thesis's user-facing
contribution mirrors Aeneas, in Akkadian.
- https://www.nature.com/articles/s41586-025-09292-5

---

## B. The right training objective for the head (ordinal, not classification)

### 6. CORAL (Cao et al. 2019) & CORN (Shi, Cao, Raschka 2021)
**Idea:** Standard multiclass cross-entropy treats labels as unordered — wrong for years, where
670 BCE is *between* 700 and 640. CORAL/CORN turn an ordered label into a chain of
"is it ≥ threshold k?" binary questions, with rank-consistency guarantees (predictions can't say
"≥700 yes but ≥640 no"). CORN drops CORAL's weight-sharing limitation for more capacity.
**How we use it:** Motivates treating dating as **ordinal/interval regression** in **Pillar 0/2** —
the interval-target loss and the "respect the order" framing come from here.
- CORAL: https://arxiv.org/abs/1901.07884 · CORN: https://arxiv.org/abs/2111.08851 · code: https://github.com/Raschka-research-group/coral-pytorch

### 7. Rank-N-Contrast (Zha et al., NeurIPS 2023 spotlight)
**Idea:** A contrastive loss for *regression*: arrange embeddings so that distance in feature space
matches distance in the target (year). Texts close in time sit close; far-apart periods sit far
apart — a clean continuous timeline instead of fragmented clusters.
**How we use it:** The principle behind **Pillar 2's** pairwise ranking loss and **Pillar 3's**
cross-genre positive / same-genre-hard-negative pair mining.
- https://arxiv.org/abs/2210.01189 · code: https://github.com/kaiwenzha/Rank-N-Contrast

### 8. SimCSE (Gao, Yao, Chen, EMNLP 2021)
**Idea:** Make sentence embeddings robust by pulling two slightly-perturbed *views* of the same
sentence together. Even trivial noise (dropout) works as the augmentation.
**How we use it:** The intuition behind **Pillar 3's consistency loss** — but our views are
*historically motivated* (name-masked, formula-removed, cropped), so the date prediction must stay
stable when we strip the shortcuts ("dear King X"). That's how we prove the model isn't cheating.
- https://arxiv.org/abs/2104.08821

---

## C. Forcing honesty (deconfounding)

### 9. Domain-Adversarial Training / Gradient Reversal — DANN (Ganin et al. 2015)
**Idea:** Attach a small "adversary" net that tries to predict a nuisance variable (here: ruler,
genre, length, corpus) from the representation. A **gradient-reversal layer** flips the gradient
during backprop, so the main encoder is pushed to make that variable *un*predictable — it learns
features that work for the task but hide the confound.
**How we use it:** The core of **Pillar 3's** nuisance adversary. With the ruler caveat: ruler
correlates with time genuinely, so we run *with* and *without* the ruler adversary and report the
tradeoff, rather than scrubbing all ruler info.
- https://arxiv.org/abs/1505.07818

---

## D. Diachronic text dating (direct precedents)

### 10. TALM — Time-Aware Language Modeling for Historical Text Dating (EMNLP Findings 2023)
**Idea:** Learn *time-specific* word representations (a word's meaning/usage drifts across eras) and
model documents hierarchically on top of them; beats prior dating methods on a Chinese diachronic
corpus.
**How we use it:** Evidence that "model the diachronic drift directly" is the right framing, and a
baseline-design reference for **Pillar 2**. Confirms dating ≠ topic classification.
- https://aclanthology.org/2023.findings-emnlp.911/

### 11. TicTac — contrastive/metric learning for text dating (ACL Findings 2025)
**Idea:** Fine-tune with contrastive learning over two kinds of temporal relations between
documents, plus metric learning on the text↔period distance. Closest published cousin to our
"before/after pair" idea.
**How we use it:** Validates **Pillar 2/3's** pairwise + metric approach; worth mirroring its
relation types when designing pair sampling.
- https://aclanthology.org/2025.findings-acl.1129/

---

## E. Parked / deferred (read only when we revisit)

### 12. CLSS — semi-supervised contrastive regression via spectral seriation (NeurIPS 2023) — *Pillar 4, PARKED*
**Idea:** Use *unlabeled* data in regression by recovering an ordinal ranking from the feature
similarity matrix (spectral seriation) and using it as extra supervision.
**Why parked:** Our 2M unlabeled words have *no* dates, and unsupervised ordering on frozen
embeddings is ≈ what the geodesic work already tried without success. Revisit only on the labeled
royal inscriptions.
- https://openreview.net/forum?id=ij3svnPLzG · code: https://github.com/xmed-lab/CLSS

### 13. Snorkel — weak supervision (Ratner et al., VLDB 2017) — *Pillar 4, PARKED*
**Idea:** Combine many noisy "labeling functions" (heuristics) and estimate their accuracies to
produce probabilistic labels, instead of trusting any one heuristic.
**Why parked:** Only relevant if we generate weak date labels (ruler/eponym mentions) for the
unlabeled pool — tied to P4.
- https://arxiv.org/abs/1711.10160

### 14. Anthropic dictionary learning / Sparse Autoencoders — *Pillar 5, DEFERRED*
**Idea:** Transformer activations are dense and polysemantic; an SAE decomposes them into many
sparse, more interpretable "features" that are better units of analysis than raw neurons.
**Why deferred:** Interpretability should be *guided* by Pillar 1's findings (and ideally run on a
deliberately chosen, lesson-informed model), not on the current frozen activations blind.
- Towards Monosemanticity: https://transformer-circuits.pub/2023/monosemantic-features
- Scaling Monosemanticity: https://transformer-circuits.pub/2024/scaling-monosemanticity/

---

**If you only read five:** Thalesian model card (2), Aeneas (5), mT5/uMT5 (3), Rank-N-Contrast (7),
and DANN/gradient-reversal (9). Those cover the *why* (what Thalesian is), the *target* (Aeneas-style
output), and the two methods (ordinal contrast + deconfounding) that make our head honest.
