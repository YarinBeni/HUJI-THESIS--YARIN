# Deck verification report — thesis_story_9.html

Date: 2026-08-02. Scope: every quantitative claim and protocol description on
all 31 slides, re-derived from the committed result files and probe code by
four independent verification passes (slides 1-8, 9-15, 16-22, 23-31). This
was triggered by the discovery that the fragment-year tables were reading the
ruler-stratified Monte-Carlo key while the slide text described the
ruler-grouped protocol.

## What verified clean

Every numeric table in the deck now matches its committed source file exactly,
cell by cell:

- Slide 6 (cell A): all 90 Ridge R2 cells, all 90 Ridge Spearman cells, all
  PLS cells, and the paper row against `paper_reference.json`.
- Slide 9 (entities): all rows against `summary_entity_best.csv` (bare,
  ent_last).
- Slides 10/16 (fragment year): all rows against the `mc_group` blocks
  (GroupKFold-by-ruler inside 200 balanced draws), TF-IDF included.
- Slides 11/17 (fragment place): all rows against the `mc_site` blocks.
- Slide 14 (king token): all cells against `p1_year_mc.csv` +
  `tfidf_mc_recomputed.json`.
- Slide 15 (identity autopsy): all committed rows against `p1_maxking.csv`.
- Slide 21 (T9): against the committed kp0 metrics and rescored recalls.
- Slide 22 (T12/T10): all 32 cells against `t12_forced_dating.csv` and
  `t10_mc_cleanings.csv`.
- Slide 23 (finetune): all deltas against `finetune/results/scoreboard_best.csv`.
- Slide 24 (shuffle): all 16 cells against `e5_shuffle.csv`.
- Slide 25 (kernels): all 20 rows against `p9_gkpls.csv` + PLS refs.
- Slide 26 (dial): all 100 cells against `p8_lambda.csv`.
- Slide 28 (winner): traced to `geodesic/maximal_figs` T1/T2 (cuneiform-400M
  PLS .411 / Ridge .364, rank 1 in both).
- Slide 30 (tokenizer): 6.22 / 4.43 against `tokenization_audit.csv`.
- Protocol descriptions on slides 5, 10, 11, 16, 17, 22 match the code
  (`akk_modes.py`, `probe_akk_group.py`, `probe_entity.py`, `mc_probe.py`).

## Errors found and fixed (all corrected in the deck)

Protocol / read-out class (the serious kind):

1. Slides 10/16 fragment-year tables were built from the ruler-STRATIFIED
   `mc` key (rulers on both sides of every split; TF-IDF .707) while the text
   described the grouped protocol. Switched to `mc_group` (TF-IDF .330,
   best trained .329-.400), Spearman-only since grouped R2 collapses to the
   same constant (~-0.22) for every arm. Takeaways rewritten from the grouped
   numbers.
2. Slides 11/17 claimed the geo draws "hold out whole find-spots". The code
   (`akk_modes.mc_site`) stratifies BY SITE: sites appear on both sides.
   Text now says these tables measure matching a fragment to a known site,
   and the find-spot headline was reworded.
3. Slide 6 encoder cells mixed mean-pooled Ridge with last-pooled PLS from a
   different layer. The PLS value now reads the same site (mean) and layer
   as the Ridge value it shares a cell with.
4. Slide 18's takeaway called the layer figure "balanced"; it is a single
   stratified holdout, not the balanced draws. Word removed.
5. Slide 15 claimed "same activations, same draws" as slide 14 and "8
   rulers, chance .20". The maxking config is 5 rulers x 9 king-found
   fragments on its own draw matrix; chance .20 is right for 5 classes.
   Both corrected.

Numbers corrected:

6. Slide 21: Qwen3-8B reign dating is 7/8 in the committed scores, not 8/8
   (the 8th answer was truncated and factually wrong). Table + takeaway.
7. Slide 13: committed MLM validation curve ends at 3.02, not 3.24 (3.24 is
   the epoch-7 value).
8. Slide 10: twenty-two rulers have <=3 fragments, not eighteen.
9. Slide 9: held-out entities are exactly 7 rulers / 5 sites per draw, not
   "6 to 7"; the obscure-vs-famous margin is about HALF the famous margin
   (.244 vs .452), not a quarter (also fixed on slide 31); the carrier-
   sentence shift is ~.01 on year but larger on place.
10. Slide 6 / 31: reproduction is within .03 of the paper (Art gap .025),
    not .02.
11. Slide 25: G-KPLS <= RBF-KPLS in 36 of 44 cells (committed CSV), not 30/34.
12. Slide 23: the data gap to pretraining is five to six orders of magnitude,
    not four; Qwen3-32B arms equal base only at display precision; the 120B
    was finetuned through LoRA (now stated).

Overclaims softened to match the data:

13. Slide 28: "only arm that clears both controls" -> only arm that clears
    them by more than the spread of the draws; PLS and Ridge agree on the
    winner, not the full ordering.
14. Slide 29: the encoder ordering holds in three of four panels (year
    average: uMT5's first-layer peak edges AKK-300M), now stated.
15. Slide 24: Ridge flips sign on the largest shuffle cost (Qwen3-8B); the
    TF-IDF row is the Ridge arm, now labeled.
16. Slides 14/15: the untrained name-token score beats half the trained arms
    (not "most"/"several"); name-token averaging does not collapse for the
    MLM (.38); target has 18 distinct year values, not 8.
17. Slides 3/5: the TF-IDF floor is word + character n-grams; random twins
    exist for four of the seven decoders; encoder last-token probes exist
    (mean is the reported read-out, not the only one); uMT5-base is not
    translation-finetuned.
18. Slides 7/8: depth range is 8 to 80 layers; random controls peak at the
    network edges (not only the start); on space their PLS-k curves keep
    creeping to k~16-32; the "23/27/31 all Akkadian there is" wording hedged
    to "effectively all the published Akkadian".
19. Slide 4: matrix slide ranges corrected (B spans 9-12, C spans 13-20);
    slide 16's "two slides ago" cross-reference and "cut to the same length"
    (actually capped at the first 30 words) fixed.
20. Slide 13: the MLM paper is cited Lazar et al., EMNLP 2021 (first author;
    Stanovsky is senior author). NOTE: an earlier instruction asked for
    "Stanovsky et al."; changed for citation correctness, flag if preferred
    otherwise.

## Removed for lack of a committed source

- Slide 15's MLM row (.970/.977): no maxking result for the MLM exists
  anywhere in the repo (absent from the JSON dir, the CSV, and
  RESULTS_maxking.md). Rerun the maxking probe for the MLM arm and restore
  the row if wanted.

## Remaining items that cannot be file-verified

- Slide 3's statements about what Gurnee & Tegmark report (scaling trend,
  layer profile, MLP probes, Spearman): the paper text is not in the repo and
  the sandbox cannot reach arXiv. They match the paper as known; spot-check
  against the PDF.
- Slide 13's 4.55 -> 3.02 curve belongs to the committed `models/baseline`
  run; the deck's activations come from `models/baseline_retrained`, whose
  training stats are not committed. Commit that run's stats to close the gap.
- "Effectively all the published Akkadian": the corpus size (2.45M words) is
  committed; completeness of coverage is asserted, not evidenced.
