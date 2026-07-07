# Deck consolidation audit — thesis_story_9.html

> **STATUS 2026-07-06: EXECUTED.** A1–A4 accepted and applied (deck now 21 slides,
> controls-first order: T9 → P2-siteMC → P1 → maxking → P3-merged → P7-v2 → T10 → translation),
> full ladder added to every probing table, footnote technicalities cleaned. A5 kept both
> (part-1 figures + P1 table). Below is the original audit for the record.
>
> **UPDATE 2026-07-07:** Yarin removed the merged P3 timeline slide for now (deck = 20 slides;
> P3 numbers remain in results/csv/p3_*.csv and the walkthrough). P1 TF-IDF row now shows real
> balanced-MC numbers in the fixed-k convention (results/tfidf_mc_recomputed.json). Translation
> slide expanded to YEAR + GEO tables with lat+lon and akk-tier0/akk-maximal reference columns.

Purpose: find (A) duplicate experiments shown under different configs, (B) model gaps
per slide, (C) exposed technicalities to clean. **Nothing deleted yet** — each item has
a recommendation for Yarin to accept/reject. Guiding principle (Yarin): one concrete
story; the canonical protocol is **balanced-MC · maximal-family cleaning · mean pool ·
PLS(+Ridge) · Spearman**; anything pre-MC or non-canonical needs a reason to stay.

---

## A. Duplicates / supersessions

| # | slides | verdict | recommendation |
|---|---|---|---|
| A1 | **14 (P2, GroupKFold-by-site)** vs **20 (P2, site-balanced MC)** | same experiment, old vs canonical protocol | **DROP 14, keep 20.** 20 is the MC version matching the P1 protocol. If a "held-out-site generalization" point is ever needed, it lives in `results/csv/p2_geography.csv`, not a slide. |
| A2 | **18 (P7 v1: tier0-only, classification-only)** vs **22 (P7 v2: 3 cleanings, cls+reg, per-k curves)** | v2 strictly supersedes v1 | **DROP 18, keep 22.** Optionally move 18's "best k / k@90%" localization columns into 22's footnote. |
| A3 | **19 (T10 tier0-only)** vs **23 (T10 all cleanings)** | 23 contains every row of 19 (mean site) | **DROP 19, keep 23.** 19's extra `8B king_last` row → one footnote line on 23 if wanted. |
| A4 | **17 (P3 v1: PCA/Isomap unsupervised)** vs **21 (P3 v2: PLS-geodesic supervised)** | NOT duplicates — complementary halves of one argument ("no unsupervised timeline" + "even supervised projection fails") | **MERGE into ONE timeline slide**: columns = 3a PCA-1D · 3a Isomap-1D · 3b nearest-anchor ρ · v2 interp ρ (tier0), footnote for maximal/maxking v2 numbers. One slide, whole timeline story. |
| A5 | **15 (P1 balanced-MC table)** vs **original figure slides 4/6/7** (PLS-vs-Ridge, k-sweep, scale-flat — all the same maximal-balanced year probe) | same numbers, two presentations (part-1 figures tell the original story; 15 extends with tier0 + king sites) | **KEEP BOTH** but decide the hand-off: part 1 = "the finding", slide 15 = "the stress-test extension". Alternative if the deck must shrink: drop figure slide 6 (k-sweep detail) — it's methodology, already stated in the pipeline slide. |
| A6 | P1 **non-MC (GroupKFold-single)** version | exists only in CSVs/walkthrough — **already not a slide** | nothing to do (matches the "MC only" principle). Same for T10-GKF: CSV-only. ✓ |

Net if all accepted: 25 → **21 slides** (drop 14, 18, 19; merge 17+21).

## B. Model gaps per follow-up slide

Legend: ✓ on slide · (csv) = exists in results/csv but omitted from the slide · ⏳ in flight · — by design.

| slide | 1.7B | 8B | 32B | gpt-oss | AKK-300m | cunei-400m | uMT5 | random | MLM | TF-IDF |
|---|---|---|---|---|---|---|---|---|---|---|
| 13 T9 | ✓ | ✓ | ✓ | ✓ | — | — | — | — | — | — |
| 15 P1-MC | ✓ | ✓ | ✓ | ✓ | **(csv)** | ✓ | **(csv)** | ✓ | ✓ | ✓ |
| 16 maxking | **(csv)** | ✓ | ✓ | ✓ | **(csv)** | ✓ | ✓ | ✓ | —¹ | —¹ |
| 17 P3 v1 | ✓ | ✓ | ✓ | ✓ | **(csv)** | ✓ | ✓ | ✓ | —¹ | —¹ |
| 20 P2 site-MC | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | —¹ | —¹ |
| 21 P3 v2 | ✓ | ✓ | ✓ | ✓ | **(csv)** | ✓ | ✓ | ✓ | —¹ | —¹ |
| 22 P7 v2 | ✓ | ✓ | ✓ | ✓ | **(csv)** | ✓ | ✓ | ✓ | —¹ | —¹ |
| 23 T10 | ✓ | ✓ | ✓ | ⏳ tier0 (max/maxking ✓) | —² | —² | —² | —² | —² | —² |
| 24 translation | ✓ | ✓ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ✓ | —¹ | —¹ |

¹ MLM has no extractor for these arms (open option); TF-IDF has no per-layer/per-token
activations (bag-of-signs; cited baseline in P1 only). By design, not gaps.
² T10 prompts a chat model; random/encoders are not meaningful there. By design.

**Real fixable omissions (rows exist in CSVs, just not on slides): AKK-300m on 15/16/17/21/22;
uMT5 + 1.7B on 15/16.** These were dropped for slide-height only. Recommendation: add them —
8-row tables render fine (slide 20 proves it) — so every probing slide shows the full ladder.

## C. Technicality exposure to clean (fix-history / job names)

| slide | exposure | recommendation |
|---|---|---|
| 13 T9 | ~~strict vs rescored kp1 columns + token-budget story~~ | **DONE** — single "kp1 recall" column, clean footnote (per Yarin's decision) |
| 20, 22 | footnotes cite job names (J14, J9b) | drop job names; keep only the CSV pointer |
| 23 | "gpt-oss × tier0 extraction failed twice (under diagnosis)" | replace with neutral "pending" until the cell lands, then delete |
| 24 | pending-rows note cites J17b/J17c | fine while pending; delete when rows land |
| 21 | "(Yarin's method)" in the eyebrow | keep or rename to "supervised timeline" for the advisor version — style call |
| walkthrough | keeps the full strict/rescored audit trail | KEEP there (it is the justification doc); the deck stays clean |

## D. Consistency notes (no action, just awareness)

- Slide 15's `mean maximal` column and part-1 figure slides 4/6/7 report the same
  protocol — numbers agree (both balanced-MC). If an advisor cross-checks, they match.
- Slide 20 (site-MC) replaces slide 14's GKF numbers with *higher* lat values (0.51–0.72
  vs 0.20–0.43): different CV → different absolute levels. Another reason not to show both.
- P3 v1 (17) reports |ρ| (sign-free); P3 v2 (21) reports signed ρ for text projection.
  The merged slide should state this explicitly.

## E. Suggested final order (if A1–A4 accepted → 21 slides)

0–11 unchanged (original story) · 12 divider+standard · 13 T9 · 14 P1-MC (full ladder) ·
15 maxking · 16 P3 timeline (merged v1+v2) · 17 P7 v2 · 18 P2 site-MC · 19 T10 (complete) ·
20 translation · 21 contributions/discussion (move current 11 to the end?) — or keep 11
where it is and end on translation. Yarin's call.
