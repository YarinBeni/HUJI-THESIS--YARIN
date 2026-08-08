# E1 — pairwise chronology: "which fragment is earlier?"

Phase-2 experiment E1 (see `../DECIDED_EXPERIMENTS.md`). Recasts fragment dating
from *predict the year* to *order the pair*, which (a) multiplies the usable data —
the corpus supports 628,454 ordered pairs against the ~1.1k labeled fragments the
regression ever saw, (b) readmits the 33 long-tail rulers the balanced regression
design had to discard, and (c) leans only on *relative* order, so a systematic bias
in the absolute year labels cannot manufacture the result.

## The balancing problem, and the design that answers it

Fragment counts per ruler run 268 → 1, and pairing **squares** the skew:
Ashurbanipal × Sennacherib alone is 63,516 possible pairs; a tail × tail
ruler-pair is exactly one. Three mechanisms, each catching what the previous one
cannot:

1. **Quota per ruler-pair (sampling).** Each Monte-Carlo draw takes
   `min(m, n_i, n_j)` fresh pairs per eligible ruler-pair, m = 21 (echoing the
   regression's k = 21), no fragment reused within a ruler-pair within a draw.
   Since every ruler sits in ~39 ruler-pairs, equalizing ruler-pairs equalizes
   rulers automatically. Data exhaustion happens **across** draws — the big grids
   are resampled fresh every draw, so 100 draws sweep them broadly; `--m 100
   --draws 10` is the written-in robustness setting that pushes coverage further.
2. **Weight 1/m_ij (training).** A tail ruler-pair can still only contribute 1
   pair vs 21 from the giants, so every pair carries `sample_weight = 1/m_ij`:
   each ruler-pair contributes total weight 1 to the loss no matter its size.
3. **Macro-averaging (evaluation).** Metrics are computed per ruler-pair and then
   averaged over ruler-pairs — never over raw pairs. A ruler-pair with one pair
   and a ruler-pair with 21 count the same.

What balancing cannot fix, stated instead of hidden: a 1-fragment ruler's every
pair flows through that one fragment (pseudo-replication), and the independent
units are ruler-pairs (777 eligible), not the 628k pairs. Uncertainty is
mean ± sd over draws now; ruler-level wild bootstrap joins in the E8 pass.

**Splits.** Rulers are shuffled into 5 folds per draw; a pair *trains* only when
both its rulers are in train folds, *tests* only when both are in the test fold,
and straddling pairs are unused — the pairwise analog of the GroupKFold-by-ruler
protocol that killed the leak in the regression design. ~18% of a draw is
testable per split; reshuffling folds every draw restores coverage of the rest.

**Eligibility.** Year-labeled fragments only: 1,187 fragments, 40 rulers, 777
ruler-pairs. The 9 undated rows drop out, which also removes the entire `ribo`
pseudo-ruler (all its rows are undated — investigated, not an issue). Same-year
fragment pairs carry no order and are never emitted.

## The two experiments

**Probe (`probe_pairs.py`, jobs F1).** A linear pairwise-logistic scorer on
activation differences — a Bradley-Terry model with linear features. It learns a
**time direction without ever seeing an absolute year**, saved to
`results/directions/` for the E3 cosine comparison against the frozen cell-A name
direction. Arms: the full ladder (OLMo-2, Llama-2 7/13/70B + random twins, Qwen3
1.7/8/32B, gpt-oss-120b, random) plus the `tfidf_char` floor (char_wb 2–5-gram,
the exact vectorizer of `tfidf_akk.py`). Layer chosen on a cheap selection pass,
then the full MC runs at that layer (same mild optimism as the deck's
holdout-best-layer convention; flagged in the JSON as
`layer_selected_on_same_protocol`).

**Behavioural (`behavioral_pairs.py`, jobs F2).** Ask the model itself: "Was Text
A composed earlier than Text B? Answer only Yes or No", single-token logit
read-out, **both presentation orders per pair** with the P(A,B)/P(B,A)
consistency check — the protocol of El-Shangiti et al. (NAACL 2025) adapted from
entity names to whole inscriptions. Instruct-capable arms only (qwen3 family);
base models have no Yes/No calibration and F1 already covers their
representations.

**Reading the two together.** Trained arms beating the TF-IDF floor at pairwise
ordering, having failed to beat it at regression, would mean document-level time
exists and absolute calibration was the bottleneck. The floor still winning says
the collapse is deeper than task format. The dyear-binned accuracies separate
"only tells 600-year gaps apart" from real resolution.

## Files

| file | what |
|---|---|
| `pairs_data.py` | pair engine: eligibility, balanced draws, ruler folds. Self-test: `python pairs_data.py` |
| `probe_pairs.py` | representation probe, all arms + tfidf floor |
| `behavioral_pairs.py` | Yes/No querying, both orders, consistency |
| `aggregate_pairs.py` | tidy CSVs from the result JSONs |
| `sbatch/F1_pairs_probe.sbatch` | CPU array, 14 arms × {akk_maximal, eng_tier0} |
| `sbatch/F2_pairs_behavioral.sbatch` | GPU array, qwen3 arms × {eng_tier0, akk_maximal} |
| `sbatch/F3_pairs_robustness.sbatch` | CPU array, the `--m 100 --draws 10` pass on the headline arms (writes `*.m100.json`) |

Results land in `results/probes/`, `results/directions/`, `results/behavioral/`,
summarized by `aggregate_pairs.py` into `results/summary_*.csv`.

## Progress

- [x] Design settled: quota m=21 + 1/m_ij weights + macro-over-ruler-pairs; both-ruler-held-out folds (2026-08-07)
- [x] `pairs_data.py` written; self-test passes (1,187 frags / 40 rulers / 777 ruler-pairs; per-ruler-pair weight sums verified = 1; label balance .49)
- [x] `probe_pairs.py` written; local smoke on `tfidf_char` (2 draws): macro_acc ≈ .65 — the floor to beat
- [x] `behavioral_pairs.py`, `aggregate_pairs.py`, F1/F2 sbatch written
- [x] Full local `tfidf_char` floor, akk_maximal (100 draws): **macro_acc = .658 ± .038**, auc .652 — the number to beat
- [x] Full local `tfidf_char` floor, eng_tier0 (100 draws): macro_acc = .586 ± .038 — the floor is markedly LOWER on the English gloss than on raw Akkadian (.658), consistent with the deck's finding that Akkadian surface orthography itself carries period signal
- [x] F1 on cluster (job 22587): all 13 arms landed (llama2_70b eng_tier0 came in last: .587, on the floor)
- [x] F2 on cluster (job 22588): all 3 qwen3 arms — behavioural task is DEGENERATE (massive No-bias: yes_rate 0–.38, order-consistency 0–.54, macro ≈ chance); the probe is the informative read
- [x] F3 robustness (job 22589): all 4 arms; m=100 gives the same picture, uniformly a touch lower
- [x] First read of the probe table (see below) — headline: **akk_maximal replicates the collapse in pairwise form** (every arm within noise of the .658 floor, random twins interleaved with trained models at the top); **eng_tier0 shows real separation** (trained OLMo-2 .634 and Qwen3-8B .636 vs floor .586, with olmo2_7b_random at the BOTTOM, .553 — a ~.08 trained-vs-twin gap that raw Akkadian entirely lacks)
- [x] RESULTS.md written — full tables, dyear bins (trained edge lives at 0–75 yr resolution in eng), m100 comparison, behavioural degeneracy
- [x] `e8_inference.py` written (E8: dyadic ruler bootstrap on paired contrasts + ruler-permutation with refit); local smoke passes (perm null ≈ .49 vs obs .57 on the tfidf floor)
- [x] F4 ran on cluster (job 22607, both tasks)
- [x] F4 numbers folded into RESULTS.md §4 — quote the permutation dissociation, not the pairwise contrasts
- [x] E3 hook done in F5: cosine(pairwise direction, cell-A name direction) ≈ .01 on every arm — chance-level orthogonality; the two time axes are different axes
- [x] E8 pass done in F4 (dyadic ruler bootstrap + permutation-with-refit); tables in RESULTS.md §4
- [ ] Esarhaddon within-ruler pairs (E6; needs a small `--within-ruler` extension of `pairs_data.draw_pairs`)
