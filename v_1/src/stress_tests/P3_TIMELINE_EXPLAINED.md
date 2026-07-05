# P3 "timeline" experiments — explained from zero, with a worked example

This explains the whole P3 family in plain language: what goes in, what comes out,
what the metric is, and how a prediction is made. Two pipelines exist:
**P3-v1** (unsupervised: PCA-1D + Isomap-1D, `timeline_p3.py`) and
**P3-v2** (supervised: PLS-3D + geodesic, `timeline_p3_pls.py`). Same anchors, same
texts — they differ only in HOW the timeline is built.

---

## 0. The question P3 asks

P1 asked: "can a trained probe read the year out of the text embedding?"
P3 asks something different: **"does the model's embedding space ALREADY contain a
timeline — a 1-D path where earlier↔later is laid out geometrically — and do real
texts sit on it?"** No probe is trained on the texts; we only look at geometry.

---

## 1. The ingredients

### 1a. Anchors = calibration points with a KNOWN date
An anchor is a short **English** sentence that names a date or a ruler. We know its
year by construction. Examples (real rows from `anchors/qwen3_8b/anchors.json`):

| kind  | year BCE | prompt |
|-------|----------|--------|
| ruler | 705 | "an Akkadian royal inscription from the reign of Sargon II" |
| ruler | 681 | "an Akkadian royal inscription from the reign of Sennacherib" |
| year  | 697 | "an Akkadian royal inscription from the year 697 BCE" |
| year  | 507 | "an Akkadian royal inscription from the year 507 BCE" |

There are **153 anchors = 40 ruler prompts** (one per ruler that has a year label;
its year = that ruler's corpus year) **+ 113 year prompts** (one every 10 years
across the span seen in the corpus, 7→1132 BCE).
⚠️ Known flaw: the corpus has a few junk year labels (bad `sub_period` parses), so
anchors like "the year 7 BCE" exist. Kept for comparability; a filtered rerun is an
open option.

Each anchor is pushed through the frozen model and **mean-pooled**: average the
hidden state of its ~10 tokens → **one vector**. Done at every layer. So per model
per layer we have a cloud of **153 points, each with a known year**.

Why English? Because T9 proved the models KNOW "Sargon II ≈ 705 BCE" *in English*.
The anchors sit on the model's declarative-knowledge side. The question is whether
Akkadian TEXTS connect to that knowledge geometrically.

### 1b. Texts = the 1,202 ORCC fragments
Each fragment's text (in a chosen **cleaning**: tier0 / maximal / maxking) is pushed
through the same model, mean-pooled → one vector per fragment per layer. These are
the same activation files P1 uses. The cleaning applies ONLY to the texts — anchors
are English and identical across cleanings.

### 1c. "Per layer" 
A 37-layer model gives 37 versions of everything (anchors cloud + text vectors).
The whole analysis runs independently per layer, and we report the best layer —
because "where in the network would a timeline live?" is part of the question.

---

## 2. P3-v1 — the unsupervised version (what "3a" and "3b" mean)

### 3a: "do the anchors form a line?"
Take the 153 anchor vectors at one layer. Ask: if I flatten this cloud to a single
dimension, does the resulting ORDER match the years?

Two flattenings:
* **PCA-1D (linear):** find the single straight direction with the most variance;
  each anchor's output = its position along that line (one number).
* **Isomap-1D (the "non-linear" one):** the timeline might be a CURVE, not a line.
  Isomap: (1) connect each anchor to its 10 nearest neighbours → a graph;
  (2) measure distance between any two anchors ALONG THE GRAPH (like road distance
  vs straight-line distance); (3) find the 1-D arrangement that best preserves
  those road distances. Output again = one number per anchor.

**Metric (3a):** |Spearman(position, year)| over the 153 anchors. Spearman only
compares ORDERINGS: sort anchors by position, sort by year — do the two orders
agree? 1.0 = the cloud's main axis IS time; 0 = unrelated. (Absolute value because
a 1-D axis has no inherent direction.)

Important: neither PCA nor Isomap ever sees the years. They just find the cloud's
dominant shape. 3a asks whether that shape happens to be time. That's the point —
and also the weakness Yarin identified: the dominant-variance direction might be
something else entirely (length, style), even if a time direction exists. That's
what v2 fixes.

### 3b: "do real texts land on that line?"
For each of the 1,202 text vectors: find its **nearest anchor** (cosine similarity;
both sides L2-normalized, which makes cosine ranking = Euclidean ranking). The
prediction for the text = **that anchor's year**.

Worked example: fragment Q003230's mean vector is closest to the anchor
"…from the year 647 BCE" → predicted year 647. Its true year is 631 → error 16.

**Metric (3b):** Spearman(predicted year, true year) over all dated texts.
Result observed: 3a ≈ 0.2–0.57 (anchors roughly line up), 3b ≈ 0.03–0.13 (texts do
NOT land on it) — the dissociation.

---

## 3. P3-v2 — the supervised geodesic version (Yarin's proposal, `timeline_p3_pls.py`)

The v1 objection: PCA finds max-VARIANCE directions, ignoring years. The fix: use
**PLS**, which finds directions of max COVARIANCE WITH YEAR.

**Config:** per model × cleaning (tier0/maximal/maxking) × layer. Anchors fixed;
only the text side changes with cleaning.

**Step by step:**

1. **Fit PLS(n_components=3) on the ANCHORS ONLY:** X = 153 anchor vectors
   (L2-normalized), y = their 153 years. PLS finds 3 directions along which the
   anchor cloud co-varies with year. Each anchor → a **3-D point** (its scores on
   those directions). No corpus text is involved in the fit → no leakage.

2. **Build the timeline through the 3-D anchor cloud:**
   * `pls1` = Spearman(1st PLS coordinate, year) — the straight supervised axis.
   * `geo1` = Isomap-1D over the 3-D points — the **geodesic (curved) timeline**:
     nearest-neighbour graph → distances along the graph → best 1-D arrangement →
     Spearman(coordinate, year).
   ⚠️ Read these as sanity checks, not results: PLS was FIT on these same anchors
   with these same years, so high pls1/geo1 is partly by construction (in-sample).

3. **Project the texts** (the honest part): every text vector is mapped by the SAME
   fixed PLS transform → a 3-D point in the anchor space. The texts' years were
   never shown to anything.

4. **Predict each text's year by interpolation** (Yarin's idea): find its **5
   nearest anchors** in the 3-D space; predicted year = weighted average of their
   years, weights = 1/distance (closer anchor → more influence).
   Worked example: text T lands nearest to anchors with years 697 (d=0.10),
   705 (d=0.15), 681 (d=0.30), 727 (d=0.60), 507 (d=0.90). Weights ∝ 10, 6.7, 3.3,
   1.7, 1.1 → prediction ≈ (10·697+6.7·705+3.3·681+1.7·727+1.1·507)/22.8 ≈ **692 BCE**.
   (Plain nearest-anchor m=1 is also reported for comparison.)

5. **Ruler classification (bonus):** among the 40 ruler-anchors only, find the
   text's nearest one → predicted ruler → **macro-F1** against the text's true
   ruler label. Chance ≈ 1/40.

**Metrics (v2), the ones that matter:**
* `proj_interp_spearman` — Spearman(interpolated year, true year) over texts.
* `proj_nn1_spearman` — same with nearest-anchor only.
* `ruler_macro_f1` — nearest-ruler-anchor classification.
* (`pls1`, `geo1` — anchor-side sanity checks, in-sample.)

**Baseline, as always: the random-init Qwen3-8B row.** If random's text-projection
matches the trained models', the projection is reading token identity, not a
learned timeline.

---

## 4. One-paragraph summary you can say out loud

"We plant 153 English calibration sentences with known dates — 'an inscription
from the reign of Sargon II', 'an inscription from the year 507 BCE' — and embed
them with the frozen model, one vector each. v1 asks, unsupervised: if you flatten
that anchor cloud to one dimension (straight = PCA, curved = Isomap), does the
order match the years? — and then predicts each Akkadian text's date as its
nearest anchor's date. v2 instead uses PLS to find the three directions of the
anchor cloud that co-vary with YEAR, projects every text into that 3-D space, and
predicts its date by distance-weighted interpolation over its five nearest
anchors; a curved timeline through the anchors is read out with geodesic Isomap.
The metric everywhere is Spearman rank correlation between predicted and true
years (plus macro-F1 for nearest-ruler classification), and the control is a
random-weights model run through the identical pipeline."
