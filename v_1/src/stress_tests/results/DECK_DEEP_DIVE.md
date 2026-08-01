# The deck, slide by slide: full experimental detail for the thesis

Everything below is verified against the code in the repository (the scripts named in
each section are the ground truth), not from memory of the slides. Screen numbers are
the current 31-slide deck (`thesis_story_9.html`). Purpose: context for writing the
thesis; each section gives the experiment, the data, the process in enough detail to
reproduce it conceptually, the result, and how the slide hands off to the next.

## Global infrastructure (read this first, everything else refers back to it)

**The corpus.** `v_1/data/evaluation/corpora/orcc_corpus.parquet`: ORCC royal
inscriptions, one row per fragment with `fragment_id, ruler, year, provenance,
text_maximal`. Loading (`world_models/akkadian/akk_data.py: load_fragments`) keeps only
rows with a non-null year AND ruler and non-empty Akkadian text: **1193 dated fragments
across 40 rulers**, from Itti-Marduk-balatu (1132 BC) to Antiochus I (261 BC). The
distribution is severely skewed: Ashurbanipal 268, Sennacherib 237, Esarhaddon 176,
Sargon II 144, while eighteen rulers have 3 fragments or fewer. Each ruler carries
essentially one year value (the corpus dates fragments by reign), which is the central
confound of the whole thesis: **year is a function of ruler identity**. Translations
come from `stress_tests/translation/translations.parquet`: `eng_tier0` is the faithful
literal gloss (the valid one); `eng_maximal` is an aggressive cleaner that hallucinates
king names and is excluded from every experiment. Geo comes from
`stress_tests/shared/sites_gazetteer.csv` (56 provenance strings mapped to lat/lon; 74
distinct provenance values in the corpus, so unmatched ones drop): **1068 fragments
have coordinates**.

**Text regimes.** `tier0` = minimal normalization only (whitespace, unicode).
`maximal` (`linear_probing/utils.py: clean_maximal`) = tier0 plus **eleven stacked
filters and truncation to 30 words**: strip all digits; truncate to 30 tokens; strip
Akkadian case endings (-am, -im, -um, -tam, -tim, -shum); delete w/y characters; remove
logograms (all-uppercase tokens); strip determinatives (I-, d-, lu2-, uru-, gish-,
tug2-); keep only syllabic tokens; normalize long vowels (a-macron to a etc.);
strip subscript sign indices (sign2 to sign); lowercase; strip the -mesh plural. The
measured justification (`justification_maximal_cleaning_regime.md`): at tier0 a TF-IDF
classifier reaches 99.2% period accuracy by reading document length and name spellings,
because well-preserved eras leave long royal inscriptions; cleaning drops it to 96.8%
and cuts signal-carrying distinct unigrams from 84.8% to 69.3%. Truncation removes the
length crutch, the filters remove the name/logogram crutch.

**The model set** (`world_models/wm_lib/registry.py`). Trained decoders: Llama-2
7B/13B/70B (the paper's own series, NousResearch mirrors of the identical weights),
Qwen3 1.7B/8B/32B, gpt-oss-120B. Encoders: uMT5-base (multilingual pretraining, no
translation finetune), Thalesian AKK-300M (translation finetuned on Akkadian only),
Thalesian cuneiform-400M (translation finetuned on the multilingual cuneiform family).
Controls: a **random-init twin** of each decoder (identical config and tokenizer,
weights re-initialized with seed 42, no training; the 70B twin is a materialized
checkpoint), and **TF-IDF** floors (construction varies per experiment, always stated
below). Plus our own 37M MLM (screen 13).

**Extraction** (`wm_lib/extract.py`, `tokenize_lib.py`). Text is tokenized with
`add_special_tokens=False`, a BOS token prepended iff the tokenizer defines one (Llama
yes; Qwen, gpt-oss, T5-family no), **empty prompt** (the paper's canonical condition).
All transformer-block hidden states are saved (embedding layer skipped; stride 2 for
32B+ models, last layer always kept), stored fp16. Pooling sites: `last` = hidden state
at the final token; `mean` = mask-weighted average over all non-BOS, non-pad tokens.
fp16 can overflow to +-inf (gpt-oss especially), so every probe routes features through
`probing.sanitize` (clamp non-finite to +-65504; a layer with over 1% non-finite
entries is dropped).

**The two probe families.** (1) *Replication line* (`wm_lib/probing.py: run_probe`):
ridge regression, alpha chosen by RidgeCV over `logspace(-1, 5, 13)`, targets z-scored
on train statistics and un-scaled before scoring; place targets are (longitude,
latitude) as a 2-column regression, exactly the paper's convention. (2) *Thesis line*
(`stress_tests/shared/mc_probe.py`): features L2-normalized, PLS with k swept over
{1,2,3,5} (best-k surfaced) plus a Ridge arm, inside the balanced-draw machinery below.
The Akkadian MC scripts (`akk_modes.py`) use plain Ridge with alpha fixed to n_features
(the paper's heuristic) for speed across thousands of fits.

**The balancing machinery** (the anti-confound core). *Balanced Monte-Carlo by ruler*
(`akk_modes.mc_balanced`): restrict to the 8 best-attested rulers (r8, ~1071
fragments), draw min-count = 21 fragments per ruler, 200 draws (seed 42); within each
draw, 5-fold **StratifiedKFold by ruler** (every ruler in train and test), out-of-fold
predictions scored per draw, mean +- std over draws. This answers "can you read the
date with frequency imbalance removed, rulers seen." *LORO* (`akk_modes.loro`): leave
one ruler out entirely, pool out-of-fold predictions across all rulers, score once.
This answers "can you date a king never seen in training", and it is where every
activation arm collapses to ~0. *By-site MC* (`akk_modes.mc_site` via
`probe_geo_site.py`): find-spots merged on coordinates rounded to 0.1 degrees (~11 km,
so "Nineveh" and "Kuyunjik (Nineveh)" merge), sites with under 18 geocoded fragments
dropped, leaving **10 sites**; cap 21 per site, 200 draws, StratifiedKFold by site. The
stress-test line uses a fixed committed draw matrix (`draws_matrix.npy`, 200 draws of
8x21) with **GroupKFold by ruler inside each draw** (a ruler's fragments never cross
train/test within a fold), which is why stress-line numbers (rho ~ .3) are
systematically lower than world-models MC numbers (rho ~ .5): the former is a harder,
partially held-out-ruler protocol. Both are reported honestly where they appear.

**The reading rule.** A score counts as learning only if it beats both the TF-IDF
floor and that arm's own random-init twin. This rule was applied everywhere and is what
most slides turn on.

---

## Screen 1: Title

"When Are Space and Time Linearly Represented in Language Models? Entity salience,
language resource, and pooling." Frames the thesis as a boundary-condition study of
Gurnee & Tegmark rather than an applied-dating project. **Bridge:** the question the
title asks is motivated concretely on the next slide.

## Screen 2: Motivation

No experiment. Sets the stakes: if the linear space/time geometry reported in
interpretability work is a general property, a frozen LLM is a free
dating-and-provenance instrument for archaeology, where dating cuneiform fragments is
manual, expert-bound, and mostly unresolved. Names the three axes on which our regime
differs from the tested one: obscure entities, low-resource language, whole damaged
fragments instead of marked entity tokens. **Bridge:** before testing the transfer,
establish precisely what the prior work did.

## Screen 3: The paper

No experiment; a faithful description verified against the paper's own text
(`papers/txt/.../LANGUAGE MODELS REPRESENT SPACE AND TIME.txt`). Their method: entity
name through a frozen model, residual-stream activation at the **last entity token**
per layer, **ridge regression** (lambda by leave-one-out CV) to lat/lon or year, **R2
on held-out entities**; they also report Spearman (averaged over lat and lon for
space). Six datasets, 20k-40k entities each: World (39,585), USA (29,997), NYC
(19,838) places; Historical Figures (37,539), Art (31,321), Headlines (28,389).
Findings: strong recovery (World R2 .911 at Llama-2-70B), improves with scale, rises
through the first half of layers then plateaus; nonlinear MLP probes add almost
nothing. What the setting holds fixed: salient entities, English, explicit
entity-token pooling. What it lacks: a random-init control. **Bridge:** moving straight
to Akkadian changes several factors at once, so the next slide lays out the factorial
design.

## Screen 4: The matrix

The organizing device: {salient, obscure} x {high-resource English, low-resource
Akkadian}. A = the paper's cell. B = our obscure entities written in English. C = the
same entities in raw Akkadian. D = empty (no famous entities exist in Akkadian besides
these same royal names; no honest filler). A to B isolates entity obscurity; B to C
isolates language resource; within each cell we additionally vary entity-level vs
whole-fragment input and last-token vs mean pooling. A small 2x2 position marker
repeats on every subsequent results slide. **Bridge:** the shared protocol that makes
the cells comparable.

## Screen 5: Protocol

Documents the model set, controls, pooling sites, metrics (R2 as in the paper, plus
Spearman, leaned on for year because with 8 rulers dating is a ranking problem), the
balanced-MC idea, and the reading rule (all as in "Global infrastructure" above).
Explicitly flags that two probe families coexist (replication line:
last-token/ridge/R2; thesis line: mean-pool/PLS/Spearman) and that each slide states
which it uses. **Bridge:** first, verify the paper replicates at all.

## Screen 6: Cell A reproduction (table)

**Experiment** (`world_models/`: `fetch_data.py`, `extract_acts.py`, `probe_wm.py`,
`probe_eng_pls.py`, `tfidf_baseline.py`). Data: the paper's six CSVs vendored
byte-for-byte from wesg52/world-models @ a572f16, including the paper's own train/test
split column; entity strings rebuilt with verbatim ports of their string builders
(possessive constructions for world places and art, raw names for USA/figures, full
headline including final period). Extraction: empty prompt, max 96 tokens, last-token
(and mean) pooling, every layer. Probe: per-layer ridge as above; best layer selected
by held-out test R2; then `probe_eng_pls.py` refits PLS with k in
{1,2,3,5,8,16,32,64} at that same layer (reading the committed ridge JSON, so the layer
choice is identical). TF-IDF floor: word 1-2 grams + char_wb 2-5 grams of the exact
entity strings, ridge with alpha chosen on a 10% train carve-out, same split, same
metrics. 15 arms total.

**Result.** Our Llama-2-70B lands within .02 of every published number (World .905 vs
.911, peak layer 53 ~ their 65% depth). Qwen3 scales 1.7B to 8B to 32B (.655 to .797
to .838 on World), extending the scaling claim to a second family. The controls decide
the interpretation: Llama-2-70B World .905 vs its random twin .170; Art .860 vs .029.
TF-IDF is a genuine floor (.642 World, and it beats every model on NYC, .389): trained
arms clear it, random arms sit below it. The three translation encoders land above
random but below TF-IDF (.38-.44 on World): they are not generically strong probes,
which matters when they win later. PLS tracks ridge within ~.01-.03 everywhere.
**Bridge:** the numbers are real and learned; next, where in the network they live.

## Screen 7: Cell A layer sweep (figure)

Same activations and ridge probe as screen 6, plotted per layer
(`summary_layerwise.csv`, figure by `plot_cellA_figs.py`). Panels: SPACE (mean over
World/USA/NYC) and TIME (mean over Figures/Art/Headlines) x {last-token, mean} pooling
x {R2, Spearman}, x-axis = layer/total-layers so 28-to-41-layer models are comparable,
symlog R2 axis so failing arms' negative scores fit. **Result:** every trained arm
rises to a mid/late-depth peak then plateaus (reproducing their Figure 2 shape); random
twins are flat at the bottom; the weakest arms peak at the very first layers, which is
what "no representation was built" looks like; gpt-oss-120B lands mid-pack despite
being largest. **Bridge:** depth established; how many directions.

## Screen 8: Cell A PLS-k (figure)

At each arm's best ridge layer, PLS refit with k = 1 to 64 (`results/eng_pls/`).
**Result:** trained arms keep gaining to k ~ 16 and hold; random controls saturate at
k ~ 3-5 and several then decline. The learned representation is genuinely
multi-dimensional, not one strong axis; space needs slightly more components than time
(lat/lon is 2-D). **Bridge:** cell A verified in full. Climb step one: same language,
obscure entities.

## Screen 9: Cell B, entity level (table)

**Experiment** (`world_models/akkadian/`: `build_entity_datasets.py`,
`extract_entity.py`, `probe_entity.py`, aggregated by `aggregate_entity.py`). We built
the paper's own experiment with our entities. Time dataset: **34 rulers** (from the
corpus's 40, dropping 4 whose stored year is a regnal-year artifact under 100 and 2
"Unidentified" entries), target = the median attested year of the ruler's dated
fragments, spanning 1132 to 261 BC, mirroring their Historical Figures. Space dataset:
**25 excavation sites** (provenances with >= 3 fragments, de-duplicated by coordinates
so aliases can't straddle the split, ancient toponym spelling preferred: Nineveh not
Kuyunjik), target = lon/lat, mirroring World Places. Each entity appears once **bare**
(paper-faithful) and inside five neutral carrier sentences that never mention a date or
region ("This tablet dates to the reign of Ashurbanipal."), mirroring the paper's own
prompt-robustness check (theirs prepends prompts; ours embeds mid-sentence).
Extraction records the entity's character span via tokenizer offset mapping
(prefix-retokenization fallback), giving four sites: entity-last-token, entity-mean,
sentence-last, sentence-mean; on bare rows entity-last = sentence-last by
construction, verified in tests. Probe: because 34/25 entities make a single 20%
holdout meaningless (6-7 test entities), the headline is a **200-draw Monte-Carlo over
entity-level splits** (20% of entities out per draw; all six templates of an entity
move together, so no template leaks its target), ridge (RidgeCV grid) and PLS-5, rho
and R2. TF-IDF floor: char_wb 2-4 grams, 20k features, same splits.

**Result** (bare rows, entity-last-token, rho): Llama-2-70B **.701**, gpt-oss .663,
Qwen3-32B .627, down to Qwen3-1.7B .461; controls: Llama-70B random .457, TF-IDF .344.
Both families order by size. So time survives obscurity, but the trained-vs-control
margin is ~.24 where cell A's was ~.74 in R2, and by Llama-2-7B (.527 vs its twin
.473) it is inside the MC spread. Averaging over the name instead of taking its last
token puts every model at .40-.57 and never separates from controls (stated in the
takeaway, not tabled). Place fails outright: the best value in the geo column is the
**untrained** Llama-2-70B (.459); no arm beats its twin; R2 is negative for all.
Adding the carrier sentences moves top arms by under .01. **Bridge:** entities still
work (weakly) in English; now generalize the unit to whole fragments, the paper's
Headlines analog.

## Screen 10: Cell B, fragments, year (table)

**Experiment** (`akkadian/extract_akk.py` + `probe_akk.py`). Entity = the whole
fragment's **English tier0 gloss**; empty prompt; max 256 tokens; last and mean
pooling; every layer. Per variant x r8 x year x site: best layer chosen by an internal
stratified-by-ruler 80/20 holdout, then **balanced MC by ruler** at that layer (r8,
cap 21, 200 draws, StratifiedKFold-by-ruler, ridge alpha=n_features). Table shows MC
rho and R2 for both poolings plus a per-metric difference column. TF-IDF row from the
same MC protocol on the gloss text.

**Result:** TF-IDF wins outright (rho .775). Averaging over the passage adds ~+.20 rho
to nearly every arm relative to last-token (the inverse of screen 9, because a date's
traces are distributed across a passage, not concentrated at one token), but the
controls gain equally: best trained arms (AKK-300M .740, Qwen3-8B/32B .737) sit just
above untrained Llama-70B .661 and untrained Qwen3-8B .636. No scaling trend survives.
**Bridge:** same story for space?

## Screen 11: Cell B, fragments, geo (table)

**Experiment** (`akkadian/probe_geo_site.py`). Same gloss embeddings, target = (lon,
lat) of the find-spot, which is never named in the text. 1068 geocoded fragments,
merged into 10 sites (0.1-degree merge, min 18), **by-site MC**: cap 21, 200 draws,
StratifiedKFold-by-site, ridge; best layer by an internal holdout first. TF-IDF for
this experiment is richer (char_wb 2-5 + word 1-2 grams to SVD-256, fit on train
only). **Result:** trained arms beat TF-IDF on R2 under last-token (up to .34 vs .02),
which never happens for year, but untrained twins climb with them (untrained Qwen3-8B
.337 last, .450-.463 mean, the best numbers in their columns). TF-IDF's split verdict
is diagnostic: decent rho (.535) and collapsed R2 (.022), i.e. n-grams can rank sites
but not place them. Mean pooling again beats last for almost everyone, and again the
gain is shared with the controls. **Bridge:** before entering the low-resource
language, look inside the best English-side embedding.

## Screen 12: Embedding explorer, English (figure)

**Experiment** (`stress_tests/eda/make_embedding_panels.py`, panels committed under
`e6_clusters/embedding_panels/`). The 1,202 ORCC fragments embedded by each arm at its
best year layer, reduced to 2-D four ways: t-SNE, PCA, UMAP (unsupervised) and
**supervised PLS** (k=3 fit on year; component 1 vs 2, so year separation is partly
baked in and must be read against the tfidf/random panels in the same folder). Each map
drawn six times, colored by year, ruler, period, sub-genre, provenance, and
log-length. The slide shows the best English-side arm: Qwen3-32B on the tier0 gloss,
supervised-PLS view (year-probe rho .437, rank 1/9 on that leaderboard). **Result and
reading:** the year gradient is visible, but the ruler, period, provenance and length
panels paint the same regions; the corpus's confounds are visually inseparable, which
is the pictorial justification for every balancing device in the deck. An interactive
viewer (`embedding_panels/index.html`, `interactive.html`) ships in the repo.
**Bridge:** now the language changes; first, the one model that is Akkadian all the way
down.

## Screen 13: Our MLM (architecture slide)

**Model** (specs from `justification/research_log_phases_0_to_track_a.md`,
`justification_mlm.md`, `justification_sign_level_tokenization.md`; diagram by
`plot_mlm_arch.py`). Objective: masked language modelling, not next-token, following
*Filling the Gaps in Ancient Akkadian Texts* (restoration IS the masked-token task;
bidirectional context matters when neighbors are broken), same lineage as
Ithaca/Aeneas. Architecture: 16-layer pre-norm transformer encoder, d=384, FF 1536, 8
heads (d_head 48), RoPE, RMSNorm, 36,705,229 params; MLM head over the sign
vocabulary; 15% masking with BERT's 80/10/10 corruption. Data: 2.45M words / 4.89M
signs pooled from ORACC (56.6%), eBL (40.7%), Archibab (2.7%), split 80/10/10 **at
fragment level** (no tablet crosses splits), tokenized at **sign level** ("a-na" ->
"a na") following the EvaCun 2025 shared task. Training: 10 epochs, batch 8, val loss
4.55 to 3.24. Differences from Ithaca drawn in the figure: single sign-level input row
(no word row); single restoration head; region/date heads replaced by linear probes on
frozen per-layer activations. **Role:** separates "Akkadian cannot support a timeline"
from "the LLMs never saw enough Akkadian", and is the non-translation contrast for the
encoders later. **Bridge:** with the toolkit complete, enter cell C at entity level.

## Screen 14: Cell C, king-name token (table)

**Experiment** (stress-test line: `linear_probing`-lineage extraction +
`stress_tests/shared/mc_probe.py`, results in `csv/p1_year_mc.csv`). Two read-outs on
the same fragments: (a) whole-fragment **mean** under **maximal** cleaning (names
stripped), and (b) the **ruler's name token** (its last token) inside the tier0 text
(names necessarily kept; maximal would delete them). Protocol: the committed 200-draw
matrix (8 rulers x 21), **GroupKFold-by-ruler within each draw** (this is the harder
split; see infrastructure note), features L2-normalized, PLS k in {1,2,3,5} best-k
with a Ridge arm; shuffled-label null ~ .01. Models: Qwen family, gpt-oss, the three
encoders, our MLM, TF-IDF, random Qwen3-8B (no Llamas in this pipeline). **Result:**
fragment column: everything lands .27-.39, on top of TF-IDF .271 and untrained .293;
flat in scale. Name-token column: everything jumps (.42-.70), but the **untrained
network scores .643**, beating most trained models; averaging over the name's tokens
collapses to ~0 for all (dropped to a sentence). **Bridge:** the name column needs an
autopsy before it can be read.

## Screen 15: Ruler is not chronology (table)

**Experiment** (`stress_tests/shared/mc_maxking.py`, `csv/p1_maxking.csv`). Same
king-token activations, three analyses per balanced draw: *ruler_clf*, PLS-DA (k swept
{1,2,3,5}) under StratifiedKFold predicting the ruler label, macro-F1 vs chance .20
and a shuffle baseline; *year_strat*, the pooled-year Spearman with rulers seen;
*year_group*, GroupKFold-by-ruler year Spearman (within-ruler ordering, degenerate by
construction since each ruler has one year; splits capped at 2 so test folds hold >= 2
rulers). **Result:** trained models identify the ruler at F1 .94-.99 and "order years"
at rho .93-.98, and the **untrained Qwen3-8B does both too (.946 / .926)**. Once ruler
identity is held constant the correlation is zero-to-negative for every arm. So the
name-token score is a lookup table: identify the king, get his year for free. This is
why every following slide reads whole fragments with names stripped. **Bridge:**
fragments, names stripped, in the raw language.

## Screen 16: Cell C, fragments, year (table)

**Experiment:** identical pipeline to screen 10 (`probe_akk.py`), variant =
`akk_maximal` (the 11-filter, 30-word-truncated Akkadian). Balanced MC r8, both
poolings, rho and R2, per-metric difference columns. **Result:** TF-IDF .707 tops the
table; on last-token the untrained Qwen3-8B (.499) beats every trained model; on mean
the only arm clearly above its controls is **cuneiform-400M (.699)**, against
untrained Llama-70B .588 and untrained Qwen3-8B .544. Against the same fragments in
English (screen 10) every arm loses ~.15-.20 rho: what the models had was access to
English, not to the content. The one survivor is the thread the closing act picks up.
(Not shown but in the JSONs: under LORO every arm, TF-IDF included, collapses to rho
<= .13.) **Bridge:** space, same regime.

## Screen 17: Cell C, fragments, geo (table)

Identical pipeline to screen 11, variant = `akk_maximal`. **Result:** under
last-token, trained Llamas reach R2 .23-.33 against TF-IDF .019, but untrained
Qwen3-8B posts .351, best in column; under mean, cuneiform-400M leads (.612 rho / .430
R2) with untrained twins at .385-.412 close behind. Space is partially recoverable
from how the fragments are written, but most of that recoverability is available to an
untrained network as well. **Bridge:** the layer-by-layer view of the same collapse.

## Screen 18: Cell C layer sweep (figure)

**Experiment** (`akkadian/probe_layers_pls.py`, figure by `plot_cellC_figs.py`):
per-layer ridge on a fixed stratified-by-ruler 80/20 holdout (this slide is the
holdout protocol, stated on it), akk_maximal, year (rho) and geo (R2), both poolings,
all 14 arms. **Result:** curves are flat and interleaved with the dashed controls;
most best-layers sit in the first few layers (nothing gets built with depth);
**cuneiform-400M is the exception**, rising with depth to late-layer peaks on both
targets under mean pooling; gpt-oss-120B shows a unique shape, collapsing through the
middle then climbing sharply in the last layers. **Bridge:** the dimensionality
counterpart.

## Screen 19: Cell C PLS-k (figure)

Same script, PLS k = 1 to 64 at each arm's best layer. **Result:** almost every arm
peaks at k ~ 5-16 then declines, sharply by k = 32-64 (place panels go negative); on
English the same probes kept gaining to 16 and held. A representation that is hurt by
extra directions is a thin surface feature, not a rich geometry. **Bridge:** look
inside the winner's Akkadian space.

## Screen 20: Embedding explorer, Akkadian (figure)

Same panel pipeline as screen 12; the arm shown is cuneiform-400M on maximal Akkadian,
supervised-PLS view (rho .391, rank 1/10 on that leaderboard). A coarser but visible
year gradient (the Neo-Babylonian mass separates from the Neo-Assyrian core), less
organized by ruler than the English map, length gradient damped by truncation.
**Bridge:** the probes say the LLM signal is absent; the rescue act asks whether
anything can bring it back.

## Screen 21: Rescue 1, T9 knowledge (table)

**Experiment** (`redo_t9_knowledge/`, prompts and scorers reused from the approved
phase-1a kp0/kp1/kp2 set; `csv/t9_knowledge.csv`). No probe, no Akkadian: chat models
asked in English (kp0) "When did {ruler} reign?" for the 8 balanced rulers, answers
parsed as JSON reign windows, correct if the window widened by +-10 years covers a
true reign year (accuracy is identical at +-50/+-30/+-10: models either know exactly
or are wrong at any tolerance); (kp1) "List the rulers of period P", scored as recall
of the target roster with diacritic-insensitive matching on raw text (robust to
truncation; Qwen3-8B's early 7/8 was a truncated-JSON parse artifact, rescored 8/8);
(kp2) a hallucination gate. **Result:** gpt-oss 8/8 and recall 1.00; Qwen3-32B 6/8,
1.00; Qwen3-8B 8/8, .75; Qwen3-1.7B 7/8, .50. The declarative knowledge exists.
**Bridge:** so remove the probe, or help it.

## Screen 22: Rescue 2, ask it / prompt it (two tables)

**Experiment A, "ask it"** (`t12_forced_dating/generate_forced.py` +
`score_forced.py`): each maximal-cleaned fragment shown to the chat model under four
prompt styles (pv0 bare, pv1 expert framing, pv2 five worked examples drawn
leakage-free from non-eval rulers with seed 42, pv3 chain-of-thought), the model
**forced** to output a single ruler + year guess (T11 had allowed "cannot estimate"
and small models took that exit for 87-100% of fragments); the answered year is scored
as the prediction, Spearman on the same 200 balanced draws. **Experiment B, "prompt
it"** (`redo_t10_prompt/`): keep the probe, but extract activations from **inside**
those same four prompts, pooling the mean over the fragment's span tokens; same MC
protocol (`csv/t10_mc.csv`). **Result:** answers are worse than the probe (rho <=
.374, often ~0 or negative); prompted-activation probing sits within ~.05 of bare for
every model and style, no size ordering. The signal is not hidden behind a bad
read-out or a badly posed question. **Bridge:** maybe it just needs more Akkadian.

## Screen 23: Rescue 3, continued pretraining (figure)

**Experiment** (`finetune/`: `prepare_ntp_data.py`, `train_ntp.py`, scoreboard in
`results/scoreboard_best.csv`; figure by `plot_finetune_fig.py`). Corpus:
fragment-level Akkadian from the canonical unified train split, built identically to
the probing corpus, tier0-cleaned; 11.25M Qwen tokens (~2.5M words, effectively all
published Akkadian); test fragments untouched. A tokenizer EDA ruled out vocabulary
expansion (domain BPE saturates at 3.85 tok/word, and the high-value candidate tokens
are royal/divine names, exactly the leakage channel the cleaning removes). Models:
Qwen3-1.7B/8B/32B and gpt-oss-120B, each fine-tuned with plain next-token prediction
at **four unfreezing depths** (blocks unfrozen from 0% / ~33% / ~67% / ~90% of depth;
e.g. Qwen3-8B cuts at 0/12/24/32), embeddings frozen except in the full-depth arm.
Every checkpoint re-probed with the identical balanced-MC PLS protocol on maximal.
**Result:** delta rho between -.013 and +.002 across all 16 arms; for 32B and 120B
several frozen-depth arms are byte-identical to base at the probed layers. The one
apparent gain (gpt-oss at tier0, +.048) vanishes under maximal: it had learned to
exploit text length, the exact artifact the regime removes. **Bridge:** maybe the
probe reads word order and we destroyed it with cleaning; test order directly.

## Screen 24: Rescue 4, shuffle (table)

**Experiment** (`e5_shuffle/probe_e5_mc.py`): each fragment word-capped first, then
its word list permuted **once** (seed 42), giving exactly one shuffled twin with
identical words and length; both twins extracted with identical settings, mean pool;
balanced MC (200 committed draws, GroupKFold-by-ruler), PLS best-k; the number that
matters is delta = unshuffled minus shuffled at each variant's own best layer, on
maximal Akkadian and on the tier0 English gloss. **Result:** delta between -.006 and
+.062 across all arms and both languages; TF-IDF is order-blind by construction and
loses exactly .000, which is the yardstick. The chronological signal is carried by
which words appear, not how they are arranged. **Bridge:** maybe the failure is
linearity itself.

## Screen 25: Rescue 5, curved and kernel probes (table)

**Experiment** (`p9_gkpls/gkpls.py`): mean-pooled, L2-normalized activations; k=10
nearest-neighbor graph per training fold; geodesic distances through the fixed train
graph (out-of-sample points attach to their k nearest training points,
Nystrom-centered kernel column, so no leakage); three arms per fold isolating each
ingredient: **G-KPLS** (kernel PLS on the Isomap kernel K_G = -1/2 H D^2 H, components
a in {1,2,3,5}), **RBF-KPLS** (same KPLS on the Euclidean RBF kernel,
median-heuristic bandwidth; isolates curvature vs kernel), **KRR on K_G** (kernel
ridge, lambda in {.001, .01, .1}; isolates PLS vs kernel). GroupKFold-by-ruler,
balanced draws, Spearman. **Result:** geodesic loses to plain RBF in 30 of 34 cells;
kernelizing buys nothing over the linear PLS reference; TF-IDF's RBF-KPLS lands inside
the trained band. Non-linearity is not the missing ingredient. **Bridge:** last dial,
supervision itself.

## Screen 26: Rescue 6, the supervision dial (table)

**Experiment** (`p8_lambda_probe/`): one spectral probe with a knob lambda solving
[(1-lam) X'M_h X - lam X'L_h X]v = gamma X'DX v on PCA-100 features fit per training
fold, where M_h is the centered RBF kernel of the **year labels** (HSIC-style
supervised term) and L_h is a kNN heat-graph Laplacian (pure geometry); lam=0 is
supervised-PCA on the year, lam=1 is Laplacian eigenmaps that never see the year; test
fragments are projected linearly (LPP-style), GroupKFold-by-ruler. Read-outs:
|Spearman| of the leading coordinate with the year, and ridge-on-Z predictions.
**Result:** flat in lambda for every model; the unsupervised end matches the
supervised end (whatever ordering exists is already in the cloud's shape, and it is
the surface-statistics shape, since the random model's pure-geometry direction is as
good as any trained one). **Bridge:** all rescues exhausted; state the conditions.

## Screen 27: Conditions (synthesis)

No new experiment. Reads the ladder back down: condition 1, the language must be well
represented in training (B to C drops every trained arm onto its twin); condition 2,
the entities must be salient enough to have been written about (A to B quarters the
margin); condition 3, the read-out must sit where the information is (entity-last
carries names, passage-mean carries documents; the relation inverts between levels).
Not substitutes: scale, declarative knowledge, prompting, asking, continued
pretraining on all existing Akkadian, word order, kernel probes, unsupervised
geometry. **Bridge:** the one positive result.

## Screen 28: The winner (figure, thesis-line headline)

**Experiment** (thesis line, `csv/table1_best_models.csv`): full model set on maximal
Akkadian, mean pooling, the committed 200 balanced draws, PLS best-k with Ridge
agreement, year Spearman at the best layer. **Result:** cuneiform-400M .391 leads
every arm: Qwen3-8B .339, Qwen3-1.7B .334, gpt-oss .330, MLM ~.29 band, TF-IDF .266
(ridge), random .293. It is the only arm satisfying the reading rule on this task, at
~300x fewer parameters than the largest LLM. Note the protocol difference from screen
16's .699: this is the harder GroupKFold-within-draw split; both are reported with
their own protocol stated. **Bridge:** why this model; isolate the objective.

## Screen 29: The translation line (figure)

**Experiment** (`plot_encoders_fig.py` over `layers_pls` JSONs): the three same-family
encoders layer by layer on maximal Akkadian, year (rho) and place (R2), both poolings,
with untrained Qwen3-8B dashed (no encoder random-twin exists; no 1.7B-scale twin
exists; stated on the slide). The three differ only in objective/data: uMT5-base =
multilingual pretraining, no translation finetune; AKK-300M = translation finetune on
Akkadian alone; cuneiform-400M = translation finetune on the multilingual cuneiform
family. **Result:** monotone ordering, no-finetune < Akkadian-only < multilingual, on
both targets, clearest under mean pooling where cuneiform-400M rises with depth (to
rho ~ .54 year, R2 ~ .35 place) while uMT5 decays and the control stays flat.
Consistent with the Malkin/Limisiewicz/Stanovsky finding that configuration rather
than multilinguality per se drives transfer; the three-way comparison varies size,
objective and data together, and the deck flags that itself. **Bridge:** rule out the
mundane alternative.

## Screen 30: Tokenizer (figure)

**Experiment** (`finetune/eda/tokenizer_eda.py`): tokens per Akkadian word for every
arm's tokenizer over our corpora. **Result:** cuneiform-400M is the *least* efficient
(6.22 tok/word) and wins; gpt-oss is the most efficient (4.43) and is mid-pack. If
tokenizer fit drove dating, the ranking would run the other way. The objective remains
the live explanation. **Bridge:** close.

## Screen 31: The boundary condition (final)

The main point and four takeaways (each hyperlinked to its anchor paper: Gurnee &
Tegmark ICLR 2024; Godey et al. LREC 2024; Malkin et al. NAACL 2022), with the hedged
robustness paragraph: when the input moves out of distribution in entity familiarity
or language, the world model does not come along; the maps are tied to the training
distribution more tightly than the term suggests.

---

## Three protocol asymmetries a thesis writer must keep straight

These explain every apparent numeric contradiction between slides:

1. **Stress-line vs world-models-line splits.** GroupKFold-by-ruler inside draws
   (harder, rho ~ .3 band; screens 14, 22-26, 28) vs StratifiedKFold-by-ruler inside
   draws (easier, rho ~ .4-.7 band; screens 10, 16).
2. **Holdout vs MC vs LORO.** The same arm can score .59 / .43 / -.07 (Llama-2-7B,
   Akkadian year) depending only on this choice, which is itself a finding.
3. **PLS k-grids.** {1,2,3,5} in the stress line, {1,...,64} in the layers_pls line.

All three are stated on the slides where they apply.
