"""Rebuild `thesis_story_9.html` in the ladder order defined by STORY_SPINE.md.

The deck's slide bodies (including the base64 figures) are reused verbatim from the
current HTML; this script only reorders them, drops the superseded ones, and inserts
the newly authored slides. Keeping it as a script rather than hand-editing 1.2 MB of
HTML means the deck can be regenerated whenever a new experiment lands — in
particular slide 8, which is a placeholder until the WB (cell-B entity) jobs finish.

    python build_story_deck.py                 # rewrite thesis_story_9.html in place
    python build_story_deck.py --out new.html  # write elsewhere instead
    python build_story_deck.py --check         # verify counters only

SPINE entries are either `("old", index)` — reuse that slide from the current deck —
or `("new", key)` — emit NEW_SLIDES[key]. TOTAL/TITLES/data-index are rewritten from
the result, which is the invariant the audit doc warns about.
"""
import argparse
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "thesis_story_9.html")

# The reusable slide bodies come from the *pre-reorder* 33-slide deck, not from the
# current file — otherwise a second run would try to reuse indices the first run
# already removed. Git blob SHAs are immutable and survive rebases, so pinning the
# blob keeps this script re-runnable forever (e.g. to refresh a slide when new
# results land) without committing a second 4 MB copy of the deck.
SOURCE_BLOB = "538265f1af40652a0cee88d9862766d62576b2d0"   # thesis_story_9.html @ 4177af08

# ---------------------------------------------------------------- the spine ----
# (kind, ref, title-for-the-nav-strip)
SPINE = [
    ("new", "title", "Boundary Conditions of Linear Space and Time Representations"),
    ("new", "motivation", "Why we care: a world model you could point at an excavation"),
    ("new", "paper", "Gurnee & Tegmark: a strong claim, tested in one cell"),
    ("new", "matrix", "The climbing map: entity salience x language resource"),
    ("new", "protocol", "How every experiment in this deck is set up and read"),
    ("new", "cellA_repro", "A: the paper reproduces on our models, with the controls it never ran"),
    ("new", "cellA_layers", "A: where in the network space and time live"),
    ("new", "cellA_pls", "A: how many PLS directions the world model needs"),
    ("new", "b_entity", "Step 1 — same language, obscure entities: time survives, space does not"),
    ("old", 25, "Step 2 — English gloss, whole fragments: Year"),
    ("old", 26, "Step 2 — English gloss, whole fragments: Geo"),
    ("new", "confounder", "Before cell C: the confounder controls, and what they cost"),
    ("old", 15, "Step 3a — the king-name token in raw Akkadian"),
    ("old", 27, "Step 3b — raw Akkadian, whole fragments: Year"),
    ("old", 28, "Step 3b — raw Akkadian, whole fragments: Geo"),
    ("old", 31, "Cell C — layer sweep on the Akkadian mirror"),
    ("old", 32, "Cell C — PLS components at the best layer"),
    ("new", "ruler_not_chrono", "What the king-name probe actually learns: identity, not chronology"),
    ("old", 13, "Rescue 1 — do the models simply know the dates? (T9)"),
    ("new", "ask_directly", "Rescue 2 — ask the model instead of probing it (T12 + T10)"),
    ("old", 7,  "Rescue 3 — scale and next-token finetuning on our own Akkadian"),
    ("old", 19, "Rescue 4 — is it word order? (E5 shuffle)"),
    ("old", 20, "Rescue 5 — is the probe family too weak? (P9 geodesic kernels)"),
    ("old", 21, "Rescue 6 — how much supervision would it take? (P8 dial)"),
    ("new", "conditions", "What a linear temporal world model needs in order to exist"),
    ("old", 4,  "What does work: the 400M translation encoder beats the 120B LLM"),
    ("old", 9,  "Why: the translation objective, not the language or the size"),
    ("old", 8,  "Ruling out the tokenizer"),
    ("new", "contributions", "Contributions, and what this says about world-model claims"),
]

# --------------------------------------------------------- newly authored ------
NEW_SLIDES = {

"title": """<section class="slide slide-title">
  <div class="title-inner">
    <div class="title-kicker">M.Sc. Thesis &middot; Advisor Meeting &middot; 2026</div>
    <h1 class="title-h1">Boundary Conditions of Linear Space and Time Representations in Language Models</h1>
    <div class="title-sub">From salient English entities to obscure entities, a low-resource language, and whole fragments</div>
    <div class="title-meta">M.Sc. Thesis &middot; Yarin Beer &middot; Computer Science, HUJI<br>Advisors: Prof. Gabriel Stanovsky &nbsp;&middot;&nbsp; Dr. Barak Sober</div>
  </div>
</section>""",

"motivation": """<section class="slide slide-text">
  <div class="eyebrow">Motivation</div>
  <h2 class="sh">Why we care: a world model you could point at an excavation</h2>
  <div class="text-points">
    <div class="tp"><div class="tp-h">The claim we are following up</div><div class="tp-b">Recent interpretability work reports that language models build <strong>linear internal maps of time and space</strong>. A ridge probe on frozen activations recovers when a person died, or where a city is. If that geometry is a general property of large models, it is a free measuring instrument.</div></div>
    <div class="tp"><div class="tp-h">Why an archaeologist should care</div><div class="tp-b">Dating and provenancing cuneiform fragments is done by hand, from ruler names, script style and archival context: slow, expert-bound, and for most fragments simply unresolved. A probe that reads date and find-spot straight out of a frozen model would turn a scholarly bottleneck into an inference. <strong>That is the payoff we are testing for.</strong></div></div>
    <div class="tp"><div class="tp-h">But the claim was only ever tested in its comfort zone</div><div class="tp-b">Famous entities (world capitals, celebrated figures, headline news), written in <strong>English</strong>, read out of <strong>explicitly marked entity tokens</strong>. Our regime differs on three axes at once: the entities are <strong>obscure</strong>, the language is <strong>low-resource</strong>, and the unit is a <strong>damaged fragment</strong> rather than a name. This deck climbs those axes one at a time and reports where the map survives and where it breaks.</div></div>
  </div>
</section>""",

"paper": """<section class="slide slide-text">
  <div class="eyebrow">The work we extend &middot; Gurnee &amp; Tegmark, ICLR 2024</div>
  <h2 class="sh">&ldquo;Language Models Represent Space and Time&rdquo;: a strong claim, tested in one cell</h2>
  <div class="text-points">
    <div class="tp"><div class="tp-h">Their method, which we adopt unchanged</div><div class="tp-b">Run an entity name through a frozen model; save the residual-stream activation at the <strong>last entity token</strong> for every layer; fit <strong>ridge regression</strong> (&lambda; by leave-one-out CV) to latitude/longitude or to a year; report <strong>R&sup2; on held-out entities</strong>. Six datasets, roughly 20&ndash;40k entities each: World, USA, NYC (space) and Figures, Art, Headlines (time).</div></div>
    <div class="tp"><div class="tp-h">Their finding</div><div class="tp-b">Recovery is strong (World R&sup2; = .911 at Llama-2-70B), rises with scale, and improves through the first half of the layers before plateauing. Nonlinear MLP probes add almost nothing, which is their argument that the structure really is linear.</div></div>
    <div class="tp"><div class="tp-h">What the setting quietly holds fixed</div><div class="tp-b"><strong>Three things at once.</strong> Every dataset is <em>salient</em> entities; every dataset is <em>English</em>; and the read-out is always an <strong>explicit entity token</strong>, a span the experimenter marks in advance. So the study cannot separate the model's geometry from the well-represented slice of the world it was trained on, and it never asks whether the same information is recoverable <strong>implicitly</strong>, from a whole text in which no entity is singled out. Our corpus varies all three.</div></div>
    <div class="tp"><div class="tp-h">And one control it does not run</div><div class="tp-b">No <strong>random-initialised twin</strong>. Without it, a probe's success cannot be separated from what a linear map extracts from any high-dimensional representation. We add that control everywhere, plus a <strong>TF-IDF character n-gram floor</strong>.</div></div>
  </div>
</section>""",

"matrix": """<section class="slide slide-text">
  <div class="eyebrow">The design</div>
  <h2 class="sh">The climbing map: entity salience &times; language resource</h2>
  <div class="exp-config">Going straight from the paper's setting to Akkadian changes <strong>two things at once</strong>, so a collapse could not be attributed to either. We lay the space out as a 2&times;2 and climb it one factor at a time.</div>
  <table class="rtbl matrix"><thead><tr><th></th><th>High-resource language (English)</th><th>Low-resource language (Akkadian)</th></tr></thead><tbody>
    <tr><td><span class="mdl">Salient entities</span></td><td><strong>A:</strong> the paper's cell. World capitals, famous figures, headlines. <em>Slides 6&ndash;8.</em></td><td><strong>D: empty.</strong> No famous entities exist in Akkadian outside these same royal names, so no honest filler exists.</td></tr>
    <tr><td><span class="mdl">Obscure entities</span></td><td><strong>B:</strong> Assyrian rulers and find-spots, written in English. <em>Slides 9&ndash;11.</em></td><td><strong>C:</strong> the same entities in raw Akkadian. <em>Slides 13&ndash;18.</em></td></tr>
  </tbody></table>
  <div class="text-points" style="margin-top:10px">
    <div class="tp"><div class="tp-h">A &rarr; B isolates entity obscurity</div><div class="tp-b">Same language, different entities. It also asks a second question: <strong>is the language itself the barrier to linear recoverability?</strong> If a faithful English translation restores the signal that raw Akkadian loses, then the knowledge is stored in activation space by <strong>meaning rather than surface form</strong>, and translation becomes a way to reach an LLM's internal representations for any low-resource language.</div></div>
    <div class="tp"><div class="tp-h">B &rarr; C isolates language resource</div><div class="tp-b">Same entities, different language. This is what tells us whether the model has learned the <strong>actual patterns of Akkadian</strong> from the real text, or whether it only ever had access to the content once someone rendered it in a language it knows.</div></div>
    <div class="tp"><div class="tp-h">Inside every cell: what counts as the entity, and how we pool</div><div class="tp-b">We first run the paper's <strong>entity-level</strong> protocol, then extend to <strong>whole fragments</strong>, and pool both at the <strong>last token</strong> (theirs) and as a <strong>mean over the text</strong> (new).</div></div>
  </div>
</section>""",

"protocol": """<section class="slide slide-text">
  <div class="eyebrow">Protocol</div>
  <h2 class="sh">How every experiment in this deck is set up, and how to read its numbers</h2>
  <div class="cfg">
    <div class="cfg-k">Models</div><div class="cfg-v">Llama-2 <strong>7B / 13B / 70B</strong> (the paper's own series) &middot; Qwen3 <strong>1.7B / 8B / 32B</strong> &middot; <strong>gpt-oss-120B</strong> &middot; three small seq2seq translation encoders (<strong>cuneiform-400M</strong>, <strong>AKK-300M</strong>, <strong>uMT5-base</strong>).</div>
    <div class="cfg-k">Controls</div><div class="cfg-v">a <strong>random-initialised twin</strong> of every decoder (same architecture and tokenizer, untrained weights) and a <strong>TF-IDF character n-gram floor</strong>. Both appear as rows in every results table, never as a slide of their own.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v"><strong>entity last token</strong> (the paper's read-out, an explicitly marked span) &middot; <strong>text last token</strong> (their Headlines read-out, applied to our whole fragments) &middot; <strong>mean over the text</strong> (new: no span is marked, so the signal must be recoverable implicitly, and it is the only option for the encoders, which have no causal last token).</div>
    <div class="cfg-k">Metrics</div><div class="cfg-v"><strong>R&sup2;</strong> on held-out entities, as in the paper, and <strong>Spearman &rho;</strong>, which the paper also reports. We lean on &rho; for year because dating is a ranking problem and our year distribution is compressed and unevenly spaced.</div>
    <div class="cfg-k">Balancing</div><div class="cfg-v"><strong>balanced Monte-Carlo</strong>, used wherever the corpus is ours: each of the 8 dense rulers is capped at the same number of fragments and the probe is refit over <strong>200 redrawn splits</strong>, so no score can come from one over-represented king. The paper's English datasets are already balanced and need none of this.</div>
    <div class="cfg-k">Verdict</div><div class="cfg-v">a score witnesses <em>learning</em> only if it beats <strong>both</strong> the TF-IDF floor <strong>and</strong> that arm's own random twin. Beating neither means the probe found geometry that any random projection would have offered.</div>
  </div>
  <p class="fig-note">Two protocol families run through the deck, and each slide states which one it uses in the box at the top. The <strong>replication line</strong> (cells A, B, C) follows the paper: last-token pooling, ridge, R&sup2;. The <strong>thesis line</strong> (the closing slides) uses what the applied dating work settled on: mean pooling, PLS, Spearman. Numbers are comparable within a line, not across the two.</p>
</section>""",

"cellA_repro": """<section class="slide slide-text">
  <div class="eyebrow">A &middot; salient entities, English &middot; the paper's own setting</div>
  <h2 class="sh">The paper reproduces on our models, and the controls it never ran hold up</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">the paper's six English datasets (World, USA, NYC &rarr; <strong>latitude/longitude</strong>; Figures, Art, Headlines &rarr; <strong>year</strong>). Each entity's last-token embedding, every layer, probed on held-out entities. We reproduce it with <strong>our extended set of models</strong> (both decoder families, gpt-oss-120B, three translation encoders) and with the <strong>controls the paper never ran</strong>.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">best-layer held-out test <strong>R&sup2;</strong> for all six datasets, as in the paper. Each cell is <strong>Ridge | PLS</strong>, PLS taken at its best k &le; 64 at that same layer.</div>
  </div>
  <p class="tbl-cap">best-layer held-out test R&sup2; &nbsp;&middot;&nbsp; space | time &nbsp;&middot;&nbsp; each cell = Ridge | PLS</p>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">World</th><th class="num">USA</th><th class="num">NYC</th><th class="num">Figures</th><th class="num">Art</th><th class="num">Headlines</th></tr></thead><tbody>
    <tr><td><span class="mdl">Llama-2-70B</span> (paper)</td><td class="num">.911</td><td class="num">.864</td><td class="num">.359</td><td class="num">.835</td><td class="num">.885</td><td class="num">.746</td></tr>
    <tr><td><span class="mdl">Llama-2-70B</span> (ours)</td><td class="num">.905|.903</td><td class="num">.846|.831</td><td class="num">.363|.326</td><td class="num">.833|.827</td><td class="num">.860|.853</td><td class="num">.757|.748</td></tr>
    <tr><td><span class="mdl">Llama-2-13B</span></td><td class="num">.883|.880</td><td class="num">.808|.793</td><td class="num">.272|.240</td><td class="num">.802|.799</td><td class="num">.780|.770</td><td class="num">.663|.652</td></tr>
    <tr><td><span class="mdl">Llama-2-7B</span></td><td class="num">.859|.857</td><td class="num">.788|.775</td><td class="num">.249|.205</td><td class="num">.784|.778</td><td class="num">.770|.756</td><td class="num">.592|.582</td></tr>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.807|.806</td><td class="num">.656|.658</td><td class="num">.112|.079</td><td class="num">.803|.803</td><td class="num">.739|.748</td><td class="num">.510|.500</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">.838|.839</td><td class="num">.702|.691</td><td class="num">.187|.147</td><td class="num">.806|.800</td><td class="num">.727|.718</td><td class="num">.605|.598</td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.797|.793</td><td class="num">.634|.617</td><td class="num">.117|.075</td><td class="num">.774|.768</td><td class="num">.658|.649</td><td class="num">.557|.548</td></tr>
    <tr><td><span class="mdl">Qwen3-1.7B</span></td><td class="num">.655|.660</td><td class="num">.450|.440</td><td class="num">.080|.050</td><td class="num">.693|.691</td><td class="num">.449|.437</td><td class="num">.476|.465</td></tr>
    <tr><td><span class="mdl">uMT5-base</span></td><td class="num">.438|.284</td><td class="num">.325|.264</td><td class="num">.133|.044</td><td class="num">.494|.401</td><td class="num">.153|.096</td><td class="num">.349|.234</td></tr>
    <tr><td><span class="mdl">cuneiform-400M</span></td><td class="num">.399|.279</td><td class="num">.344|.297</td><td class="num">.114|.045</td><td class="num">.460|.401</td><td class="num">.126|.084</td><td class="num">.343|.295</td></tr>
    <tr><td><span class="mdl">AKK-300M</span></td><td class="num">.381|.249</td><td class="num">.312|.274</td><td class="num">.120|.036</td><td class="num">.448|.358</td><td class="num">.123|.099</td><td class="num">.300|.213</td></tr>
    <tr class="rand"><td><span class="mdl">TF-IDF</span> (floor)</td><td class="num">.642</td><td class="num">.536</td><td class="num">.389</td><td class="num">.645</td><td class="num">.116</td><td class="num">.448</td></tr>
    <tr class="rand"><td><span class="mdl">Llama-2-70B random</span>*</td><td class="num">.170</td><td class="num">.240</td><td class="num">.014</td><td class="num">.198</td><td class="num">.029</td><td class="num">.148</td></tr>
    <tr class="rand"><td><span class="mdl">Llama-2-13B random</span>*</td><td class="num">.282|.269</td><td class="num">.290|.277</td><td class="num">.044|.025</td><td class="num">.284|.267</td><td class="num">.038|.058</td><td class="num">.267|.265</td></tr>
    <tr class="rand"><td><span class="mdl">random Qwen3-8B</span>*</td><td class="num">.327|.311</td><td class="num">.379|.370</td><td class="num">.059|.018</td><td class="num">.276|.264</td><td class="num">.055|.080</td><td class="num">.196|.174</td></tr>
  </tbody></table>
  <p class="fig-note"><strong>On English the published result holds, and it holds against controls the paper never ran.</strong> Our Llama-2-70B lands within .02 of every published number, and the effect extends to a second family: Qwen3 scales 1.7B &rarr; 8B &rarr; 32B exactly as the paper's scaling claim predicts. The gap to the controls is large in <em>both</em> space and time (Llama-2-70B World .905 against its random twin .170; Art .860 against .029), so the geometry is learned rather than architectural, and TF-IDF is a genuine floor that trained models clear and random ones fall below. The three <strong>translation models</strong> (uMT5-base, cuneiform-400M, AKK-300M) sit <em>above</em> the random twins but <em>below</em> TF-IDF on English: they are not generically good probes, which matters when they win on Akkadian later. <strong>PLS tracks Ridge closely</strong> for every decoder, so the signal is genuinely low-rank rather than spread thinly over all dimensions. <em>* = control.</em></p>
</section>""",

"cellA_layers": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">A &middot; depth</div>
  <h2 class="sh">Where in the network space and time live</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">per-layer <strong>ridge</strong> probe (no PLS on this slide), all six English datasets pooled into two groups: <strong>SPACE</strong> = World, USA, NYC (predicting latitude/longitude) and <strong>TIME</strong> = Figures, Art, Headlines (predicting year). Plotted against <strong>normalised depth</strong> (layer / total layers) so models of different depth, 28 to 41 layers, are directly comparable, as in the paper's Figure 2.</div>
    <div class="cfg-k">Reading</div><div class="cfg-v">left = last token (the paper's read-out), right = mean pool (new). TIME = Spearman &rho;, SPACE = R&sup2; on a <strong>symlog</strong> axis, meaning linear near zero and logarithmic further out, so the deep negative scores of the failing arms fit on the page without squashing the 0 to 1 band where everything interesting happens. Dashed = random-init controls; &#9733; = each arm's best layer.</div>
  </div>
  <div class="fig-wrap">{{IMG:29}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>The Qwen and Llama families sit clearly above both the random controls and the encoder-decoder translation models. <strong>Where each arm peaks is as informative as how high it peaks:</strong> for the weakest arms the best layer is at the very start of the network, which is what "no signal was built" looks like, whereas every arm that beats its controls peaks in the <strong>middle-to-late</strong> layers, reproducing the paper's depth profile for time under both poolings. In space the best layers are spread more evenly across depth and sit less often at the very start. Two arms are the exception worth watching in the space panels: they track near the bottom for most of the network under both poolings and then <strong>rise sharply in the last few layers</strong>.</div>
</section>""",

"cellA_pls": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">A &middot; dimensionality</div>
  <h2 class="sh">How many PLS directions the world model needs</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">at each arm's <strong>best ridge layer</strong> from the previous slide, refit with <strong>PLS</strong> using k = 1 to 64 components. Same six English datasets in the same two groups: <strong>SPACE</strong> = World, USA, NYC (latitude/longitude), <strong>TIME</strong> = Figures, Art, Headlines (year).</div>
    <div class="cfg-k">Reading</div><div class="cfg-v">TIME = Spearman &rho;, SPACE = R&sup2;; both poolings shown; dashed = random-init controls; &#9733; = the k that maximises the score.</div>
  </div>
  <div class="fig-wrap">{{IMG:30}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>Most arms settle at around <strong>k &asymp; 16</strong> components. Year is the more concentrated of the two, space is more spread out, but both converge near 16. That is intuitively what the strong scores require: with only a handful of directions there would be no room for a meaningful subspace, and we would expect the weak results the <strong>random-init controls</strong> show, which saturate by k &asymp; 3 to 5 and gain nothing after that. The learned representation is genuinely multi-dimensional rather than one strong axis, a distinction that a single best-layer R&sup2; table hides completely.</div>
</section>""",

"b_entity": """<section class="slide slide-text">
  <div class="eyebrow">Cell B &middot; entity level &middot; the paper's own protocol</div>
  <h2 class="sh">Step 1 &mdash; hold the language fixed, swap famous entities for obscure ones</h2>
  <div class="cfg">
    <div class="cfg-k">Setup</div><div class="cfg-v">two datasets built to mirror theirs exactly: <strong>assyrian_ruler</strong> (34 rulers &rarr; year) mirrors <em>historical_figure</em>, and <strong>mesopotamian_place</strong> (25 find-spots &rarr; lon/lat) mirrors <em>world_place</em>. Each entity appears once <strong>bare</strong> (the paper-faithful row) and inside five neutral carrier sentences that never mention a date or region.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v">four sites, so the paper's protocol and ours are both visible: <strong>ent_last</strong> (last token of the ruler/place name &mdash; theirs), <strong>ent_mean</strong>, <strong>last</strong> (last token of the whole sentence &mdash; their <em>headline</em> protocol) and <strong>mean</strong>. On the bare rows the entity <em>is</em> the string, so ent_last = last by construction.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">ridge and PLS-5, <strong>R&sup2;</strong> and <strong>&rho;</strong>, over a <strong>200-draw Monte-Carlo of entity-level splits</strong> (20% of entities held out per draw; all six templates of an entity always move together, so no template can leak its target).</div>
  </div>
  <p class="tbl-cap">bare entity string &mdash; year &rho; (34 rulers) &middot; geo &rho; (25 find-spots), MC mean</p>
  <table class="rtbl compact"><thead><tr><th rowspan="2">model</th><th colspan="2" class="num">YEAR &mdash; ruler names</th><th class="num">GEO</th></tr><tr><th class="num">entity-last (theirs)</th><th class="num">mean (ours)</th><th class="num">entity-last</th></tr></thead><tbody>
    <tr><td><span class="mdl">Llama-2-70B</span></td><td class="num"><strong>.701</strong></td><td class="num">.463</td><td class="num">.429</td></tr>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.663</td><td class="num">.438</td><td class="num">.413</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">.627</td><td class="num">.436</td><td class="num">.458</td></tr>
    <tr><td><span class="mdl">Llama-2-13B</span></td><td class="num">.618</td><td class="num">.483</td><td class="num">.441</td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.596</td><td class="num">.500</td><td class="num">.344</td></tr>
    <tr><td><span class="mdl">Llama-2-7B</span></td><td class="num">.527</td><td class="num">.477</td><td class="num">.384</td></tr>
    <tr><td><span class="mdl">cuneiform-400M</span></td><td class="num">.456</td><td class="num">.509</td><td class="num">.398</td></tr>
    <tr><td><span class="mdl">AKK-300M</span></td><td class="num">.488</td><td class="num">.567</td><td class="num">.315</td></tr>
    <tr class="rand"><td><span class="mdl">Llama-2-70B random</span>*</td><td class="num">.457</td><td class="num">.314</td><td class="num">.459</td></tr>
    <tr class="rand"><td><span class="mdl">Llama-2-7B random</span>*</td><td class="num">.473</td><td class="num">.311</td><td class="num">.278</td></tr>
    <tr class="rand"><td><span class="mdl">random Qwen3-8B</span>*</td><td class="num">.171</td><td class="num">&minus;.038</td><td class="num">.321</td></tr>
    <tr class="rand"><td><span class="mdl">TF-IDF</span> (floor)</td><td class="num">.344</td><td class="num">.344</td><td class="num">.296</td></tr>
  </tbody></table>
  <div class="text-points" style="margin-top:9px">
    <div class="tp"><div class="tp-h">Time survives obscurity &mdash; but only at the top of the ladder, and only with their pooling</div><div class="tp-b">Llama-2-70B (&rho; <strong>.701</strong>) clears both gates: its own random twin (.457) and the floor (.344), and the ladder orders monotonically (70B &gt; 13B &gt; 7B, 32B &gt; 8B &gt; 1.7B). But the margin is <strong>&asymp;.24</strong> where cell A's was <strong>&asymp;.74</strong> in R&sup2;, and by Llama-2-7B (.527 vs its twin .473) the gap is inside the spread. <strong>Mean pooling erases the effect entirely</strong> &mdash; every decoder falls to &asymp;.44&ndash;.50 and the ordering inverts, with AKK-300M on top. On a bare name the paper's entity-last-token choice is doing real work.</div></div>
    <div class="tp"><div class="tp-h">Space does not survive at all</div><div class="tp-b">No arm beats its twin on find-spots: the best score in the whole geo column belongs to <strong>random-init Llama-2-70B (.459)</strong>, and R&sup2; is negative for every arm including TF-IDF. Recovering coordinates for Nineveh or Borsippa from the name alone is simply not something these models do &mdash; which makes the fragment-level geo result two slides on (where trained arms <em>do</em> clear the floor) the more surprising of the two.</div></div>
  </div>
  <p class="fig-note">34 rulers and 25 find-spots against the paper's thousands, so a 20% split holds out 6&ndash;7 entities and the MC spread is wide (&plusmn;.21&ndash;.39 on &rho;). Read the <strong>ordering against the two controls</strong>, not the third decimal. Splits are by entity, so all six templates of a ruler always move together. <em>* = control.</em></p>
</section>""",

"confounder": """<section class="slide slide-text">
  <div class="eyebrow">Cell C &middot; the controls this corpus forces on us</div>
  <h2 class="sh">Before we read cell C: what &ldquo;year&rdquo; means when eight rulers carry the corpus</h2>
  <div class="exp-config">Our year label is almost a ruler label: <strong>eight rulers account for most dated fragments</strong>, and every fragment of a ruler shares one year. A probe can therefore score well by recognising a king's name &mdash; a spelling-recognition task, not a chronology. Three devices separate the two, and each one costs signal.</div>
  <p class="tbl-cap">Akkadian maximal &middot; r8 &middot; year &mdash; the same arms under three protocols (&rho;)</p>
  <table class="rtbl compact"><thead><tr><th>arm</th><th class="num">hold-out &rho;</th><th class="num">balanced MC &rho;</th><th class="num">leave-one-ruler-out &rho;</th></tr></thead><tbody>
    <tr class="rand"><td><span class="mdl">TF-IDF</span> (floor)</td><td class="num">.793</td><td class="num">.707</td><td class="num">.129</td></tr>
    <tr><td><span class="mdl">Llama-2-70B</span></td><td class="num">.596</td><td class="num">.331</td><td class="num">.024</td></tr>
    <tr><td><span class="mdl">Llama-2-7B</span></td><td class="num">.591</td><td class="num">.433</td><td class="num">&minus;.067</td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.533</td><td class="num">.396</td><td class="num">.065</td></tr>
    <tr class="rand"><td><span class="mdl">Llama-2-7B random</span>*</td><td class="num">.495</td><td class="num">.438</td><td class="num">&minus;.015</td></tr>
    <tr class="rand"><td><span class="mdl">random Qwen3-8B</span>*</td><td class="num">.482</td><td class="num">.499</td><td class="num">.048</td></tr>
  </tbody></table>
  <div class="text-points" style="margin-top:10px">
    <div class="tp"><div class="tp-h">What the three columns do</div><div class="tp-b"><strong>maximal cleaning</strong> strips royal names and titles from the text; <strong>balanced Monte-Carlo</strong> caps each of the 8 rulers at 21 fragments over 200 draws, removing frequency imbalance; <strong>LORO</strong> holds out an entire ruler, so the probe must date a king it has never seen.</div></div>
    <div class="tp"><div class="tp-h">The verdict, and why every later slide uses these devices</div><div class="tp-b">Under plain hold-out everything looks respectable &mdash; and TF-IDF already leads at &rho; .793, which is the tell. Balanced MC pulls the trained arms down <em>onto their random twins</em> (Llama-7B .433 vs its own random .438). Under LORO <strong>every arm collapses to &asymp; 0</strong>. The hold-out number was ruler-identity memorisation from surface spelling, not a timeline. <em>* = control.</em></div></div>
  </div>
</section>""",

"ruler_not_chrono": """<section class="slide slide-text">
  <div class="eyebrow">Cell C &middot; what the entity-token probe actually learns</div>
  <h2 class="sh">The king-name token is nearly perfect at identity &mdash; and useless for chronology</h2>
  <div class="exp-config">Slide 12 showed the king-name token scoring far above whole-fragment pooling. This slide asks what that score <em>is</em>. Three read-outs on the same probe: can it name the ruler, can it order years <em>across</em> rulers, and can it order years <em>within</em> a ruler's own fragments.</div>
  <p class="tbl-cap">king-name last token &mdash; ruler macro-F1 (chance .20) &middot; year &rho; pooled &middot; year &rho; within ruler</p>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">ruler F1</th><th class="num">year &rho; (pooled)</th><th class="num">year &rho; (within-ruler)</th></tr></thead><tbody>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.989</td><td class="num">.974</td><td class="num">&minus;.055</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">.982</td><td class="num">.977</td><td class="num">&minus;.190</td></tr>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.982</td><td class="num">.967</td><td class="num">&minus;.550</td></tr>
    <tr><td><span class="mdl">cuneiform-400M</span></td><td class="num">.943</td><td class="num">.957</td><td class="num">&minus;.148</td></tr>
    <tr class="rand"><td><span class="mdl">random Qwen3-8B</span>*</td><td class="num">.946</td><td class="num">.926</td><td class="num">.195</td></tr>
  </tbody></table>
  <div class="text-points" style="margin-top:10px">
    <div class="tp"><div class="tp-h">The control settles it</div><div class="tp-b">The <strong>random-initialised</strong> Qwen3-8B reaches ruler F1 <strong>.946</strong> and pooled year &rho; <strong>.926</strong> &mdash; matching every trained model, including the 120B. Distinguishing eight fixed spellings needs no learned chronology at all; an untrained projection of the token identity suffices.</div></div>
    <div class="tp"><div class="tp-h">And the third column is the one that matters</div><div class="tp-b">Once ruler identity is held constant, the correlation is <strong>zero or negative for every arm</strong> (&minus;.55 to +.20). The probe has learned <em>which king</em>, and the year comes along for free because each king has one year. That is a lookup table, not a temporal representation &mdash; and it is exactly why LORO (previous slide) goes to zero.</div></div>
  </div>
</section>""",

"ask_directly": """<section class="slide slide-text">
  <div class="eyebrow">Rescue 2 &middot; behavioural, not activation-based</div>
  <h2 class="sh">Ask the model instead of probing it &mdash; and prompt it as hard as you like</h2>
  <div class="cfg">
    <div class="cfg-k">Setup</div><div class="cfg-v">Two ways of removing the probe from the loop. <strong>T12</strong>: show the fragment to a chat model under four prompt styles (bare / expert framing / few-shot k=5 / chain-of-thought) and take its <strong>generated answer</strong> (ruler + year BCE, forced single guess) as the prediction. <strong>T10</strong>: keep the probe but read activations from inside those same four prompts.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">Spearman(prediction, true) on the same 200 balanced draws, maximal cleaning throughout.</div>
  </div>
  <p class="tbl-cap">T12 &mdash; the generated answer as the prediction (&rho;)</p>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">bare</th><th class="num">expert</th><th class="num">few-shot</th><th class="num">CoT</th></tr></thead><tbody>
    <tr><td><span class="mdl">Qwen3-1.7B</span></td><td class="num">.010</td><td class="num">.123</td><td class="num">.009</td><td class="num">&minus;.074</td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.173</td><td class="num">.029</td><td class="num">.374</td><td class="num">.125</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">&minus;.045</td><td class="num">.159</td><td class="num">.304</td><td class="num">.216</td></tr>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.168</td><td class="num">.299</td><td class="num">.239</td><td class="num">.217</td></tr>
  </tbody></table>
  <p class="tbl-cap" style="margin-top:8px">T10 &mdash; probing the prompted activations (&rho;, PLS best-k)</p>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">bare</th><th class="num">expert</th><th class="num">few-shot</th><th class="num">CoT</th></tr></thead><tbody>
    <tr><td><span class="mdl">Qwen3-1.7B</span></td><td class="num">.313</td><td class="num">.313</td><td class="num">.287</td><td class="num">.314</td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.284</td><td class="num">.283</td><td class="num">.285</td><td class="num">.285</td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">.310</td><td class="num">.328</td><td class="num">.332</td><td class="num">.327</td></tr>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.310</td><td class="num">.318</td><td class="num">.333</td><td class="num">.320</td></tr>
  </tbody></table>
  <p class="fig-note">Neither escape route works. <strong>Generated answers are far worse than the probe</strong> (&rho; &le; .37 against the probe's &asymp; .33 baseline, and often near zero or negative) &mdash; the models will name Assyrian kings fluently in conversation, but cannot convert a stripped fragment into a date. And <strong>prompting moves nothing</strong>: across four styles and four scales the probed values sit within &asymp;.05 of bare, with no ordering by model size. The signal is not being hidden by a bad read-out; it is not there.</p>
</section>""",

"conditions": """<section class="slide slide-text">
  <div class="eyebrow">Synthesis</div>
  <h2 class="sh">What a linear temporal world model needs in order to exist</h2>
  <div class="exp-config">Reading the ladder back down, the collapse is attributable, and the failed rescues say what is <em>not</em> responsible.</div>
  <div class="text-points">
    <div class="tp"><div class="tp-h">Condition 1 &mdash; the language must be well represented in training</div><div class="tp-b">This is the load-bearing factor. Holding the entities fixed and moving English &rarr; Akkadian (B &rarr; C) drops the trained arms <strong>onto their own random twins</strong>, while a character n-gram floor beats all of them. The geometry the paper found does not transfer to a language the model barely saw.</div></div>
    <div class="tp"><div class="tp-h">Condition 2 &mdash; the entities must be salient enough to have been written about</div><div class="tp-b">A &rarr; B weakens the signal before any language change, and the n-gram floor starts to lead on time. A linear timeline needs entities the model has read <em>about</em>, repeatedly, in dated context &mdash; not merely entities it can spell.</div></div>
    <div class="tp"><div class="tp-h">What does <em>not</em> substitute for either condition</div><div class="tp-b"><strong>Scale</strong> (gpt-oss-120B lands mid-pack on English and near-random on Akkadian; the ladder is flat from 1.7B to 120B) &middot; <strong>prompting</strong> (four styles, no movement) &middot; <strong>asking directly</strong> &middot; <strong>more Akkadian via next-token finetuning</strong> (&Delta;&rho; &asymp; 0 at every scale and unfreezing depth) &middot; <strong>a stronger probe</strong> (kernel and geodesic PLS, and the supervision dial, buy nothing).</div></div>
    <div class="tp"><div class="tp-h">And what the &ldquo;signal&rdquo; in cell C actually was</div><div class="tp-b">Ruler-identity memorisation from surface spelling: near-perfect at naming the king, <strong>zero once ruler identity is held constant</strong>, and matched by an untrained network. Any world-model claim on a corpus like this must report a within-entity read-out; a pooled correlation does not distinguish a timeline from a lookup table.</div></div>
  </div>
</section>""",

"contributions": """<section class="slide slide-text">
  <div class="eyebrow">Contributions and discussion</div>
  <h2 class="sh">What we contribute, and what it says about world-model claims</h2>
  <div class="text-points">
    <div class="tp"><div class="tp-h">1 &mdash; A decomposition of a general claim</div><div class="tp-b">The salience &times; language-resource matrix turns &ldquo;LLMs represent space and time&rdquo; into a testable, attributable statement. Cell B is the control that makes the attribution possible, and cell D's emptiness is itself a finding about what can be asked of an ancient corpus.</div></div>
    <div class="tp"><div class="tp-h">2 &mdash; The controls the original setting lacked</div><div class="tp-b">Random-initialised twins and an n-gram floor, applied to every arm in every cell. On English they confirm the paper (trained .905 vs random .170 on world places); on Akkadian they overturn the apparent signal. The same probe, the same corpus &mdash; only the control tells them apart.</div></div>
    <div class="tp"><div class="tp-h">3 &mdash; A confounder-controlled benchmark for Akkadian chronology</div><div class="tp-b">Maximal cleaning, balanced Monte-Carlo, by-site geo splits and leave-one-ruler-out, on a cleaned royal-inscription corpus we intend to release &mdash; along with the finding that without those devices a bag of character n-grams looks like a world model.</div></div>
    <div class="tp"><div class="tp-h">4 &mdash; A positive result, and the mechanism behind it</div><div class="tp-b">A <strong>400M multilingual translation encoder</strong> outperforms every LLM up to 120B at dating Akkadian, and the same-size ablations locate the cause in the <strong>translation objective</strong> rather than size, tokenizer, or Akkadian exposure. Where pretraining scale cannot build the map, learning to map a language onto meaning apparently can.</div></div>
    <div class="tp"><div class="tp-h">Open</div><div class="tp-b">Does the encoder's advantage survive leave-one-ruler-out, or is it the same lookup table in a smaller package? That is the experiment that decides whether this is a dating system or a second cautionary tale.</div></div>
  </div>
</section>""",
}

# eyebrow rewrites so reused slides announce their act
EYEBROW_PATCHES = {
    24: "Cell A &middot; the paper's own setting &middot; salient entities, English",
    29: "Cell A &middot; depth",
    30: "Cell A &middot; dimensionality",
    25: "Cell B &middot; fragment level &middot; English gloss &middot; balanced Monte-Carlo",
    26: "Cell B &middot; fragment level &middot; English gloss &middot; by-site geo",
    15: "Cell C &middot; entity level &middot; the king-name token",
    27: "Cell C &middot; fragment level &middot; raw Akkadian &middot; balanced Monte-Carlo",
    28: "Cell C &middot; fragment level &middot; raw Akkadian &middot; by-site geo",
    31: "Cell C &middot; depth &middot; the Akkadian mirror",
    32: "Cell C &middot; dimensionality &middot; the Akkadian mirror",
    13:  "Rescue 1 &middot; does it already know?",
    7:   "Rescue 3 &middot; scale and continued pretraining",
    19:  "Rescue 4 &middot; word order",
    20:  "Rescue 5 &middot; non-linear probes",
    21:  "Rescue 6 &middot; supervision",
    4:   "The positive result",
    9:   "Mechanism &middot; the objective, isolated",
    8:   "Mechanism &middot; ruling out the tokenizer",
}


EXTRA_CSS = """
/* --- added by build_story_deck.py --- */
.rtbl.matrix{font-size:15px;margin-top:6px;}
.rtbl.matrix th,.rtbl.matrix td{padding:13px 18px;vertical-align:top;line-height:1.45;}
.rtbl.matrix thead th{font-size:12px;letter-spacing:.05em;}
.rtbl.matrix td:first-child{white-space:nowrap;}
.takeaway.tight{font-size:12.5px;line-height:1.45;padding:9px 15px;}
.takeaway.tight .tk-label{font-size:9px;}
.cfg.tight{font-size:11.5px;gap:3px 11px;margin-bottom:8px;}
.cfg.tight .cfg-k{font-size:9px;}
.slide.fig-major .sh{font-size:23px;margin-bottom:9px;}
.slide.fig-major .eyebrow{margin-bottom:6px;}
.slide.fig-major .fig-wrap{margin:0 0 8px;}
</style>"""


def inject_css(head):
    return head.replace("</style>", EXTRA_CSS, 1)


def image_of(source_slides, idx, alt=None):
    """The <img> tag from a source slide, so a rewritten slide keeps its figure.
    The old alt text is replaced (it still carried the old slide's title)."""
    m = re.search(r'<img[^>]*>', source_slides[idx])
    if not m:
        raise SystemExit(f"slide {idx} has no <img> to reuse")
    tag = re.sub(r'\s*alt="[^"]*"', '', m.group(0))
    return tag.replace('<img', f'<img alt="{alt or "figure"}"', 1)


def parse_slides(html):
    out = {}
    for m in re.finditer(
            r'<section class="slide[^"]*" data-index="(\d+)">.*?</section>', html, re.S):
        out[int(m.group(1))] = m.group(0)
    return out


def strip_index(section):
    return re.sub(r'(<section class="slide[^"]*)" data-index="\d+"', r'\1"', section, count=1)


def set_eyebrow(section, text):
    if '<div class="eyebrow">' in section:
        return re.sub(r'<div class="eyebrow">.*?</div>',
                      f'<div class="eyebrow">{text}</div>', section, count=1, flags=re.S)
    # slides that never had one: insert after the opening tag
    return re.sub(r'(<section class="slide[^"]*">\n)',
                  rf'\1  <div class="eyebrow">{text}</div>\n', section, count=1)


def build(html):
    old = parse_slides(html)
    missing = [ref for kind, ref, _ in SPINE
               if (kind == "old" and ref not in old)
               or (kind == "new" and ref not in NEW_SLIDES)]
    if missing:
        raise SystemExit(f"spine references unknown slides: {missing}")

    bodies, titles = [], []
    for i, (kind, ref, title) in enumerate(SPINE):
        sec = strip_index(old[ref]) if kind == "old" else NEW_SLIDES[ref]
        for tok in set(re.findall(r'\{\{IMG:(\d+)\}\}', sec)):
            sec = sec.replace("{{IMG:%s}}" % tok, image_of(old, int(tok), alt=title))
        if kind == "old" and ref in EYEBROW_PATCHES:
            sec = set_eyebrow(sec, EYEBROW_PATCHES[ref])
        sec = re.sub(r'(<section class="slide[^"]*)"',
                     rf'\1" data-index="{i}"', sec, count=1)
        bodies.append(sec)
        titles.append(title)

    head = inject_css(html[:html.index('<section')])
    tail = html[html.rindex('</section>') + len('</section>'):]
    tail = re.sub(r'const TOTAL = \d+;', f'const TOTAL = {len(SPINE)};', tail)
    titles_js = ", ".join(
        '"' + t.replace("\\", "\\\\").replace('"', '\\"') + '"' for t in titles)
    tail = re.sub(r'const TITLES = \[.*?\];', f'const TITLES = [{titles_js}];',
                  tail, flags=re.S)
    return head + "\n".join(bodies) + tail


def check(html):
    n_sec = html.count("<section")
    idx = [int(x) for x in re.findall(r'data-index="(\d+)"', html)]
    total = int(re.search(r'const TOTAL = (\d+)', html).group(1))
    titles = re.search(r'const TITLES = \[(.*?)\];', html, re.S).group(1)
    n_titles = len(re.findall(r'"(?:[^"\\]|\\.)*"', titles))
    ok = (n_sec == total == n_titles == len(idx)) and idx == list(range(len(idx)))
    print(f"sections={n_sec} indices={'contiguous 0..%d' % (len(idx)-1) if idx == list(range(len(idx))) else idx} "
          f"TOTAL={total} titles={n_titles}  ->  {'OK' if ok else 'MISMATCH'}")
    return ok


def read_source(path=None):
    """The 33-slide deck the spine refers to: a file if given, else the pinned blob."""
    if path:
        with open(path) as f:
            return f.read()
    return subprocess.run(["git", "cat-file", "blob", SOURCE_BLOB],
                          cwd=HERE, capture_output=True, text=True,
                          check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--source", default=None,
                    help="33-slide source deck (default: the pinned git blob)")
    ap.add_argument("--check", action="store_true", help="only verify the current deck")
    args = ap.parse_args()

    if args.check:
        with open(DECK) as f:
            raise SystemExit(0 if check(f.read()) else 1)

    new = build(read_source(args.source))
    with open(args.out, "w") as f:
        f.write(new)
    print(f"[write] {args.out}  ({len(new)/1e6:.2f} MB, {len(SPINE)} slides)")
    check(new)


if __name__ == "__main__":
    main()
