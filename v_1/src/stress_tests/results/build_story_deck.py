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
    ("new", "title", "When Are Space and Time Linearly Represented in Language Models?"),
    ("new", "motivation", "Why we care: a world model you could point at an excavation"),
    ("new", "paper", "Gurnee & Tegmark: a strong claim, tested in one cell"),
    ("new", "matrix", "The climbing map: entity salience x language resource"),
    ("new", "protocol", "How every experiment in this deck is set up and read"),
    ("new", "cellA_repro", "A: the paper reproduces on our models, with the controls it never ran"),
    ("new", "cellA_layers", "A: where in the network space and time live"),
    ("new", "cellA_pls", "A: how many PLS directions the world model needs"),
    ("new", "b_entity", "Obscure entities in English: the date survives, the place does not"),
    ("new", "b_frag_year", "Whole fragments in English: a bag of character n-grams dates them best"),
    ("new", "b_frag_geo", "Whole fragments in English: place is no better than an untrained network"),
    ("new", "mlm_model", "Our own Akkadian model: a small masked language model trained on the corpus"),
    ("new", "c_kingtoken", "Raw Akkadian: the king's name is readable, the chronology is not"),
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

# which matrix cell each slide sits in (spine position -> cell); "" = show the map
# with nothing active yet, as orientation.
CELLMAP_AT = {4: "", 5: "A", 6: "A", 7: "A", 8: "B", 9: "B", 10: "B",
              12: "C", 13: "C", 14: "C", 15: "C", 16: "C", 17: "C"}

# --------------------------------------------------------- newly authored ------
NEW_SLIDES = {

"title": """<section class="slide slide-title">
  <div class="title-inner">
    <div class="title-kicker">M.Sc. Thesis &middot; Advisor Meeting &middot; 2026</div>
    <h1 class="title-h1">When Are Space and Time Linearly Represented in Language Models?</h1>
    <div class="title-sub">Entity salience, language resource, and pooling</div>
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
    <div class="cfg-k">Metrics</div><div class="cfg-v">best-layer held-out <strong>R&sup2;</strong> (the paper's headline) and <strong>Spearman &rho;</strong> (which the paper also reports), for all six datasets. Every cell shows <strong>Ridge</strong> then <strong>PLS</strong>, PLS taken at its best k &le; 64 on the same layer.</div>
  </div>
  <p class="tbl-cap">best-layer held-out test score &middot; each cell = <strong>Ridge</strong>|<span style="color:#6b7484">PLS</span> &middot; PLS at its best k &le; 64 on the same layer</p>
  {{TABLE:cellA}}
  <p class="fig-note"><strong>On English the published result holds, and it holds against controls the paper never ran.</strong> Our Llama-2-70B lands within .02 of every published number, and the effect extends to a second family: Qwen3 scales 1.7B &rarr; 8B &rarr; 32B exactly as the paper's scaling claim predicts. The gap to the controls is large in <em>both</em> space and time (Llama-2-70B World .905 against its random twin .170; Art .860 against .029), so the geometry is learned rather than architectural, and TF-IDF is a genuine floor that trained models clear and random ones fall below. The three <strong>translation models</strong> (uMT5-base, cuneiform-400M, AKK-300M) sit <em>above</em> the random twins but <em>below</em> TF-IDF on English: they are not generically good probes. <strong>PLS tracks Ridge closely</strong> for every decoder. <em>* = control.</em></p>
</section>""",

"cellA_layers": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">A &middot; depth</div>
  <h2 class="sh">Where in the network space and time live</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">per-layer <strong>ridge</strong> probe (no PLS on this slide), all six English datasets pooled into two groups: <strong>SPACE</strong> = World, USA, NYC (predicting latitude/longitude) and <strong>TIME</strong> = Figures, Art, Headlines (predicting year). Plotted against <strong>normalised depth</strong> (layer / total layers) so models of different depth, 28 to 41 layers, are directly comparable, as in the paper's Figure 2.</div>
    <div class="cfg-k">Reading</div><div class="cfg-v">rows = SPACE then TIME; columns = <strong>last token</strong> (the paper's read-out) then <strong>mean pool</strong> (new), each shown under <strong>both metrics</strong>, so nothing here departs from the paper: we only add the read-outs it also reports. R&sup2; uses a <strong>symlog</strong> axis, linear near zero and logarithmic further out, so the deep negative scores of the failing arms fit on the page without squashing the 0 to 1 band. Dashed = random-init controls; &#9733; = each arm's best layer.</div>
  </div>
  <div class="fig-wrap">{{FIG:fig_cellA_layers.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>The Qwen and Llama families sit clearly above both the random controls and the encoder-decoder translation models. <strong>Where each arm peaks is as informative as how high it peaks:</strong> for the weakest arms the best layer is at the very start of the network, which is what "no signal was built" looks like, whereas every arm that beats its controls peaks in the <strong>middle-to-late</strong> layers, reproducing the paper's depth profile for time under both poolings. In space the best layers are spread more evenly across depth and sit less often at the very start. Two arms are the exception worth watching in the space panels: they track near the bottom for most of the network under both poolings and then <strong>rise sharply in the last few layers</strong>.</div>
</section>""",

"cellA_pls": """<section class="slide slide-figure fig-major">
  <div class="eyebrow">A &middot; dimensionality</div>
  <h2 class="sh">How many PLS directions the world model needs</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">at each arm's <strong>best ridge layer</strong> from the previous slide, refit with <strong>PLS</strong> using k = 1 to 64 components. Same six English datasets in the same two groups: <strong>SPACE</strong> = World, USA, NYC (latitude/longitude), <strong>TIME</strong> = Figures, Art, Headlines (year).</div>
    <div class="cfg-k">Reading</div><div class="cfg-v">rows = SPACE then TIME; columns = <strong>last token</strong> then <strong>mean pool</strong>, each under <strong>both metrics</strong>. Dashed = random-init controls; &#9733; = the k that maximises the score; the dash-dot vertical line marks k = 16.</div>
  </div>
  <div class="fig-wrap">{{FIG:fig_cellA_plsk.png}}</div>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span>Most arms settle at around <strong>k &asymp; 16</strong> components. Year is the more concentrated of the two, space is more spread out, but both converge near 16. That is intuitively what the strong scores require: with only a handful of directions there would be no room for a meaningful subspace, and we would expect the weak results the <strong>random-init controls</strong> show, which saturate by k &asymp; 3 to 5 and gain nothing after that. The learned representation is genuinely multi-dimensional rather than one strong axis, a distinction that a single best-layer R&sup2; table hides completely.</div>
</section>""",

"b_entity": """<section class="slide slide-text">
  <div class="eyebrow">B &middot; the same test as the paper, on our entities</div>
  <h2 class="sh">Obscure entities in English: the date survives, the place does not</h2>
  <div class="cfg tight">
    <div class="cfg-k">Setup</div><div class="cfg-v">we repeat the paper's own two experiments and change <strong>only who the entities are</strong>. For time, in place of its famous historical figures (slide 6) we use the <strong>34 Assyrian and Babylonian rulers</strong> attested in our corpus, spanning <strong>1132 to 261 BC</strong> (Ashurbanipal, Sennacherib, Sargon II), and predict each ruler's year. For space, in place of its world places we use the <strong>25 excavation sites</strong> our fragments come from (Nineveh, Babylon, Assur, Kalhu) and predict latitude and longitude.</div>
    <div class="cfg-k">Input</div><div class="cfg-v">every name is probed <strong>on its own</strong>, exactly as the paper does (<em>&ldquo;Ashurbanipal&rdquo;</em>), and also inside five short sentences that never mention a date or a region (<em>&ldquo;This tablet dates to the reign of Ashurbanipal.&rdquo;</em>). That mirrors the paper's own robustness check, where the same entity is probed under several surrounding prompts. The table reports the name-alone rows; adding the sentences moves the top scores by less than 0.01.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v"><strong>name, last token</strong>: the activation at the final token of the name, which is the paper's read-out. <strong>name, average</strong>: the activation averaged over all the name's tokens, which is our addition and the only option for the translation encoders, since they have no causal last token.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">Spearman &rho;, averaged over <strong>200 redrawn splits</strong> that hold out 20&#37; of the entities each time. Every cell is <strong>Ridge</strong>|<span style="color:#6b7484">PLS</span>.</div>
  </div>
  {{TABLE:entity}}
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>Time holds up, and the scaling law with it:</strong> reading the last token of the name, Llama-2-70B leads at &rho; .701, both families order by size, and the top arms clear their own random twin (.457) and the n-gram baseline (.344). But the margin is roughly a quarter of what it was on famous entities, and by Llama-2-7B (.527 against its twin .473) it is inside the noise. We also tried a second read-out, <strong>averaging the activation over all of the name's tokens</strong>, which the paper never uses: it lands every model between .40 and .57 on year and never separates from the random controls on either target, so it is left out of the table. <strong>Space fails outright</strong>: the best number in either place column belongs to an <em>untrained</em> Llama-2-70B (.459), so no model beats its control. 34 rulers and 25 sites means 6 to 7 held-out entities per draw, so read the ordering, not the third decimal.</div>
</section>""",

"b_frag_year": """<section class="slide slide-text">
  <div class="eyebrow">B &middot; from names to whole fragments</div>
  <h2 class="sh">Whole fragments in English: a bag of character n-grams dates them best</h2>
  <div class="cfg tight">
    <div class="cfg-k">Task</div><div class="cfg-v">predict <strong>the year a fragment was written</strong>. The entity is no longer a name but a <strong>whole tablet fragment</strong>, read in its English translation, so this is the paper's <em>news headlines</em> experiment carried over to our corpus: a full passage rather than a short name, with no entity marked anywhere in it. A fragment reads like <em>&ldquo;&hellip; the palace of my lordship which is in Nineveh I rebuilt and completed &hellip;&rdquo;</em>.</div>
    <div class="cfg-k">Data</div><div class="cfg-v">the corpus names <strong>40 rulers across 1193 dated fragments</strong>, but it is severely unbalanced: Ashurbanipal alone accounts for 268 and eighteen rulers have three fragments or fewer. We therefore restrict to the <strong>8 best-attested rulers</strong>, which is the panel an Assyriologist would actually trust and which still covers most of the corpus. Every ruler is then capped at the same number of fragments so frequency cannot stand in for chronology.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v"><strong>text, last token</strong>: the activation at the final token of the passage, which is what the paper uses for headlines. <strong>text, average</strong>: averaged over the whole passage. Nothing points the model at a date, so the signal has to be recoverable from the passage as a whole.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">Spearman &rho; and R&sup2;, averaged over <strong>200 balanced draws</strong> that cap each of the 8 best-attested rulers at the same number of fragments, so no single king can carry the score.</div>
  </div>
  {{TABLE:frag:eng_tier0:year}}
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>The n-gram baseline wins outright</strong> (&rho; .775), ahead of every embedding in the table. Pooling matters more than the model does: averaging over the passage adds about <strong>+.20 &rho;</strong> to almost every arm, which is the opposite of what we saw on bare names, because a date leaves its trace across the whole passage rather than at one token. But the arms that gain the most from it gain nothing <em>over their controls</em>: at the top, AKK-300M (.740) and Qwen3-8B (.737) sit barely above an <strong>untrained</strong> Llama-2-70B (.661) and an untrained Qwen3-8B (.636). Moving from famous names to obscure passages, the scaling law is gone and the models no longer separate from noise.</div>
</section>""",

"b_frag_geo": """<section class="slide slide-text">
  <div class="eyebrow">B &middot; from names to whole fragments</div>
  <h2 class="sh">Whole fragments in English: place is no better than an untrained network</h2>
  <div class="cfg tight">
    <div class="cfg-k">Task</div><div class="cfg-v">predict <strong>where the tablet was dug up</strong>, its latitude and longitude, from the same fragments and the same English translations as the previous slide. This is the paper's world-place experiment moved to whole passages: the find-spot is <em>never named in the text</em>, so unlike a city name there is nothing to look up and the coordinates have to come from whatever the passage implies.</div>
    <div class="cfg-k">Data</div><div class="cfg-v">1068 fragments whose find-spot is known, grouped into the <strong>10 excavation sites</strong> with enough material to hold one out, again capped equally so that a large site cannot carry the score. The year slide restricts by ruler; this one restricts by site, because the confound here is which dig a fragment came from rather than which king wrote it.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v">the same two read-outs: <strong>text, last token</strong> and <strong>text, average</strong>.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">Spearman &rho; and R&sup2; over <strong>200 draws that hold out whole find-spots</strong>, capping each of the 10 merged sites equally, so a probe cannot succeed by memorising which site a fragment came from.</div>
  </div>
  {{TABLE:frag:eng_tier0:geo}}
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>Every trained model is matched by its own untrained twin.</strong> Averaging over the passage, the best arm is cuneiform-400M (&rho; .640, R&sup2; .445), but an <em>untrained</em> Llama-2-7B reaches .606 / .463 and an untrained Qwen3-8B .587 / .450, so the reading rule is not satisfied anywhere in this table. As with the year, averaging beats last-token pooling for nearly every arm, and again the gain is shared with the controls rather than earned by training. The n-gram baseline is the interesting exception: it ranks well on &rho; (.535) yet collapses on R&sup2; (.022), meaning it orders sites roughly but cannot place them.</div>
</section>""",

"mlm_model": """<section class="slide slide-text">
  <div class="eyebrow">C &middot; the model we built for this language</div>
  <h2 class="sh">Before testing on Akkadian: we trained our own small model on the corpus</h2>
  <div class="text-points">
    <div class="tp"><div class="tp-h">Why an extra model at all</div><div class="tp-b">Every arm so far was trained by someone else, on text that is overwhelmingly English. To ask whether Akkadian itself can support a linear timeline, we need at least one model whose entire experience <em>is</em> Akkadian. It also gives the comparison a floor from the other direction: if a model trained only on this language cannot find the signal either, the problem is not simply that the big models never saw enough Akkadian.</div></div>
    <div class="tp"><div class="tp-h">The objective, and where it comes from</div><div class="tp-b"><strong>Masked language modelling</strong>, not next-token prediction, following <em>Filling the Gaps in Ancient Akkadian Texts</em> (Fetaya et al., EMNLP 2021), which showed that restoring a damaged tablet <em>is</em> the masked-token task and that bidirectional context matters when the neighbouring signs are broken too. The same instinct drives <em>Ithaca</em> and its successor in Nature (2025) for Greek and Latin: read the whole context, then fill the gap.</div></div>
    <div class="tp"><div class="tp-h">What we trained</div><div class="tp-b">A <strong>37M-parameter encoder</strong> (16 layers, d = 384, RoPE, RMSNorm pre-norm), masked at 15&#37;, on <strong>2.45M words / 4.9M signs</strong> pooled from ORACC, eBL and Archibab, split by fragment so no tablet crosses train and test. Text is tokenised at the <strong>sign level</strong>, following the EvaCun 2025 shared task on lemmatisation and token prediction for cuneiform, which keeps our units comparable to the current benchmark for this language.</div></div>
    <div class="tp"><div class="tp-h">How to read it in the tables that follow</div><div class="tp-b">It appears as <strong>MLM</strong>. It is small, it is the only arm that is Akkadian all the way down, and it is <em>not</em> a translation model, which makes it the right comparison for the cuneiform and AKK encoders later: those two also see Akkadian, but through a translation objective.</div></div>
  </div>
</section>""",

"c_kingtoken": """<section class="slide slide-text">
  <div class="eyebrow">C &middot; raw Akkadian &middot; entity token against whole fragment</div>
  <h2 class="sh">In raw Akkadian the king's name is easy to read and the chronology is not</h2>
  <div class="cfg tight">
    <div class="cfg-k">The step</div><div class="cfg-v">we now change the <strong>language</strong>, holding the entities fixed: the same royal inscriptions, read in <strong>Akkadian</strong> rather than in translation. Two read-outs, mirroring the two we have used throughout: the <strong>ruler's name token</strong>, which is the closest thing this corpus has to the paper's marked entity, and the <strong>average over the whole fragment</strong>, where nothing is marked and the date must be recoverable from the passage as a whole.</div>
    <div class="cfg-k">Cleaning</div><div class="cfg-v">the fragment column uses our <strong>maximal</strong> regime, an eleven-filter pipeline plus truncation to 30 words. It strips digits, logograms (all-capital tokens), determinatives, case endings and plural markers, normalises long vowels, and lowercases everything. The reason is measured, not stylistic: without it a bag of character n-grams reaches 99&#37; accuracy by reading <em>document length and royal-name spellings</em>, because well-preserved eras leave long inscriptions and poorly-preserved ones leave scraps. Truncation removes the length crutch, the filters remove the name crutch. The name column deliberately keeps the names, since that is what it is measuring.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">Spearman &rho; over 200 balanced draws (8 rulers, 21 fragments each, grouped by ruler within every draw). Each cell is <strong>PLS</strong>|<span style="color:#6b7484">Ridge</span>. A shuffled-label null sits at about 0.01.</div>
  </div>
  <table class="rtbl compact"><thead><tr><th>model</th><th class="num">whole fragment, average &nbsp;(names stripped)</th><th class="num">the ruler's name token &nbsp;(names kept)</th></tr></thead><tbody>
    <tr><td><span class="mdl">gpt-oss-120B</span></td><td class="num">.316<i>|.273</i></td><td class="num">.645<i>|.153</i></td></tr>
    <tr><td><span class="mdl">Qwen3-32B</span></td><td class="num">.332<i>|.302</i></td><td class="num">.645<i>|.425</i></td></tr>
    <tr><td><span class="mdl">Qwen3-8B</span></td><td class="num">.339<i>|.111</i></td><td class="num">.480<i>|.466</i></td></tr>
    <tr><td><span class="mdl">Qwen3-1.7B</span></td><td class="num">.334<i>|.072</i></td><td class="num">.606<i>|.500</i></td></tr>
    <tr><td><span class="mdl">cuneiform-400M</span></td><td class="num">.391<i>|.339</i></td><td class="num">.513<i>|.139</i></td></tr>
    <tr><td><span class="mdl">AKK-300M</span></td><td class="num">.300<i>|.308</i></td><td class="num">.688<i>|.459</i></td></tr>
    <tr><td><span class="mdl">uMT5-base</span></td><td class="num">.277<i>|.273</i></td><td class="num">.423<i>|.407</i></td></tr>
    <tr><td><span class="mdl">MLM</span> (ours, 37M)</td><td class="num">.286<i>|.288</i></td><td class="num">.704<i>|.649</i></td></tr>
    <tr class="rand"><td><span class="mdl">TF-IDF</span></td><td class="num">.271<i>|.269</i></td><td class="num">&ndash;</td></tr>
    <tr class="rand"><td><span class="mdl">random Qwen3-8B</span></td><td class="num">.293<i>|.226</i></td><td class="num">.643<i>|.495</i></td></tr>
  </tbody></table>
  <div class="takeaway tight"><span class="tk-label">Key takeaway</span><strong>The two columns are measuring different things, and only one of them is chronology.</strong> On the whole fragment every arm lands between .27 and .39, on top of the n-gram baseline (.271) and of an <em>untrained</em> Qwen3-8B (.293): in raw Akkadian, with the names stripped, nothing separates from noise. The name column looks far better, but the same untrained network scores <strong>.643</strong> there, matching every trained model and beating most of them, so that column is reading which king is named rather than when he ruled. Averaging over the ruler's name tokens, the third read-out, collapses to near zero for everyone and is left out. This is the same pattern as English at entity level, one step further down: obscurity weakened the signal, and switching to the low-resource language removes what was left.</div>
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
.rtbl.compact.wide{font-size:9.5px;}
.rtbl.compact.wide th,.rtbl.compact.wide td{padding:1.5px 3px;}
.rtbl.compact.wide td i{font-style:normal;color:var(--ink-light);}
.rtbl.compact.wide thead tr:first-child th{border-bottom:none;padding-bottom:0;}
.rtbl.matrix{font-size:15px;margin-top:6px;}
.rtbl.matrix th,.rtbl.matrix td{padding:13px 18px;vertical-align:top;line-height:1.45;}
.rtbl.matrix thead th{font-size:12px;letter-spacing:.05em;}
.rtbl.matrix td:first-child{white-space:nowrap;}
.takeaway.tight{font-size:12.5px;line-height:1.45;padding:9px 15px;}
.takeaway.tight .tk-label{font-size:9px;}
.cfg.tight{font-size:11.5px;gap:3px 11px;margin-bottom:8px;}
.cfg.tight .cfg-k{font-size:9px;}
/* figure-first slides: shrink every non-figure block so the plot gets the page */
.slide.fig-major{padding:26px 30px 30px;}
.slide.fig-major .sh{font-size:19.5px;line-height:1.15;margin-bottom:5px;}
.slide.fig-major .eyebrow{font-size:9.5px;margin-bottom:4px;}
.slide.fig-major .cfg.tight{font-size:10px;line-height:1.35;gap:2px 9px;margin-bottom:5px;}
.slide.fig-major .cfg.tight .cfg-k{font-size:8px;}
.slide.fig-major .fig-wrap{margin:0 0 5px;}
.slide.fig-major .fig-wrap img{max-height:100%;width:auto;border:none;}
.slide.fig-major .takeaway.tight{font-size:10.5px;line-height:1.38;padding:6px 12px;}
.slide.fig-major .takeaway.tight .tk-label{font-size:8px;margin-right:7px;}
.cellmap{position:absolute;top:15px;right:18px;display:grid;
         grid-template-columns:auto 34px 34px;grid-auto-rows:auto;gap:2px;
         font-family:var(--sans);z-index:5;}
.cm-h{font-size:7.5px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;
      color:var(--ink-light);text-align:center;padding-bottom:1px;}
.cm-r{font-size:7.5px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;
      color:var(--ink-light);text-align:right;padding-right:3px;align-self:center;}
.cm-c{width:34px;height:22px;border:1px solid var(--border);border-radius:3px;
      display:flex;align-items:center;justify-content:center;
      font-size:11px;font-weight:800;color:var(--ink-light);background:#fbfcfd;}
.cm-c.on{background:var(--green);border-color:var(--green);color:#fff;
         box-shadow:0 1px 5px rgba(26,92,58,.35);}
.cm-c.done{background:var(--green-bg);border-color:var(--green-mid);color:var(--green);}
.cm-c.na{background:#f4f4f6;color:#c3c7d0;border-style:dashed;}
</style>"""


WM = os.path.abspath(os.path.join(HERE, "..", "..", "world_models"))
DATASETS = [("world_place", "World"), ("us_place", "USA"), ("nyc_place", "NYC"),
            ("historical_figure", "Figures"), ("art", "Art"), ("headline", "Headlines")]
TABLE_ROWS = [
    ("llama2_70b", "Llama-2-70B", False), ("llama2_13b", "Llama-2-13B", False),
    ("llama2_7b", "Llama-2-7B", False), ("gpt_oss_120b", "gpt-oss-120B", False),
    ("qwen3_32b", "Qwen3-32B", False), ("qwen3_8b", "Qwen3-8B", False),
    ("qwen3_1b7", "Qwen3-1.7B", False), ("umt5_base", "uMT5-base", False),
    ("thalesian_cunei400m", "cuneiform-400M", False),
    ("thalesian_akk300m", "AKK-300M", False),
    ("tfidf", "TF-IDF", True),
    ("llama2_70b_random", "Llama-2-70B random*", True),
    ("llama2_13b_random", "Llama-2-13B random*", True),
    ("llama2_7b_random", "Llama-2-7B random*", True),
    ("random", "random Qwen3-8B*", True),
]


def _fmt(v):
    if v is None or v != v:
        return "&ndash;"
    t = f"{v:.3f}"
    return t.lstrip("0") if t.startswith("0.") else t


def cellA_table():
    """Best-layer English results: every dataset x {R2, Spearman} x {Ridge | PLS}.
    Built from the committed CSV/JSON so the slide cannot drift from the results."""
    import csv
    import json as _json

    def load(name):
        with open(os.path.join(WM, "results", name)) as f:
            return {r["method"]: r for r in csv.DictReader(f)}
    r2, rho = load("summary_best_layer_r2.csv"), load("summary_best_layer_spearman.csv")

    pls = {}
    pls_dir = os.path.join(WM, "results", "eng_pls")
    for arm in os.listdir(pls_dir) if os.path.isdir(pls_dir) else []:
        pls[arm] = {}
        for ds, _ in DATASETS:
            f = os.path.join(pls_dir, arm, f"{ds}.last.json")
            if not os.path.exists(f):
                continue
            at = _json.load(open(f))["pls_at_best_layer"]
            pls[arm][ds] = (max(v["test_r2"] for v in at.values()),
                            max(v["test_spearman"] for v in at.values()))

    head = ('<table class="rtbl compact wide"><thead><tr><th rowspan="2">model</th>'
            + "".join(f'<th colspan="2" class="num">{lab}</th>' for _, lab in DATASETS)
            + '</tr><tr>'
            + "".join('<th class="num">R&sup2;</th><th class="num">&rho;</th>'
                      for _ in DATASETS) + '</tr></thead><tbody>')
    body = []
    for arm, label, ctrl in TABLE_ROWS:
        cells = []
        for ds, _ in DATASETS:
            gr = float(r2[arm][ds]) if arm in r2 else None
            gs = float(rho[arm][ds]) if arm in rho else None
            pr, ps = pls.get(arm, {}).get(ds, (None, None))
            cells.append(f'<td class="num">{_fmt(gr)}<i>|{_fmt(pr)}</i></td>'
                         f'<td class="num">{_fmt(gs)}<i>|{_fmt(ps)}</i></td>')
        tr = '<tr class="rand">' if ctrl else '<tr>'
        body.append(tr + f'<td><span class="mdl">{label}</span></td>'
                    + "".join(cells) + '</tr>')
    paper = ('<tr><td><span class="mdl">Llama-2-70B (paper)</span></td>'
             + "".join(f'<td class="num">{v}</td><td class="num">&ndash;</td>'
                       for v in [".911", ".864", ".359", ".835", ".885", ".746"])
             + '</tr>')
    return head + paper + "".join(body) + "</tbody></table>"



AKK = os.path.join(WM, "akkadian", "results")
ROWS_B = [
    ("llama2_70b", "Llama-2-70B", False), ("llama2_13b", "Llama-2-13B", False),
    ("llama2_7b", "Llama-2-7B", False), ("gpt_oss_120b", "gpt-oss-120B", False),
    ("qwen3_32b", "Qwen3-32B", False), ("qwen3_8b", "Qwen3-8B", False),
    ("qwen3_1b7", "Qwen3-1.7B", False), ("umt5_base", "uMT5-base", False),
    ("thalesian_cunei400m", "cuneiform-400M", False),
    ("thalesian_akk300m", "AKK-300M", False),
    ("tfidf", "TF-IDF", True),
    ("llama2_70b_random", "Llama-2-70B random", True),
    ("llama2_13b_random", "Llama-2-13B random", True),
    ("llama2_7b_random", "Llama-2-7B random", True),
    ("random", "random Qwen3-8B", True),
]


def _rp(ridge, pls):
    """One cell: ridge value, then the PLS value in grey."""
    return f'<td class="num">{_fmt(ridge)}<i>|{_fmt(pls)}</i></td>'


def entity_table():
    """Cell B, entity level: rulers -> year and find-spots -> lat/lon, the paper's
    pooling only. The average-over-the-name read-out is reported in the takeaway,
    since it never separates from the controls."""
    import csv
    rows = {}
    with open(os.path.join(AKK, "summary_entity_best.csv")) as f:
        for r in csv.DictReader(f):
            rows[(r["arm"], r["entity_type"], r["site"], r["rows"])] = r

    def cell(arm, et):
        site = "text" if arm == "tfidf" else "ent_last"
        r = rows.get((arm, et, site, "bare"))
        if not r:
            return None, None
        return float(r["ridge_mc_rho"]), float(r["pls5_mc_rho"])

    head = ('<table class="rtbl compact"><thead>'
            '<tr><th>model</th>'
            '<th class="num">YEAR &nbsp;&middot;&nbsp; 34 rulers &rarr; year</th>'
            '<th class="num">PLACE &nbsp;&middot;&nbsp; 25 find-spots &rarr; latitude, longitude</th>'
            '</tr></thead><tbody>')
    body = []
    for arm, label, ctrl in ROWS_B:
        tr = '<tr class="rand">' if ctrl else '<tr>'
        body.append(tr + f'<td><span class="mdl">{label}</span></td>'
                    + _rp(*cell(arm, "assyrian_ruler"))
                    + _rp(*cell(arm, "mesopotamian_place")) + '</tr>')
    return head + "".join(body) + "</tbody></table>"


def frag_table(variant, target):
    """Cell B/C, fragment level: both poolings x {Spearman, R2}, balanced draws."""
    import json as _json

    def read(arm, site):
        if target == "year":
            p = os.path.join(AKK, "probes", arm, f"{variant}.r8.year.{site}.ridge.json")
            key = "mc"
        else:
            p = os.path.join(AKK, "probes_geosite", arm,
                             f"{variant}.{site}.geo_site.json")
            key = "mc_site"
        if not os.path.exists(p):
            return None, None
        m = _json.load(open(p))[key]
        return m.get("spearman_mean"), m.get("r2_mean")

    def tfidf_year():
        import csv
        with open(os.path.join(AKK, "summary_ALL_modes_full.csv")) as f:
            for r in csv.DictReader(f):
                if (r["arm"] == "tfidf" and r["variant"] == variant
                        and r["rulers"] == "r8" and r["target"] == "year"):
                    return float(r["mc_rho"]), float(r["mc_r2"])
        return None, None

    head = ('<table class="rtbl compact"><thead>'
            '<tr><th rowspan="2">model</th>'
            '<th colspan="3" class="num">Spearman &rho;</th>'
            '<th colspan="3" class="num">R&sup2;</th></tr>'
            '<tr><th class="num">last token</th><th class="num">average</th>'
            '<th class="num">difference</th>'
            '<th class="num">last token</th><th class="num">average</th>'
            '<th class="num">difference</th></tr></thead><tbody>')
    body = []
    for arm, label, ctrl in ROWS_B:
        if arm == "tfidf":
            if target == "year":
                rho, r2 = tfidf_year()
            else:
                rho, r2 = read("tfidf", "text")
            tr = '<tr class="rand">'
            body.append(
                tr + f'<td><span class="mdl">{label}</span></td>'
                f'<td class="num" colspan="2">{_fmt(rho)}</td>'
                '<td class="num">&ndash;</td>'
                f'<td class="num" colspan="2">{_fmt(r2)}</td>'
                '<td class="num">&ndash;</td></tr>')
            continue
        rl, r2l = read(arm, "last")
        rm, r2m = read(arm, "mean")

        def d(a, b):
            if a is None or b is None:
                return "&ndash;"
            v = b - a
            return f'{"+" if v > 0 else ""}{_fmt(v)}'
        tr = '<tr class="rand">' if ctrl else '<tr>'
        body.append(tr + f'<td><span class="mdl">{label}</span></td>'
                    f'<td class="num">{_fmt(rl)}</td><td class="num">{_fmt(rm)}</td>'
                    f'<td class="num">{d(rl, rm)}</td>'
                    f'<td class="num">{_fmt(r2l)}</td><td class="num">{_fmt(r2m)}</td>'
                    f'<td class="num">{d(r2l, r2m)}</td></tr>')
    return head + "".join(body) + "</tbody></table>"


CELL_ORDER = ["A", "B", "C"]


def cellmap(active):
    """Small 2x2 position marker: green = the cell this slide is in, pale green =
    cells already covered, dashed grey = D, which has no honest filler."""
    done = set(CELL_ORDER[:CELL_ORDER.index(active)]) if active in CELL_ORDER else set()

    def cls(c):
        if c == "D":
            return "cm-c na"
        if c == active:
            return "cm-c on"
        return "cm-c done" if c in done else "cm-c"
    return (
        '<div class="cellmap">'
        '<div></div><div class="cm-h">Eng</div><div class="cm-h">Akk</div>'
        f'<div class="cm-r">salient</div><div class="{cls("A")}">A</div>'
        f'<div class="{cls("D")}">&#10005;</div>'
        f'<div class="cm-r">obscure</div><div class="{cls("B")}">B</div>'
        f'<div class="{cls("C")}">C</div>'
        '</div>')


def inject_css(head):
    return head.replace("</style>", EXTRA_CSS, 1)


def figure_tag(name, alt=""):
    """Inline a freshly rendered figure from world_models/results/figs as base64."""
    import base64
    path = os.path.join(WM, "results", "figs", name)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}">'


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
        for fig in set(re.findall(r'\{\{FIG:([^}]+)\}\}', sec)):
            sec = sec.replace("{{FIG:%s}}" % fig, figure_tag(fig, alt=title))
        if "{{TABLE:cellA}}" in sec:
            sec = sec.replace("{{TABLE:cellA}}", cellA_table())
        if "{{TABLE:entity}}" in sec:
            sec = sec.replace("{{TABLE:entity}}", entity_table())
        for m in set(re.findall(r'\{\{TABLE:frag:([a-z0-9_]+):([a-z]+)\}\}', sec)):
            sec = sec.replace("{{TABLE:frag:%s:%s}}" % m, frag_table(*m))
        for tok in set(re.findall(r'\{\{IMG:(\d+)\}\}', sec)):
            sec = sec.replace("{{IMG:%s}}" % tok, image_of(old, int(tok), alt=title))
        if kind == "old" and ref in EYEBROW_PATCHES:
            sec = set_eyebrow(sec, EYEBROW_PATCHES[ref])
        sec = re.sub(r'(<section class="slide[^"]*)"',
                     rf'\1" data-index="{i}"', sec, count=1)
        if i in CELLMAP_AT:
            sec = re.sub(r'(<section[^>]*>)', r'\1\n  ' + cellmap(CELLMAP_AT[i]),
                         sec, count=1)
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
