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

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "thesis_story_9.html")

# ---------------------------------------------------------------- the spine ----
# (kind, ref, title-for-the-nav-strip)
SPINE = [
    ("old", 0,  "Diachronic Interpretable Dating of Low-Resource Ancient Akkadian"),
    ("new", "motivation", "Why we care: a world model you could point at an excavation"),
    ("new", "paper", "Gurnee & Tegmark: LLMs linearly represent space and time — in one cell"),
    ("new", "matrix", "The climbing map: salience × language resource"),
    ("new", "protocol", "The protocol, the ladder, and the two controls that gate every claim"),
    ("old", 24, "Cell A — the paper reproduces on our ladder (and the control it never ran)"),
    ("old", 29, "Cell A — where in the network do space & time live?"),
    ("old", 30, "Cell A — how many PLS components does the world model need?"),
    ("new", "b_entity", "Step 1 — same language, obscure entities (the paper's own protocol)"),
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

"motivation": """<section class="slide slide-text">
  <div class="eyebrow">Motivation</div>
  <h2 class="sh">Why we care: a world model you could point at an excavation</h2>
  <div class="text-points">
    <div class="tp"><div class="tp-h">The claim we are following up</div><div class="tp-b">Recent interpretability work reports that language models build <strong>linear internal maps of time and space</strong>: a ridge probe on frozen activations recovers when a person died or where a city is. If that geometry is a general property of large models, it is a free measuring instrument.</div></div>
    <div class="tp"><div class="tp-h">Why an archaeologist should care</div><div class="tp-b">Dating and provenancing cuneiform fragments is done by hand, from ruler names, script style and archival context &mdash; slow, expert-bound, and for most fragments simply unresolved. A probe that reads date and find-spot straight out of a frozen model would turn a scholarly bottleneck into an inference. <strong>That is the payoff we are testing for.</strong></div></div>
    <div class="tp"><div class="tp-h">But the claim was only ever tested in its comfort zone</div><div class="tp-b">Famous entities (world capitals, celebrated figures, headline news), written in <strong>English</strong>, as short entity strings. Our regime differs on three axes at once: the entities are <strong>obscure</strong>, the language is <strong>low-resource</strong>, and the unit is a <strong>damaged fragment</strong>, not a name. This deck climbs those axes one at a time and reports where the map survives and where it dies.</div></div>
    <div class="tp"><div class="tp-h">Where we end up</div><div class="tp-b">The linear world model is a property of the <strong>language a model was trained on</strong>, not of its scale &mdash; and the system that actually dates Akkadian is a <strong>400M translation encoder</strong>, not the 120B LLM.</div></div>
  </div>
</section>""",

"paper": """<section class="slide slide-text">
  <div class="eyebrow">The work we extend &middot; Gurnee &amp; Tegmark (2023)</div>
  <h2 class="sh">&ldquo;Language Models Represent Space and Time&rdquo; &mdash; a strong claim, tested in one cell</h2>
  <div class="text-points">
    <div class="tp"><div class="tp-h">Their method, which we adopt unchanged</div><div class="tp-b">Build an entity string; run it through a frozen model; take the hidden state at the <strong>last token of the entity</strong>, layer by layer; fit <strong>ridge regression</strong> to a real-world coordinate (lat/lon) or a year; report <strong>R&sup2; on held-out entities</strong>. Six datasets: world / US / NYC places, historical figures, art, news headlines.</div></div>
    <div class="tp"><div class="tp-h">Their finding</div><div class="tp-b">Recovery is strong (world places R&sup2; &asymp; .91 at Llama-2-70B), improves with scale, and peaks in the middle-to-late layers &mdash; read as evidence of a genuine, linearly-accessible world model rather than surface correlation.</div></div>
    <div class="tp"><div class="tp-h">What the setting quietly holds fixed</div><div class="tp-b">Every one of those six datasets is <strong>salient entities in English</strong>. Salience and language never vary, so the experiment cannot say whether the geometry belongs to <em>the model</em> or to <em>the well-represented slice of the world it was trained on</em>. That is the question our corpus is unusually well suited to answer.</div></div>
    <div class="tp"><div class="tp-h">And one control it does not run</div><div class="tp-b">No <strong>random-initialised twin</strong>. Without it, a probe's success cannot be separated from what a linear map can extract from any high-dimensional representation. We add that control everywhere, plus a <strong>TF-IDF character n-gram floor</strong>.</div></div>
  </div>
</section>""",

"matrix": """<section class="slide slide-text">
  <div class="eyebrow">The design</div>
  <h2 class="sh">The climbing map: entity salience &times; language resource</h2>
  <div class="exp-config">Going straight from the paper's setting to Akkadian changes <strong>two things at once</strong>, so a collapse could not be attributed to either. We therefore lay the space out as a 2&times;2 and climb it one factor at a time.</div>
  <table class="rtbl compact"><thead><tr><th></th><th>High-resource language (English)</th><th>Low-resource language (Akkadian)</th></tr></thead><tbody>
    <tr><td><span class="mdl">Salient entities</span></td><td><strong>CELL A</strong> &mdash; the paper's cell: world capitals, famous figures, headlines. <em>Slides 5&ndash;7.</em></td><td><strong>CELL D &mdash; empty.</strong> No famous entities exist in Akkadian outside these same royal names; no honest filler exists.</td></tr>
    <tr><td><span class="mdl">Obscure entities</span></td><td><strong>CELL B</strong> &mdash; Assyrian rulers and find-spots, written in English. <em>Slides 8&ndash;10.</em></td><td><strong>CELL C</strong> &mdash; the same entities in raw Akkadian. <em>Slides 12&ndash;17.</em></td></tr>
  </tbody></table>
  <div class="text-points" style="margin-top:12px">
    <div class="tp"><div class="tp-h">The two comparisons the map buys us</div><div class="tp-b"><strong>A &rarr; B</strong> holds the language fixed at English and changes only <em>who the entities are</em> &mdash; it isolates <strong>entity obscurity</strong>. <strong>B &rarr; C</strong> holds the entities fixed and changes only <em>what language they are written in</em> &mdash; it isolates <strong>language resource</strong>.</div></div>
    <div class="tp"><div class="tp-h">A third axis, inside each cell: what counts as the entity</div><div class="tp-b">The paper's unit is a short name or headline. Ours is ultimately a <strong>whole damaged fragment</strong>. So within B and C we first run the paper's own <em>entity-level</em> protocol, then extend to fragments &mdash; and pool both at the <strong>last token</strong> (theirs) and as a <strong>mean</strong> (ours).</div></div>
  </div>
</section>""",

"protocol": """<section class="slide slide-text">
  <div class="eyebrow">Protocol</div>
  <h2 class="sh">The ladder, the two read-outs, and the controls that gate every claim</h2>
  <div class="cfg">
    <div class="cfg-k">Ladder</div><div class="cfg-v">Llama-2 <strong>7B / 13B / 70B</strong> (the paper's ladder) &middot; Qwen3 <strong>1.7B / 8B / 32B</strong> &middot; <strong>gpt-oss-120B</strong> &middot; three small encoders (<strong>cuneiform-400M</strong>, <strong>AKK-300M</strong>, <strong>uMT5-base</strong>).</div>
    <div class="cfg-k">Controls</div><div class="cfg-v">a <strong>random-initialised twin</strong> of every decoder (identical architecture and tokenizer, untrained weights) and a <strong>TF-IDF character n-gram floor</strong>. These are rows in every table that follows, never a slide of their own.</div>
    <div class="cfg-k">Reading rule</div><div class="cfg-v">a score witnesses <em>learning</em> only if it beats <strong>both</strong> the TF-IDF floor <strong>and</strong> that arm's own random twin. Beating neither means the probe found geometry that any random projection of the input would have offered.</div>
    <div class="cfg-k">Read-outs</div><div class="cfg-v"><strong>R&sup2;</strong> on held-out entities, as in the paper, plus <strong>Spearman &rho;</strong> for year &mdash; dating is a ranking problem, and &rho; is robust to the compressed, unevenly-spaced year distribution of a royal-inscription corpus.</div>
  </div>
  <div class="text-points">
    <div class="tp"><div class="tp-h">Two protocol families appear in this deck &mdash; read the box on each slide</div><div class="tp-b">The <strong>replication line</strong> (cells A/B/C, slides 5&ndash;17) follows the paper: <em>last-token pooling, ridge, R&sup2;</em>, with balanced Monte-Carlo added because our corpus is unbalanced and theirs is not. The <strong>thesis line</strong> (slides 25&ndash;27) uses the protocol the applied dating work settled on: <em>mean pooling, PLS, Spearman</em>, which is what the encoders require &mdash; they have no causal last token. Numbers are comparable <em>within</em> a line, not across the two.</div></div>
  </div>
</section>""",

"b_entity": """<section class="slide slide-text">
  <div class="eyebrow">Cell B &middot; entity level &middot; the paper's own protocol</div>
  <h2 class="sh">Step 1 &mdash; hold the language fixed, swap famous entities for obscure ones</h2>
  <div class="cfg">
    <div class="cfg-k">Setup</div><div class="cfg-v">two datasets built to mirror theirs exactly: <strong>assyrian_ruler</strong> (34 rulers &rarr; year) mirrors <em>historical_figure</em>, and <strong>mesopotamian_place</strong> (25 find-spots &rarr; lon/lat) mirrors <em>world_place</em>. Each entity appears once <strong>bare</strong> (the paper-faithful row) and inside five neutral carrier sentences that never mention a date or region.</div>
    <div class="cfg-k">Pooling</div><div class="cfg-v">four sites, so the paper's protocol and ours are both visible: <strong>ent_last</strong> (last token of the ruler/place name &mdash; theirs), <strong>ent_mean</strong>, <strong>last</strong> (last token of the whole sentence &mdash; their <em>headline</em> protocol) and <strong>mean</strong>. On the bare rows the entity <em>is</em> the string, so ent_last = last by construction.</div>
    <div class="cfg-k">Metric</div><div class="cfg-v">ridge and PLS-5, <strong>R&sup2;</strong> and <strong>&rho;</strong>, over a <strong>200-draw Monte-Carlo of entity-level splits</strong> (20% of entities held out per draw; all six templates of an entity always move together, so no template can leak its target).</div>
  </div>
  <div class="fig-wrap"><div class="placeholder-box">
    <div class="ph-icon">&#9203;</div>
    <div class="ph-hint">Awaiting the WB jobs</div>
    <div class="ph-slot">world_models/akkadian/results/RESULTS_entity.md</div>
    <div class="ph-sub">WB0 build &rarr; WB1 extract &rarr; WB2 probe &rarr; WB3 aggregate &mdash; then rerun build_story_deck.py</div>
  </div></div>
  <p class="fig-note"><strong>Read this slide with its sample size in view.</strong> 34 rulers and 25 places, against the paper's thousands of entities, so a 20% split holds out 6&ndash;7 entities: point R&sup2; values will be unstable and the Monte-Carlo spread will be wide. The claim this slide can support is about the <strong>ordering of arms</strong> against the TF-IDF floor and the random twins &mdash; not about a precise R&sup2;.</p>
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
        if kind == "old" and ref in EYEBROW_PATCHES:
            sec = set_eyebrow(sec, EYEBROW_PATCHES[ref])
        sec = re.sub(r'(<section class="slide[^"]*)"',
                     rf'\1" data-index="{i}"', sec, count=1)
        bodies.append(sec)
        titles.append(title)

    head = html[:html.index('<section')]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DECK)
    ap.add_argument("--check", action="store_true", help="only verify the current deck")
    args = ap.parse_args()

    with open(DECK) as f:
        html = f.read()
    if args.check:
        raise SystemExit(0 if check(html) else 1)

    new = build(html)
    with open(args.out, "w") as f:
        f.write(new)
    print(f"[write] {args.out}  ({len(new)/1e6:.2f} MB, {len(SPINE)} slides)")
    check(new)


if __name__ == "__main__":
    main()
