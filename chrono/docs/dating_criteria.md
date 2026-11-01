# P1.0 — How philologists date Akkadian royal inscriptions, and what CJB does with each criterion

**Status: DRAFT for expert review.** This is the P1.0 deliverable from the
plan addendum (`phd_plan_chrono_jepa.md`): a survey of every criterion
Assyriologists actually use to date and periodize Akkadian royal
inscriptions, with each criterion assigned to exactly one of three roles in
the Chrono-JEPA-Barlow system. Nothing here is final until an Assyriologist
signs off (P6.4 pre-work); rows carry an explicit confidence mark, and the
open-questions section at the end lists every decision we want overruled or
confirmed. Web verification from this environment was partial (several
academic domains are egress-blocked); rows verified online today are marked
`verified-web`, the rest are from prior knowledge and marked `unverified`.

**The three classes** (from the addendum, P1.0):

- **(a) augmentation** — remove or normalize it in a training view, so the
  representation *cannot* lean on it. These are shortcut carriers: token
  identities and fixed strings that date the composition context by lookup,
  not by language. They feed `chrono/augment` (P1.1-P1.4).
- **(b) confound** — a variable that correlates with date in *this* corpus
  but is not a property of the language (metadata, archive, editorial
  layer, preservation). It cannot be deleted from the text (or is not in
  the text at all); instead we audit s(x) against it and deconfound
  (HSIC arm, held-out splits, leakage probes — P1.5, P3.5, P4.2).
- **(c) legitimate diachronic feature** — productive grammar, orthography,
  or lexicon distributed across running text: evidence a philologist would
  accept even in an anonymized fragment. The representation *should* use
  these, and the SAE (P6.x) is expected to rediscover them — this is the
  ChronoAtlas target list.

**Decision rule used throughout.** If a probe could exploit the criterion as
a lookup key for one reign/edition/archive → (a). If it lives in metadata or
the acquisition pipeline rather than the text → (b). If it is a distributed
property of the language of the fragment → (c). Where a criterion is
genuinely both a real diachronic marker *and* a shortcut (titulary is the
paradigm case), it is split into two rows: the surface string is (a), the
system-level pattern is (c). One warning applies globally: Neo-Babylonian
scribes **archaize deliberately** (orthography, script, lexicon), so several
(c) features are expected to be non-monotone in t — old-looking surface ≠
early date. Flagged per row.

**Corpus scope.** The eligible corpus (SLA §3) is 1,187 royal-inscription
fragments, 40 rulers, dominated by Neo-Assyrian (939 rows; Tiglath-pileser
III → Sîn-šarru-iškun) and Neo-Babylonian (217; Nabopolassar → Nabonidus),
with a Middle Babylonian tail (Nebuchadnezzar I) and a few Achaemenid/
Hellenistic strays. All are genre = royal inscription, written largely in
the Standard Babylonian literary register; the deepest usable splits are
therefore NA vs NB conventions and *within*-NA / *within*-NB drift, not the
full OA/OB → NB textbook ladder. Years are astronomical t inside chrono/
(SLA §1): larger = later; the survey says "later" and means larger t.

**Which text stream carries what.** The corpus ships two transliteration
tiers, and they do not preserve the same criteria:

- `text_eng` (source `text_tier0`) keeps sumerograms in caps (`LUGAL`,
  `DINGIR-MEŠ`), determinatives as hyphenated prefixes/suffixes (`m-`,
  `d-`, `giš-`, `-ki`), and sign-index subscripts (`ša2`, `u2`, `git2`).
- `text_akk` (source `text_maximal`) strips logograms, sign indices and
  aleph entirely (checked on Sennacherib doc 1: tier0 `m-d-EN-ZU-ŠEŠ-MEŠ-
  eri-ba LUGAL GAL LUGAL dan-nu LUGAL KUR aš-šur-ki ...` → maximal
  `m-eri-ba dan-nu aš-šur-ki ...`).

So orthographic and logographic criteria (families 2-3) are **only
observable in the tier0 stream**; every implementation note below says which
stream it targets. Augmentation registry names refer to SLA §4
(`mask_ruler`, `strip_formula`, `crop*`, `orthonorm`, `drop_span`);
proposed new names are in the shortlist at the end.

---

## 1. Dialect stage and grammar (morphology, syntax)

The textbook periodization (Old/Middle/Neo Assyrian; Old/Middle/Neo/Late
Babylonian; von Soden, GAG) dates a text by which stage of the language it
is written in. Royal inscriptions complicate this: from the Middle Assyrian
period on they are composed in the Standard Babylonian (SB) literary
dialect, so the dating evidence is usually the *leakage* of the scribe's
vernacular (Assyrianisms in NA-composed SB; late Babylonian vernacular
features in NB) rather than the register itself.

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Subordinative -u vs Assyrian -(ū…)ni | SB uses Bab. -u; the Assyrian double marking in -ni betrays an Assyrian scribe → NA-period composition | (c) | productive morphology across running text; exactly what an anonymized fragment still shows | interp target: P6.2 screening for features firing on subordinate-clause verb endings; too token-level for a safe regex augmentation | GAG §83; Hämeen-Anttila SAAS 13 (2000); verified-web: high |
| Ventive use and allomorphy (-a(m)/-nim; NA distribution; ventive+subjunctive co-occurrence constraints differ Ass./Bab.) | Assyrian vs Babylonian scribal habit; frequency drifts within NA | (c) | distributed verbal morphology | interp target only; note ventive is also formula-borne (dedication verbs) so SAE hits must be checked against family 6 overlap | GAG; Hämeen-Anttila SAAS 13; verified-web (ventive/subjunctive exclusivity in Bab.): high |
| Mimation (-m on case endings) | present OB/OA; lost by MB — in this corpus only relevant as deliberate archaism in NB inscriptions | (c) | orthographic-morphological; NB archaizing makes it NON-monotone in t (warning) | interp target; count `-um`/`-am` word-final spellings in tier0 as an audit column too | GAG §63; standard; unverified: high |
| Assyrianisms in SB royal inscriptions (e.g. NA issi for itti, urdu for wardu, NA pronominal forms, vowel-harmony spellings) | marks NA-period composition inside the shared SB register | (c) | vernacular leakage is a classic dating argument; distributed, not a single string | interp: seed P6.2 with a lexical list of known Assyrianisms (ask expert; Luukko's grammatical-variation work and the NA treebank are the source to mine) | Hämeen-Anttila SAAS 13; Luukko, Grammatical Variation in Neo-Assyrian (SAAS 16, 2004); unverified: med-high |
| iptaras (perfect) encroaching on iprus (preterite) as narrative past | late Babylonian development; SB royal register resists it, so frequency is a subtle lateness cue | (c) | distributed verbal syntax | interp target; low expected effect size in this register — do not gate on it | GAG §80; standard; unverified: med |
| Case-system decay (confused/frozen case vowels) | vernacular NB/LB feature; occasional slips in NB royal SB | (c) | distributed morphology; another candidate for genuine within-NB lateness | interp target; screening = mismatch rate between expected and written case vowel (needs parsed sample; expert help) | Streck, Neubabylonisch grammar lit.; unverified: med |
| Increased analytic ša-genitive periphrasis vs bound construct | slow 1st-millennium drift | (c) | distributed syntax | interp target; `ša` frequency is also clause-type-driven — audit against family 7 structure | standard; unverified: med |
| Wholesale register choice (SB literary vs vernacular) | constant across this corpus (all royal SB) | (b) | zero variance here, but any stray vernacular passage (quoted speech, oaths) correlates with genre-within-genre | audit only: flag docs with quoted direct speech (`um-ma`) in confound_table | corpus check; high |

## 2. Orthography and syllabary

Philologists date manuscripts (as opposed to compositions) largely by
spelling habits. In this corpus the loudest split is NA vs NB spelling
culture; NB royal orthography is *deliberately archaizing* (Schaudig's
grammar of the Nabonidus corpus documents this at length), which is the
strongest instance of the non-monotonicity warning.

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Plene spellings (extra vowel signs: na-a-du, re-e-a, mu-ut-ne-en-nu-u) | NB royal orthography is lavishly plene/broken; NA more compact; within NB, plene habits vary by reign | (c) | genuine scribal-culture diachrony a philologist uses; distributed over every word | interp target (atlas family "orthographic"); ALSO add `plene_collapse` stress view to the EVAL battery only (shortlist #9) to measure how much of s(x) rides on it — a (c) feature can still be over-relied on | Schaudig, Die Inschriften Nabonids (AOAT 256, 2001); Da Riva, GMTR 4 (2008); unverified: high |
| CvC vs Cv-vC sign choice (dan vs da-an, kit vs ki-it) | NA uses CvC signs freely; NB archaizing prefers broken Cv-vC writings | (c) | same as plene: distributed sign-inventory habit | tier0 only; screening = ratio of CvC-capable syllables written broken; part of `plene_collapse` normalization | Schaudig 2001; standard; unverified: high |
| Sign-value fashion (which index of a value is in use: ša2, u2, li2 …) | sign-value inventories drift by period and school | (c) | the transliteration index encodes WHICH sign the scribe chose — real palaeographic-adjacent evidence surviving into tier0 | tier0 only (maximal strips indices). interp target; also audit: index distribution may partly reflect EDITORIAL convention, see family 10 row on RINAP/RIBo | Labat, Manuel d'épigraphie akkadienne; unverified: med-high |
| Sibilant / sandhi spellings (e.g. Assyrian š→s before dental: is-sa-kan-type writings) | Assyrian phonology leaking into spelling → NA | (c) | distributed phonological orthography | interp target; overlaps family 1 Assyrianisms — tag jointly in taxonomy | GAG; Hämeen-Anttila SAAS 13; unverified: med |
| Historical / morphographemic spellings (writing dropped sounds) | later periods spell historically; NB archaizes | (c) | distributed | interp target only | standard; unverified: med |
| Damage, restoration and cleaning artifacts (lacunae, stripped aleph leaving `ra- i`, pre-masked variants' `[PN]`, mask density) | preservation and editorial cleaning correlate with archive (Kuyunjik prisms are well preserved) and hence with period | (b) | not language: acquisition-side | audit: confound_table already carries mask_count; ADD lacuna/artifact token counts (isolated single-char tokens, bracket residue) as columns; leakage probe target in P3.5 | corpus inspection; high |
| Overall orthographic "compactness" (chars per word, signs per word) | NA compact vs NB broken → near-duplicate of length-family statistics | (b) | a one-dimensional summary statistic; supports collapse-to-period shortcut rather than ordering | audit: add signs-per-word to confound_table; HSIC candidate if leakage probe fires | this project's F28 logic; high |

## 3. Logograms, sumerograms, determinatives

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Gross logogram (sumerogram) density | NA royal inscriptions are heavily logographic (LUGAL, KUR, DINGIR-MEŠ); NB royal inscriptions are strikingly syllabic (archaizing) — near-perfect NA/NB separator in this corpus | (a) | philologically real, but as a *density* it is a caps-token ratio: a trivial low-dimensional shortcut to period membership, useless for ordering within period | new aug `logonorm` (shortlist #8), tier0 stream: spell out top-N sumerograms to normalized syllabic lemma (LUGAL→šarru, KUR→mātu, DINGIR→ilu, E2→bītu …) or strip to a typed `<LOG>`; note text_akk (maximal) already deletes logograms, so training the akk branch on maximal is itself this augmentation | Da Riva GMTR 4; Schaudig 2001; corpus inspection (Sennacherib vs Nabonidus samples); high |
| Lexeme-specific logographic vs syllabic choice (WHICH words a scribe logographs) | fine-grained school/period habit, e.g. how "king", "temple", month words are written per reign | (c) | distributed, many-dimensional — the legitimate residue once gross density is normalized | interp target on tier0 embeddings; only meaningful in runs WITHOUT `logonorm` in the menu — record menu per run so P6.x knows what was visible | standard editorial knowledge; unverified: med-high |
| Determinative repertoire and frequency (d-, m-, giš-, uru-, kur-, -ki) | usage conventions differ NA vs NB and drift over time (e.g. URU before city names is characteristically Assyrian practice) | (c) | distributed classifier-morpheme usage; real evidence | tier0 only. interp target; CAUTION: `orthonorm` (SLA §4) collapses determinatives — a menu containing orthonorm hides this (c) feature from the invariant representation; keep one no-orthonorm arm (see open question 3) | standard; unverified: med |
| Determinatives attached to NAMES (m-/d-/uru- immediately before a masked span) | flags that a name was there even after masking | (a) | a typed mask that leaves `m-` in place still leaks "a personal name stood here" count and position | implementation detail of P1.1: mask_ruler/mask_pn must consume the leading determinative into the `<RULER>`/`<PN>` token; property-test this | this project (E3b/LEACE identity findings); high |

## 4. Royal titulary and epithets

The single most-used practical dating tool for royal inscriptions: each
king, and often each *phase of a reign*, has a characteristic title set
(Seux's repertory; Cifola's variant analysis; the RIMA/RINAP introductions
date many texts this way — e.g. "king of the four quarters" is not held by
all NA kings and is sometimes adopted only after specific conquests,
verified-web). Exactly because it is so diagnostic, it is the paradigm
shortcut: titulary is a fixed string near the document opening that
identifies the ruler as surely as his name does.

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Imperial title strings: šar kiššati "king of the universe", šar kibrāt erbetti "king of the four quarters", šarru dannu, šarru rabû | NA standard repertoire; presence/absence and order fingerprint individual kings and reign phases | (a) | fixed formulaic strings; ruler-lookup by proxy | `strip_titulary` rule set (shortlist #6): dedicated regexes for the opening cascade `LUGAL GAL(-u2) LUGAL dan-nu LUGAL KUR aš-šur(-ki) LUGAL la ša2-na-an` and syllabic equivalents on both streams; distinct from generic strip_formula so precision is measurable per family (P1.2 gate: precision ≥ .9) | Seux, Épithètes royales (1967); Cifola 1995; RINAP intros; verified-web (four-quarters attestation pattern): high |
| šar māt Šumeri u Akkadi "king of Sumer and Akkad", šakkanakki Bābili | claimed only when Babylon is held → dates NA texts to specific political phases; standard in NB | (a) | political-status string; near-ID for reign phase | include in `strip_titulary` | Seux 1967; RINAP/RIBo; unverified: high |
| NB pious titulary: zānin Esagil u Ezida "provider of Esagil and Ezida", rubû na'du, rē'û kīnu, migir dMarduk | standard for Nebuchadnezzar II / Neriglissar / Nabonidus; short, pious, non-military — instantly separates NB from NA | (a) | fixed strings | `strip_titulary` NB rule subset (syllabic patterns: za-ni-in e2-sag-il2 u3 e2-zi-da etc.) | Berger, Die neubabylonischen Königsinschriften (AOAT 4/1, 1973); Da Riva GMTR 4; unverified: high |
| Titulary SYSTEM style (imperial-military cascade vs short pious dedication style; epithet count; deity-linked epithets) | the system itself evolved: MA→NA inflation, NB deflation/piety turn, Nabonidus' idiosyncratic Sîn-centered epithets | (c) | after string-stripping, the residual *style* of self-presentation is a real diachronic-ideological signal the SAE should find | atlas target; screen SAE features on strip_titulary-VIEWS vs originals: features surviving stripping but tracking style are the interesting ones | Cifola 1995; Karlsson, Relations of Power (2016); Tadmor, History and Ideology; unverified: med-high |
| Within-reign titulary phases (titles earned mid-reign; Esarhaddon/Ashurbanipal epithet growth) | dates a text WITHIN a reign — finer than our ruler-level t can resolve | (b) | our label t is one composition year per doc via ruler proxy; within-reign phase structure creates label noise, not signal we can score | note for A1/A3: reign-proxy intervals (ruler_table t_min/t_max) already absorb this; document as known label-noise source in eval writeups | RINAP intros (Frahm, Einleitung in die Sanherib-Inschriften, AfO Bh. 26, 1997 for the method); verified-web (mid-reign adoption): med-high |
| Official-title lexicon (ša rēši / LU2.SAG variants, turtānu, rab šāqê, bēl pīḫāti/pāḫutu) | administrative terminology and its spellings evolve across NA and into NB | (c) | lexical-administrative drift is legitimate; the *lexeme*, unlike a personal name, recurs across reigns | interp target; the ša-rēši spelling variants (logographic LU2.SAG vs syllabic ša re-e-ši) are a nice tier0 test case for lexeme-specific orthography (family 3) | PNA; SAA glossaries; unverified: med |

## 5. Genealogy and filiation formulae

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Filiation chain "PN1 son of PN2 … descendant of PN3" | ancestor names identify the ruler uniquely (Esarhaddon: son of Sennacherib, descendant of Sargon) | (a) | pure token identity, one hop removed | ancestors are themselves rulers: ensure ruler_spans / name_variants (ignite_anchor approach, SLA §3) covers ALL 40 ruler names wherever they occur, not only the authoring ruler — mask_ruler then handles ancestors for free; verify in P1.1 tests | RINAP conventions; high |
| Presence/absence fingerprints (Sargon II usually omits filiation; Sennacherib never names Sargon; Nabopolassar "mār lā mammāna" son of nobody; Nabonidus son of the non-royal Nabû-balāssu-iqbi) | famous dating/attribution arguments | (a) | the *pattern of omission* is a per-ruler fingerprint, an identity key | after masking, the SHAPE `<RULER> son of <RULER>` vs no filiation still leaks a fingerprint bit — `strip_titulary` should therefore remove the whole genealogy clause, not just mask names inside it | standard NA/NB history; unverified: high |
| Legitimation rhetoric style (divine election, "eternal seed of kingship", fictive ancestors) | evolves NA→NB; usurpers argue differently in every period | (c) | distributed rhetoric, not a string; genuine ideology diachrony | atlas target, expert-guided; overlaps titulary style row | Tadmor; Karlsson 2016; unverified: med |

## 6. Curse, blessing and concluding formulae

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Curse formulae (may DN1, DN2 … overthrow his kingship, blot out his name) | deity sequence and phrasing evolve by period and by king; NA curses invoke Aššur-led lists, NB invoke Marduk/Nabû-led lists | (a) | closed-class formulaic strings at fixed structural position; per-reign fingerprints | `strip_curse` rule set (shortlist #7): trigger lexicon (ar-rat, li-ru-ur, lis-kip, temen, šu-mi3 šaṭ-ru …) + trailing-position heuristic; measure precision on the P1.2 expert-labeled 100-doc sample | RIMA/RINAP editorial practice; Watanabe & Parpola SAA 2 (treaty curses, comparative); unverified: med-high |
| Blessing for the restorer / address to a future prince (ana rubê arkî) | structural slot present across NA+NB building texts; phrasing drifts | (a) | formulaic string | include in `strip_curse` family (same structural tail) | standard; unverified: high |
| Curse-theology SYSTEM (which deities curse, in what order, with what powers) | genuine religious-history diachrony | (c) | after string removal, the conceptual deity-role pattern remains an atlas target; philologists genuinely periodize by deity-list order | interp target; candidate for the expert study's feature-naming sessions (P6.4) | Tallqvist, Akkadische Götterepitheta; unverified: med |

## 7. Narrative and building-account structure

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Annalistic organization by numbered campaign (ina maḫrê girriya "on my first campaign", Sennacherib-style) vs palû (regnal-year) organization (Tiglath-pileser III, Sargon II) | organizing principle differs by king and evolves within NA | (a) | the ordinal + organizer word pair is a two-token reign fingerprint AND a within-reign edition marker | `mask_num` (shortlist #5) replaces ordinals/numerals with `<NUM>`; `strip_formula` gets rules for campaign/palû headers | Tadmor, Annals of Tiglath-pileser III; Frahm AfO Bh. 26; unverified: high |
| Genre composition: military annals + summary inscriptions (NA) vs building-and-piety accounts with prayers, no military narrative (NB) | the loudest structural NA/NB split | (b) | genuinely cultural-diachronic, but in this corpus it is inseparable from object type and archive (prism/slab = NA palace, cylinder = NB temple) — F28 measured object type at −.094; treating it as text-internal signal would just relearn sub_genre | audit: sub_genre + a coarse has-military-narrative flag in confound_table; deconfound via object_held_out split (SLA §3) and HSIC arm; crops (`crop8..64`) are the mechanical hedge that breaks global document structure | F28 ladder (this repo); Grayson RIMA intros; high |
| Building-account slot sequence (predecessor's work decayed → foundations sought → rebuilt → inscription deposited → future prince addressed) | stable macro-structure across NA+NB; phrasing of each slot drifts (anḫūssu uddiš, temen formulas, ina ūmēšūma transitions) | (a) | the slot-filler phrases are formulaic strings | `strip_formula` rules for transition markers (i-na u4-me-šu-ma, anḫūssu, temen-); slot ORDER itself is near-constant → carries no date, ignore | Grayson; Da Riva GMTR 4; unverified: med-high |
| Embedded prayers (long Marduk/Sîn prayers in NB; Nabonidus' Sîn theology) | NB-typical; Nabonidus-specific theology is reign-diagnostic | (c) | distributed devotional content; the theological PROFILE is legitimate religious diachrony (see family 9 deity row) | atlas target | Schaudig 2001; Beaulieu, Reign of Nabonidus; unverified: high |

## 8. Date formulae, eponyms, month names, numerals

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Assyrian eponym (līmu) dates | eponym name = exact Julian year (Millard's canon 910-649) — the strongest single dating device in NA texts | (a) | literally a year label written in the text; keeping it makes the task label-reading, not representation learning | `mask_official` / dedicated `<EPONYM>` typed mask over `li-mu/li-me PN` patterns; must fire before any rank-loss training | Millard, The Eponyms of the Assyrian Empire (SAAS 2, 1994); unverified: high |
| Babylonian regnal-year formulas (MU.N.KAM, "in my Nth year") | explicit year-within-reign | (a) | same: an in-text label | `mask_num` + rule in strip_formula for date clauses | standard; high |
| Month names (Nisannu … Addaru; intercalaries) | standardized Babylonian cultic calendar is shared NA/NB by the 1st millennium; a month name dates a day, not a period | (b) | near-zero period information within this corpus, but month mentions correlate with text type (building rituals) → weak genre proxy | audit column only (month-token count); no augmentation warranted; ASK EXPERT whether any Assyrian-calendar month survivals in early NA texts are worth a (c) row | standard calendrics; unverified: med |
| Numbers and quantifications (booty counts, army sizes, measurements) | magnitudes are typological/manipulated and get inflated across editions of the same composition — De Odorico's core result; specific figures fingerprint editions and reigns | (a) | figures are edition-identity carriers, and numeral tokens are trivially memorable | `mask_num` (shortlist #5): map all numerals (and number-sumerograms in tier0) to `<NUM>`; keep a count column in confound_table (numeral density itself is a genre proxy → audit) | De Odorico, SAAS 3 (1995); verified-web (bibliographic + thesis of the book): high |

## 9. Onomastics, prosopography, synchronisms

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Ruler's own name and variants | absolute identity → F28's largest single carrier (−.150) | (a) | already the SLA's first augmentation | `mask_ruler` (exists, SLA §4) over ruler_spans; P1.1 gate ≥ 98% span recall | this repo (F28, E3b); high |
| Foreign contemporaries (Taharqa, Teumman, Merodach-baladan, Urartian and Elamite kings) and officials in narrative | synchronisms pinpoint reigns — the philologist's precision tool, the model's cheapest shortcut | (a) | token identity | `mask_pn` gazetteer mask (shortlist #1): harvest name lists from RINAP/RIBo indices + PNA; on tier0 the `m-` determinative gives a high-recall trigger for unlisted names | PNA (Radner/Baker eds.); RINAP indices; unverified: high |
| Toponyms, incl. period-bound foundations (Dur-Šarrukin only exists after Sargon II; Kār-X foundations; Nineveh as capital from Sennacherib) | place-name inventory dates texts historically, not linguistically | (a) | identity-by-geography | `mask_place` (shortlist #3): uru-/kur- determinative triggers in tier0 + gazetteer for maximal | standard historical geography; unverified: high |
| Divine NAME tokens (Aššur, Marduk, Sîn, Nabû, Ištar of Nineveh/Arbela) | pantheon prominence is reign-diagnostic (Nabonidus' Sîn; Sargonid Ištar pair; Nabû's NA rise) | (a) | the plan's design commitment 2 already lists `<DIVINE>` masks: a bare deity token is one-hop identity (patron god → dynasty/reign) | `mask_divine` (shortlist #2): `d-` determinative trigger on tier0, gazetteer on maximal; theophoric elements INSIDE personal names belong to `mask_pn`, not `mask_divine` (open question 6) | plan §4; Beaulieu; unverified: high |
| Theological PROFILE (divine epithets, temple names' liturgical contexts, deity ROLE assignments) after name masking | religious-history drift: how gods are characterized, which temples are provisioned | (c) | distributed content that survives `<DIVINE>` masking; a philologically meaningful atlas family and the fair test of whether CJB learns culture rather than tokens | interp target; screen on mask_divine views | Tallqvist; Beaulieu; unverified: med |

## 10. Palaeography, object typology, provenance, archive, edition

These criteria are central for philologists but are (mostly) **not in our
text stream**: transliteration launders sign forms, and object/find-spot
live in metadata. They act on CJB as confounds.

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Sign-form evolution / ductus (OA→NA cursive; NB script; archaizing lapidary script, e.g. Nabonidus stelae) | a primary dating tool on the artifact | (b) | invisible in transliteration except as faint sign-CHOICE echoes (family 2); cannot be augmented, only audited | no text handle; document as out-of-model evidence in ChronoAtlas report cards (P5.3) so users don't think the model saw it | Labat; Schaudig 2001 (archaizing); unverified: high |
| Object type (prism, cylinder, brick, slab, bull colossus …) | prisms/slabs = NA palaces; cylinders = Babylonian temple deposits; measured −.094 in F28 | (b) | metadata; correlates with period via deposit practice | sub_genre in confound_table; object_held_out split; HSIC arm (P4.2) | F28 (this repo); Grayson/Da Riva typologies; high |
| Object SELF-reference in the text (narû, musarû, temen, asumittu, "this prism/cylinder") | the text names its own carrier → imports the object-type confound INTO the text stream | (a) | a text-internal handle on a (b) variable — strip it so the (b) audit stays meaningful | `mask_object_ref` (shortlist #10): small lexicon of self-reference terms → `<OBJ>` | corpus reading; project reasoning; med-high |
| Provenance / find-spot / archive (Kuyunjik 497, Babylon 121, Assur, Kalhu, Khorsabad) | archives are reign-bound (Khorsabad ≈ Sargon II); measured −.046 in F28 | (b) | metadata | provenance in confound_table; source_held_out split | F28; high |
| Length (n_words) | measured ≈ 0 in F28; keep watch anyway (fragmentation correlates with archive) | (b) | metadata/preservation | already in confound_table; variance-guard only | F28; high |
| Editorial layer: RINAP vs RIBo/RIBb project conventions (transliteration habits, index usage, restoration policy) — project boundary ≈ NA/NB boundary | any project-specific convention becomes a fake period feature | (b) | acquisition-side; potentially the sneakiest confound in this corpus since it aligns exactly with the biggest period split | audit: add source-edition/project column to confound_table if recoverable from fragment_id prefixes; ASK EXPERT/data-owner which conventions differ (open question 9); `orthonorm` reduces exposure | project reasoning over ORACC/RINAP/RIBo practice; unverified: med — VERIFY |

## 11. Lexicon and language contact

| criterion | period signal | class | why | implementation note | source / confidence |
|---|---|---|---|---|---|
| Aramaic loanwords and calques | rise through NA into NB/LB; royal SB register admits them slowly, so presence is a lateness cue | (c) | distributed lexical drift | interp target; seed screening with published Aramaism lists (expert to supply; Abraham & Sokoloff's survey is a candidate) | standard; unverified: med-high |
| Iranian (Median/Old Persian) loans and names | Achaemenid marker; marginal here (3 docs) | (c) | legitimate but corpus-marginal | interp: expect nothing; note only | standard; unverified: high |
| Military-administrative terminology turnover (new weapon, troop, office, tribute terms per imperial phase) | real historical semantics | (c) | distributed lexicon | atlas target; overlaps family 4 official-title row | SAA glossaries; unverified: med |
| Deliberate lexical archaism (rare OB literary words revived in NB) | NON-monotone: late texts wearing old clothes | (c) | genuine phenomenon philologists correct for; the SAE finding an "archaism" feature with a U-shaped date profile would be a headline result, not a bug | interp target; instruct P6.2 screening to test absolute correlation against t AND against period residuals, not just linear corr | Schaudig 2001; standard NB scholarship; unverified: high |

---

## Prioritized shortlist: top 10 augmentations beyond the F28-derived menu

The SLA §4 menu (`mask_ruler`, `strip_formula`, `crop8..64`, `orthonorm`,
`drop_span`) covers F28's measured carriers. The survey adds, in priority
order (1 = build first; "train" = candidate for menu_a/menu_b, "eval" =
robustness-battery view only):

1. **`mask_pn`** (train) — typed `<PN>` over ALL personal names: ancestors,
   enemy kings, officials. Gazetteer from RINAP/RIBo name indices + PNA;
   tier0 trigger: `m-` determinative. Closes the synchronism shortcut,
   the philologist's single sharpest tool (family 9).
2. **`mask_divine`** (train) — `<DIVINE>` via `d-` prefix (tier0) +
   gazetteer (maximal). Plan §4 commitment; makes the theology-profile
   (c) test fair.
3. **`mask_place`** (train) — `<PLACE>` via uru-/kur- + gazetteer.
   Founded-city names are exact termini post quem.
4. **`strip_titulary`** (train) — dedicated opening-cascade + genealogy-
   clause rules, separate from generic strip_formula so P1.2 precision is
   measurable per family. Removes the densest per-reign fingerprint block.
5. **`mask_num`** (train) — numerals → `<NUM>` (both streams; incl. number
   sumerograms). Kills eponym-year, regnal-year, campaign-ordinal and
   De Odorico-style edition-figure shortcuts in one move.
6. **`strip_curse`** (train) — closing curse/blessing/future-prince rules;
   second densest formula block, carries deity-list fingerprints.
7. **`mask_official`** (train) — `<OFFICIAL>` for office-holders and
   `<EPONYM>` for līmu clauses (LU2.-triggered on tier0). Split from
   mask_pn so leakage audits can attribute.
8. **`logonorm`** (train, tier0 branch only) — sumerogram spell-out/strip.
   Deletes the caps-density NA/NB shortcut while leaving lexeme-specific
   spelling choice (a (c) target) testable in no-logonorm arms.
9. **`plene_collapse`** (eval only) — collapse plene V-V and broken Cv-vC
   to compact skeletons. Orthography is class (c) — we do NOT train
   invariance to it, but the battery must report how much of s(x) it
   carries (guards against orthography being the WHOLE signal).
10. **`mask_object_ref`** (train, low cost) — `<OBJ>` over narû/temen/
    musarû/self-naming phrases, so the object-type confound audit (b) is
    not defeated by the text naming its own carrier.

All are pure `(text, spans_dict, rng) -> (text, spans_dict)` functions per
SLA §4 and should land in the A2 registry with the same determinism and
min-5-words-retained guards as strip_formula. Every mask family must also
emit its count into `augment.audit.confound_table` — masking density is
itself a (b) variable (plan risk register: "augmentation leakage").

## Open questions for the Assyriologist

1. **Gold spans (blocking P1.2).** We need ~100 fragments with expert-
   labeled titulary / genealogy / curse / date-formula spans to score
   strip-rule precision. Can we source these from RINAP edition
   segmentations rather than fresh annotation?
2. **Titulary split-call.** We classify epithet STRINGS as (a) and
   titulary STYLE as (c). Is there a middle tier — epithets whose
   *presence pattern* is period-diagnostic across many kings (not
   ruler-identifying) — that deserves (c) status and atlas entries?
3. **Orthonorm scope.** SLA's `orthonorm` collapses determinatives and
   diacritics. Families 2-3 classify determinative repertoire and sign
   choice as (c) — trained-in invariance to them would hide legitimate
   signal from the SAE. Should the default training menus exclude
   orthonorm from the tier0 branch (keep it eval-only, like
   plene_collapse)?
4. **Archaizing NB orthography.** Which specific NB spelling habits are
   deliberate archaism (expected non-monotone in t) vs natural late
   development (expected monotone)? A short list per habit would let
   P6.2 pre-register the expected sign of each correlation.
5. **Assyrianism inventory.** Can you (or the Luukko treebank / SAAS 13+16
   materials) supply a closed list of Assyrianisms attested in NA royal
   SB, to seed SAE feature screening for family 1?
6. **Theophoric names.** Should divine elements inside personal names
   (Sîn-aḫḫē-erība!) mask as `<PN>` only, or as `<PN>` with a recorded
   theophoric type? The onomastic-feature finding from the M.Sc. (name-
   culture SAE features) suggests theophoric fashion is itself a real
   diachronic signal — is it (c) at the name-SYSTEM level?
7. **Month names.** Any diachronic value inside this corpus (Assyrian
   calendar residue in early NA, intercalation mentions), or safely (b)?
8. **ša rēši variants.** Is the spelling/usage drift of ša rēši and
   related office terms across NA→NB well enough charted to serve as a
   named ChronoAtlas validation target?
9. **Edition-layer confound.** Which transliteration conventions differ
   between the RINAP and RIBo/eBL layers feeding this corpus (index
   preferences, restoration policy, determinative rendering)? Anything
   editor-specific becomes a fake NA/NB feature; we need the list to
   audit it (family 10, last row).
10. **Curse-formula chronology.** Are there published seriations of NA/NB
    curse deity-lists we can use as an external check on SAE curse
    features (family 6, (c) row)?

## Key references (for the expert-review packet)

- von Soden, *Grundriss der akkadischen Grammatik* (GAG), 3rd ed. 1995 —
  dialect-stage morphology baseline.
- Hämeen-Anttila, *A Sketch of Neo-Assyrian Grammar*, SAAS 13, 2000
  (verified-web); Luukko, *Grammatical Variation in Neo-Assyrian*,
  SAAS 16, 2004 (unverified).
- De Odorico, *The Use of Numbers and Quantifications in the Assyrian
  Royal Inscriptions*, SAAS 3, 1995 (verified-web).
- Millard, *The Eponyms of the Assyrian Empire 910-612 BC*, SAAS 2, 1994.
- Seux, *Épithètes royales akkadiennes et sumériennes*, 1967; Cifola,
  *Analysis of Variants in the Assyrian Royal Titulary*, 1995.
- Schaudig, *Die Inschriften Nabonids von Babylon und Kyros' des Großen*,
  AOAT 256, 2001 — NB orthography/archaism.
- Berger, *Die neubabylonischen Königsinschriften*, AOAT 4/1, 1973;
  Da Riva, *The Neo-Babylonian Royal Inscriptions: An Introduction*,
  GMTR 4, 2008.
- Frahm, *Einleitung in die Sanherib-Inschriften*, AfO Beiheft 26, 1997 —
  the model for within-reign edition dating.
- Grayson, RIMA 1-3; Tadmor & Yamada, RINAP 1; Grayson & Novotny, RINAP
  3; Leichty, RINAP 4; Novotny & Jeffers, RINAP 5 — editorial dating
  conventions per text.
- *The Prosopography of the Neo-Assyrian Empire* (PNA), Radner & Baker
  eds. — the mask_pn gazetteer source.
- Tadmor, "History and Ideology in the Assyrian Royal Inscriptions";
  Karlsson, *Relations of Power in Early Neo-Assyrian State Ideology*,
  2016 — titulary-system diachrony.
- Labat, *Manuel d'épigraphie akkadienne* — sign forms and values.
- Watanabe & Parpola, SAA 2 — comparative curse formulae.

Sources checked online today (rest of the bibliography is from prior
knowledge, unverified from this environment):
[Akkadian royal titulary (overview)](https://en.wikipedia.org/wiki/Akkadian_royal_titulary),
[King of the Four Corners (attestation pattern)](https://en.wikipedia.org/wiki/King_of_the_Four_Corners),
[RINAP series](https://www.eisenbrauns.org/books/series/book_SeriesRoyalInscriptionsofNeo.html),
[NA treebank, Luukko et al. 2020](https://aclanthology.org/2020.tlt-1.11.pdf),
[SAAS series list](https://assyriologia.fi/natcp/saas/),
[De Odorico SAAS 3](https://www.eisenbrauns.org/books/titles/978-951-45-7125-1.html),
[Hämeen-Anttila SAAS 13](https://www.eisenbrauns.org/books/titles/978-951-45-9046-7.html).
