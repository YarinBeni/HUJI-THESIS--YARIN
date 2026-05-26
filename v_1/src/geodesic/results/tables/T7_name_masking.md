# Test 7 — TF-IDF name-masking control

**What it is:** does a char-n-gram TF-IDF model date texts by reading the *king's name* or by
period spelling? We mask all personal names — `m-`/`f-` determinative tokens AND theophoric
`d-<god>-<predicate>` sentence-names (e.g. Nabu-kudurri-usur = Nebuchadnezzar -> `[PN]`), while
keeping bare god names — then re-date. Masking module: `../../linear_probing/name_masking.py`.

**Data & split:** balanced MC (200 draws x 168 frags). Year via Ridge GroupKFold-by-ruler ->
Spearman; ruler via logistic StratifiedKFold -> Macro-F1.

**CSV `T7_name_masking.csv`** — rows = {tier0,maximal} x {unmasked,masked}. Compare masked vs
unmasked within a cleaning: year Spearman is unchanged (dating != name lookup) while ruler
Macro-F1 drops (names did carry ruler identity).

**Metric harmonization (T7h):** the masking job (`tfidf_namemask_results.json`) only persisted the
headline triple per condition — year **Spearman**, year **MAE**, ruler **Macro-F1** (each with std
and n_draws). The rest of the unified year/ruler metric sets (R2, MASE, MdAPE, shuffled-* for year;
accuracy, weighted-F1, chance-*, shuffled-* for ruler) were **not computed by the masking job** and
are a principled **N/A** here, not a gap — re-running masking with the full metric set is out of
scope (the plan says "do not re-run any masking job"). The CSV therefore carries only the three
metrics the source actually provides.
