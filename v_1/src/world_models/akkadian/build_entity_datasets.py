"""Build the CELL-B *entity-level* datasets — our obscure-entity mirror of G&T.

The paper probes short entity strings: a bare name (`historical_figure`, `us_place`),
a possessive construction (`art`, `world_place`) or a sentence (`headline`). We mirror
the first two with our own obscure entities, in English:

  * `assyrian_ruler`   <- their `historical_figure`  (name -> year)
  * `mesopotamian_place` <- their `world_place`      (name -> lon/lat)

so that A (famous entities, English) -> B (obscure entities, English) changes **only
entity salience**, holding the language and the entity-string format fixed. The
fragment-level runs (`extract_akk.py`) then change the *span*, and `akk_maximal`
changes the *language* — one factor per step.

Each entity is emitted once per template. `template=bare` is the paper-faithful row
(the entity string alone); the sentence templates put the same entity inside a short
carrier sentence, which is what lets us pool three ways:

    ent_last  — last token of the entity span      (their entity-last-token protocol)
    last      — last token of the whole sentence   (their `headline` protocol)
    mean      — mean over the whole sentence

Splits are **by entity**, so every template of a held-out ruler/place is in test:
a template of a train entity can never leak its target into the test rows.

Output (committed, small):
    data/entity_datasets/assyrian_ruler.csv
    data/entity_datasets/mesopotamian_place.csv

    python build_entity_datasets.py            # writes both
    python build_entity_datasets.py --report   # print coverage and exit
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import akk_data  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "data", "entity_datasets")

SEED = 42
TEST_RATIO = 0.2
# Reign years below this are regnal-year artefacts in the corpus ("year 7 of X"),
# not BC dates — they would put four rulers at "year 7-10 BC", centuries off.
MIN_PLAUSIBLE_YEAR = 100
MIN_TEXTS = 1          # entities attested by at least this many dated fragments
MIN_PLACE_TEXTS = 3    # a place needs a few fragments to be a real find-spot
NAME_FORM = "ancient"  # "ancient" (Nineveh) or "modern" (Kuyunjik); see build_places

# Carrier sentences. Kept deliberately plain and factual: they must not smuggle in
# the answer (no dates, no regions), and they mirror the register of the paper's
# `headline` rows. `{e}` is replaced by the entity; its char span is recorded.
RULER_TEMPLATES = [
    "{e}",                                        # bare — paper-faithful
    "The king {e} ruled over the land.",
    "An inscription commissioned by {e}.",
    "This tablet dates to the reign of {e}.",
    "The royal annals of {e} were compiled by his scribes.",
    "Tribute was delivered to {e} by the governors.",
]
PLACE_TEMPLATES = [
    "{e}",                                        # bare — paper-faithful
    "The city of {e} stood on the river.",
    "A tablet excavated at {e}.",
    "The temple at {e} was rebuilt.",
    "Merchants travelled to {e} with their goods.",
    "The provincial governor resided in {e}.",
]


def _expand(names, templates):
    """Cross entities with templates; return rows with the entity char span."""
    rows = []
    for i, name in enumerate(names):
        for t_ix, tpl in enumerate(templates):
            span_start = tpl.index("{e}")
            s = tpl.format(e=name)
            rows.append({
                "entity_ix": i,
                "template_ix": t_ix,
                "template": "bare" if tpl == "{e}" else f"t{t_ix}",
                "entity_string": s,
                "ent_start": span_start,
                "ent_end": span_start + len(name),
            })
    return pd.DataFrame(rows)


def _entity_split(n_entities, rng):
    """Hold out TEST_RATIO of *entities* (not rows)."""
    is_test = np.zeros(n_entities, dtype=bool)
    k = max(1, int(round(n_entities * TEST_RATIO)))
    is_test[rng.choice(n_entities, size=k, replace=False)] = True
    return is_test


def build_rulers(frag: pd.DataFrame) -> pd.DataFrame:
    g = (frag.groupby("ruler")
             .agg(year=("year", "median"), n_texts=("fragment_id", "size"))
             .reset_index()
             .rename(columns={"ruler": "name"}))
    g["implausible_year"] = g.year < MIN_PLAUSIBLE_YEAR
    g["unidentified"] = g.name.str.contains("Unidentified", case=False)
    keep = (~g.implausible_year) & (~g.unidentified) & (g.n_texts >= MIN_TEXTS)
    dropped = g[~keep]
    g = g[keep].sort_values("n_texts", ascending=False).reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    is_test_ent = _entity_split(len(g), rng)

    df = _expand(g.name.tolist(), RULER_TEMPLATES)
    df["name"] = df.entity_ix.map(g.name)
    # `death_year` keeps the column name the paper's historical_figure probe expects;
    # for our rulers it is the median attested year of their dated texts.
    df["death_year"] = df.entity_ix.map(g.year)
    df["n_texts"] = df.entity_ix.map(g.n_texts)
    df["is_test"] = df.entity_ix.map(pd.Series(is_test_ent))
    return df, g, dropped


def build_places(frag: pd.DataFrame, gaz: pd.DataFrame) -> pd.DataFrame:
    counts = frag.provenance.value_counts()
    gz = gaz[gaz.provenance.isin(counts.index)].copy()
    gz["n_texts"] = gz.provenance.map(counts)
    gz = gz[gz.n_texts >= MIN_PLACE_TEXTS]
    # aliases ("Nineveh" / "Kuyunjik (Nineveh)") share coordinates: keep the most
    # attested spelling per coordinate so an alias can't sit in train and test.
    gz = (gz.sort_values("n_texts", ascending=False)
            .drop_duplicates(subset=["lat", "lon"])
            .reset_index(drop=True))
    # The gazetteer writes find-spots as "modern (ancient)" — "Kuyunjik (Nineveh)",
    # "Khorsabad (Dur-Šarrukin)" — but plain entries are already ancient names
    # ("Babylon", "Sippar", "Ur"). Prefer the parenthetical so every entity is the
    # *ancient* toponym, which is the name an English-trained model would have seen
    # and keeps the naming convention uniform across the dataset. `--name-form
    # modern` keeps the excavation-site spelling instead (a strictly harder probe).
    paren = gz.provenance.str.extract(r"\(([^)]*)\)", expand=False)
    bare_name = gz.provenance.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    gz["name"] = bare_name if NAME_FORM == "modern" else paren.fillna(bare_name)

    rng = np.random.RandomState(SEED + 1)
    is_test_ent = _entity_split(len(gz), rng)

    df = _expand(gz.name.tolist(), PLACE_TEMPLATES)
    df["name"] = df.entity_ix.map(gz.name)
    df["longitude"] = df.entity_ix.map(gz.lon)
    df["latitude"] = df.entity_ix.map(gz.lat)
    df["region"] = df.entity_ix.map(gz.region)
    df["n_texts"] = df.entity_ix.map(gz.n_texts)
    df["is_test"] = df.entity_ix.map(pd.Series(is_test_ent))
    return df, gz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print coverage, do not write")
    ap.add_argument("--name-form", default="ancient", choices=["ancient", "modern"],
                    help="place spelling: ancient toponym (default) or modern dig site")
    args = ap.parse_args()
    global NAME_FORM
    NAME_FORM = args.name_form

    frag = akk_data.load_fragments()
    gaz = pd.read_csv(akk_data.GAZ)

    rulers, r_ent, r_dropped = build_rulers(frag)
    places, p_ent = build_places(frag, gaz)

    print(f"[rulers] {len(r_ent)} entities x {len(RULER_TEMPLATES)} templates "
          f"= {len(rulers)} rows; test entities: {int(r_ent.shape[0] * TEST_RATIO)}")
    print(f"         year range {r_ent.year.min():.0f}-{r_ent.year.max():.0f} BC-ish, "
          f"{len(r_dropped)} dropped (regnal-year artefact or unidentified):")
    for _, r in r_dropped.iterrows():
        why = "unidentified" if r.unidentified else f"year={r.year:.0f} < {MIN_PLAUSIBLE_YEAR}"
        print(f"           - {r['name']} ({why})")
    print(f"[places] {len(p_ent)} entities x {len(PLACE_TEMPLATES)} templates "
          f"= {len(places)} rows")
    print(f"         lon {p_ent.lon.min():.2f}-{p_ent.lon.max():.2f}, "
          f"lat {p_ent.lat.min():.2f}-{p_ent.lat.max():.2f}")
    if args.report:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    rcols = ["name", "entity_ix", "template", "entity_string", "ent_start", "ent_end",
             "death_year", "n_texts", "is_test"]
    pcols = ["name", "entity_ix", "template", "entity_string", "ent_start", "ent_end",
             "longitude", "latitude", "region", "n_texts", "is_test"]
    rulers[rcols].to_csv(os.path.join(OUT_DIR, "assyrian_ruler.csv"), index=False)
    places[pcols].to_csv(os.path.join(OUT_DIR, "mesopotamian_place.csv"), index=False)
    print(f"[write] {OUT_DIR}/assyrian_ruler.csv, {OUT_DIR}/mesopotamian_place.csv")


if __name__ == "__main__":
    main()
