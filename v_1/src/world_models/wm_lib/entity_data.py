"""Entity datasets: loading, entity-string construction, and probe targets.

The string construction is an exact port of wesg52/world-models feature_datasets/*
(make_*_prompt_dataset) so our inputs match the paper token-for-token modulo
tokenizer. Only the `empty` prompt is wired as canonical; PROMPTS keeps the hook for
adding their question prompts later.
"""
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data", "entity_datasets")

ENTITY_TYPES = [
    "world_place", "us_place", "nyc_place",
    "historical_figure", "art", "headline",
]

# entity_type -> (feature_name, is_place)
FEATURES = {
    "world_place": ("coords", True),
    "us_place": ("coords", True),
    "nyc_place": ("coords", True),
    "historical_figure": ("death_year", False),
    "art": ("release_date", False),
    "headline": ("pub_date", False),
}

PROMPTS = {et: {"empty": ""} for et in ENTITY_TYPES}


def load_entity_df(entity_type: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, f"{entity_type}.csv"))


# ---- entity-string builders (verbatim ports) --------------------------------

def _move_text_within_parentheses(input_str):
    # world_place: "Cathedral (Milan)" -> "Milan's Cathedral "
    match = re.search(r"\((.*?)\)", input_str)
    if match:
        text_within = match.group(1)
        input_str = re.sub(r"\((.*?)\)", "", input_str)
        return f"{text_within}'s {input_str}", True
    return input_str, False


def _world_place_strings(df):
    out = []
    for name in df["name"].values:
        name, processed = _move_text_within_parentheses(name)
        if not processed and "," in name:
            splits = name.split(",")
            name = f"{splits[-1].strip()}'s {','.join(splits[:-1])}"
        out.append(name)
    return out


_NYC_STOP = {"AND", "OR", "OF", "THE", "A", "AT", "&", "IN", "TO"}
_NYC_ABBV = {"FDNY", "NYCT", "YMCA", "LGA", "US", "NYC", "PS", "IS", "NYS",
             "UN", "NY", "EMS", "JCC", "NYU", "CC", "NYPD", "NYPA", "DHS"}


def _nyc_place_strings(df):
    out = []
    for location in df["name"].values:
        words = []
        for word in str(location).split():
            if word.strip() in _NYC_STOP:
                words.append(word.lower())
            elif word.strip() in _NYC_ABBV:
                words.append(word)
            else:
                words.append(word.lower().capitalize())
        out.append(" ".join(words))
    return out


def _art_strings(df):
    out = []
    for _, row in df.iterrows():
        apos = "'s" if str(row.creator)[-1] != "s" else "'"
        out.append(f"{row.creator}{apos} {row.title}")
    return out


def entity_strings(entity_type: str, df: pd.DataFrame) -> list:
    if entity_type == "world_place":
        return _world_place_strings(df)
    if entity_type == "nyc_place":
        return _nyc_place_strings(df)
    if entity_type == "art":
        return _art_strings(df)
    if entity_type == "headline":
        return [str(h) for h in df["headline"].values]  # incl. final period
    if entity_type in ("us_place", "historical_figure"):
        return [str(n) for n in df["name"].values]
    raise ValueError(f"unknown entity_type {entity_type!r}")


# ---- probe targets (port of probe_experiment.get_target_values) -------------

NS_PER_YEAR = 1e9 * 60 * 60 * 24 * 365.25


def target_values(entity_type: str, df: pd.DataFrame):
    """Returns (target array, valid row mask). Places -> (n,2) [lon, lat] like the
    paper; times -> (n,) fractional years. Rows with NaN targets are masked out."""
    feature, is_place = FEATURES[entity_type]
    if is_place:
        target = df[["longitude", "latitude"]].values.astype(np.float64)
        valid = ~np.isnan(target).any(axis=1)
        return target, valid
    if feature == "death_year":
        target = df[feature].values.astype(np.float64)
        valid = ~np.isnan(target)
        return target, valid
    # release_date / pub_date -> fractional year (their NS_PER_YEAR convention,
    # +1970 so projections read as calendar years; affine shift, scores unchanged)
    dt = pd.to_datetime(df[feature], errors="coerce", utc=True)
    valid = dt.notna().values
    target = np.full(len(dt), np.nan)
    # force ns resolution: pandas>=2 may parse as datetime64[us] and int64 would
    # then be microseconds (silent 1000x target error, caught by smoke test)
    ns = dt[valid].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    target[valid] = ns / NS_PER_YEAR + 1970.0
    return target, valid
