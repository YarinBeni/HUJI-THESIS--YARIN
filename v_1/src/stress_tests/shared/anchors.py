"""P3 (A Matter of Time) timeline anchors.

Anchors are points whose date we KNOW, embedded through the same model. We build
two anchor families from the model's *explicit/declarative* side and later (J8)
fit a 1-D timeline (UMAP / Bezier) through them, then project the ORCC text
embeddings onto it (sub-test 3b) and check the anchors themselves form an ordered
curve (sub-test 3a).

  * ruler anchors — one per ruler, English reign framing ("...reign of {ruler}").
  * year  anchors — one per sampled BCE year ("...the year {year} BCE").

Anchors are English/explicit on purpose: they carry the king->date knowledge T9
confirmed the model has. The contrast 3a(anchors form timeline) vs 3b(texts land
on it) is the dissociation rendered geometrically.
"""
from __future__ import annotations

RULER_TEMPLATES = [
    "an Akkadian royal inscription from the reign of {ruler}",
    "a text written during the reign of {ruler}, king of Assyria and Babylonia",
]
YEAR_TEMPLATES = [
    "an Akkadian royal inscription from the year {year} BCE",
    "a Mesopotamian text written in {year} BCE",
]


def ruler_year_map(df) -> dict[str, int]:
    """ruler -> representative (median) year BCE from the labeled corpus."""
    m = df.dropna(subset=["ruler", "year"]).groupby("ruler")["year"].median()
    return {r: int(round(v)) for r, v in m.items()}


def build_ruler_anchors(df, template_idx: int = 0) -> list[dict]:
    tpl = RULER_TEMPLATES[template_idx]
    anchors = []
    for ruler, year in sorted(ruler_year_map(df).items(), key=lambda kv: kv[1]):
        anchors.append({"kind": "ruler", "ruler": ruler, "year": year,
                        "prompt": tpl.format(ruler=ruler)})
    return anchors


def build_year_anchors(df, step: int = 10, template_idx: int = 0) -> list[dict]:
    """Year anchors spanning the observed BCE range at `step`-year spacing."""
    tpl = YEAR_TEMPLATES[template_idx]
    yrs = df["year"].dropna().astype(int)
    lo, hi = int(yrs.min()), int(yrs.max())
    anchors = []
    for y in range(lo, hi + 1, step):
        anchors.append({"kind": "year", "ruler": None, "year": y,
                        "prompt": tpl.format(year=y)})
    return anchors
