"""Shared fixtures. The toy corpus mirrors the corpus_chrono schema exactly
(INTERFACES.md section 3) so module tests never need the real parquet."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_corpus():
    rng = np.random.default_rng(0)
    rulers = [("Ashurbanipal", -668, -631), ("Sennacherib", -704, -681),
              ("Sargon II", -721, -705), ("Esarhaddon", -680, -669),
              ("Nabonidus", -555, -539)]
    rows = []
    for ri, (ruler, t0, t1) in enumerate(rulers):
        for k in range(24):
            t = float(rng.integers(t0, t1 + 1))
            body = " ".join(rng.choice(
                ["palace", "temple", "built", "great", "wall", "canal",
                 "foundation", "restored", "gods", "tribute"], size=30))
            rows.append(dict(
                doc_id=f"D{ri}_{k}", ruler=ruler, t=t,
                text_eng=f"{ruler} king of Assyria {body}",
                text_akk=f"{ruler.lower()} sar mat assur {body}",
                text_eng_masked="", text_akk_masked="",
                sub_genre=["prism", "slab", "brick"][k % 3],
                provenance=["Nineveh", "Kalhu", "Assur"][k % 3],
                period="Neo-Assyrian",
                n_words=32,
                ruler_spans_eng=[[0, len(ruler)]],
                ruler_spans_akk=[[0, len(ruler)]],
            ))
    return pd.DataFrame(rows)


@pytest.fixture
def toy_ruler_table(toy_corpus):
    g = toy_corpus.groupby("ruler")["t"]
    return pd.DataFrame({"ruler": g.min().index, "t_min": g.min().values,
                         "t_max": g.max().values, "proxy": True,
                         "n_docs": g.size().values})
