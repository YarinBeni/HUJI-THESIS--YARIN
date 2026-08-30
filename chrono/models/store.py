"""EmbStore — sharded on-disk embedding cache (plan P3.2; SLA section 6).

WHAT. A write-once store for view embeddings: float32 [n, d] blocks land
as compressed .npz shards under a root directory, indexed by a single
manifest.parquet with EXACT columns (id, model, layer, site, dim, shard,
row, text_sha). `put` writes a shard + manifest rows, `get` gathers rows
back in the caller's id order, `has` answers membership; a `get` with any
missing id raises KeyError naming the ids, because a silent partial read
is how stale caches poison downstream Spearmans.

WHY sharded + manifested: extraction runs on the cluster in (model,
layer, site) chunks over ~10^4 view texts, while training reads small
per-batch id sets on CPU; the manifest gives O(1) row lookup without
loading every shard, and text_sha lets P3.2's spot-check verify that a
cached vector still matches the text it was extracted from. Shard names
are a pure function of (model, layer, site, ids), so re-extraction
overwrites in place instead of accumulating orphans.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from chrono import common

MANIFEST_COLS = ["id", "model", "layer", "site", "dim", "shard", "row",
                 "text_sha"]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


class EmbStore:
    """Sharded .npz embedding store under `root` (SLA section 6)."""

    def __init__(self, root: str):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.parquet")

    def manifest(self) -> pd.DataFrame:
        if os.path.exists(self.manifest_path):
            return pd.read_parquet(self.manifest_path)
        return pd.DataFrame(columns=MANIFEST_COLS)

    @staticmethod
    def _shard_name(model, layer, site, ids) -> str:
        tag = common.sha16("\x00".join(str(i) for i in ids))
        return f"{_safe(model)}__L{int(layer)}__{_safe(site)}__{tag}.npz"

    def put(self, model, layer, site, ids, X, texts=None) -> str:
        """Store float32 X [n, d] for `ids` (order-aligned). `texts`
        (optional, order-aligned) fills text_sha; else ''. Re-putting an
        id under the same (model, layer, site) replaces it. Returns the
        shard filename."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        ids = [str(i) for i in ids]
        if X.ndim != 2 or X.shape[0] != len(ids):
            raise ValueError(f"X must be [len(ids), d]; got {X.shape} "
                             f"for {len(ids)} ids")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in a single put")
        if texts is not None and len(texts) != len(ids):
            raise ValueError("texts must align with ids")
        shard = self._shard_name(model, layer, site, ids)
        np.savez_compressed(os.path.join(self.root, shard),
                            ids=np.array(ids, dtype=str), X=X)
        rows = pd.DataFrame({
            "id": ids, "model": str(model), "layer": int(layer),
            "site": str(site), "dim": int(X.shape[1]), "shard": shard,
            "row": np.arange(len(ids), dtype=np.int64),
            "text_sha": ([common.sha16(str(t)) for t in texts]
                         if texts is not None else [""] * len(ids)),
        })
        m = self.manifest()
        if len(m):
            stale = ((m["model"] == str(model))
                     & (m["layer"] == int(layer))
                     & (m["site"] == str(site))
                     & m["id"].isin(set(ids)))
            m = m[~stale]
        pd.concat([m, rows], ignore_index=True)[MANIFEST_COLS] \
            .to_parquet(self.manifest_path, index=False)
        return shard

    def _select(self, model, layer, site) -> pd.DataFrame:
        m = self.manifest()
        if not len(m):
            return m
        return m[(m["model"] == str(model)) & (m["layer"] == int(layer))
                 & (m["site"] == str(site))]

    def has(self, model, layer, site, ids) -> np.ndarray:
        known = set(self._select(model, layer, site)["id"])
        return np.array([str(i) in known for i in ids], dtype=bool)

    def get(self, model, layer, site, ids) -> np.ndarray:
        """Gather float32 [len(ids), d] in the order of `ids`; KeyError
        listing every missing id."""
        ids = [str(i) for i in ids]
        sel = self._select(model, layer, site)
        where = {r.id: (r.shard, int(r.row)) for r in sel.itertuples()}
        missing = [i for i in ids if i not in where]
        if missing:
            shown = ", ".join(missing[:20])
            more = "" if len(missing) <= 20 else \
                f" (+{len(missing) - 20} more)"
            raise KeyError(
                f"EmbStore missing {len(missing)} id(s) for "
                f"({model}, L{layer}, {site}): {shown}{more}")
        cache, out = {}, []
        for i in ids:
            shard, row = where[i]
            if shard not in cache:
                with np.load(os.path.join(self.root, shard)) as z:
                    cache[shard] = z["X"]
            out.append(cache[shard][row])
        return np.stack(out).astype(np.float32, copy=False)
