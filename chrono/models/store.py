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

import fcntl
import os
import re
import tempfile

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
            try:
                return pd.read_parquet(self.manifest_path)
            except Exception as exc:  # noqa: BLE001 -- corrupt / half-written
                # CLUSTER FIX (C1v2 jobs 33341-33344): three extract tasks
                # wrote one store concurrently; the manifest was rewritten
                # in place by each put, so readers saw half a file
                # ("Couldn't deserialize thrift") and all three died. The
                # shards themselves are self-describing (see put), so a
                # broken manifest is rebuilt from them instead of being
                # fatal; the write path is locked + atomic from now on.
                print(f"[EmbStore] manifest unreadable ({type(exc).__name__}); "
                      "rebuilding from shards", flush=True)
                return self.rebuild_manifest()
        return pd.DataFrame(columns=MANIFEST_COLS)

    def rebuild_manifest(self) -> pd.DataFrame:
        """Reconstruct manifest.parquet from the shard files. Shards written
        before this fix carry no model/layer/site/text_sha inside and are
        skipped with a note (they are re-extracted on the next run)."""
        rows = []
        skipped = 0
        for fn in sorted(os.listdir(self.root)):
            if not fn.endswith(".npz"):
                continue
            with np.load(os.path.join(self.root, fn), allow_pickle=False) as z:
                if "model" not in z.files:
                    skipped += 1
                    continue
                ids = z["ids"].astype(str)
                rows.append(pd.DataFrame({
                    "id": ids, "model": str(z["model"]),
                    "layer": int(z["layer"]), "site": str(z["site"]),
                    "dim": int(z["dim"]), "shard": fn,
                    "row": np.arange(len(ids), dtype=np.int64),
                    "text_sha": z["text_sha"].astype(str)}))
        m = (pd.concat(rows, ignore_index=True)[MANIFEST_COLS] if rows
             else pd.DataFrame(columns=MANIFEST_COLS))
        if skipped:
            print(f"[EmbStore] rebuild: {skipped} legacy shard(s) without "
                  "metadata skipped", flush=True)
        self._write_manifest(m)
        return m

    def _write_manifest(self, m: pd.DataFrame) -> None:
        """Atomic: write to a temp file in the same directory, then
        os.replace, so a concurrent reader never sees a partial file."""
        fd, tmp = tempfile.mkstemp(prefix=".manifest.", suffix=".parquet",
                                   dir=self.root)
        os.close(fd)
        try:
            m[MANIFEST_COLS].to_parquet(tmp, index=False)
            os.replace(tmp, self.manifest_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

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
        text_sha = np.array([common.sha16(str(t)) for t in texts]
                            if texts is not None else [""] * len(ids), dtype=str)
        # self-describing shard: the manifest can be rebuilt from shards alone
        np.savez_compressed(os.path.join(self.root, shard),
                            ids=np.array(ids, dtype=str), X=X,
                            model=np.array(str(model)), layer=np.array(int(layer)),
                            site=np.array(str(site)), dim=np.array(int(X.shape[1])),
                            text_sha=text_sha)
        rows = pd.DataFrame({
            "id": ids, "model": str(model), "layer": int(layer),
            "site": str(site), "dim": int(X.shape[1]), "shard": shard,
            "row": np.arange(len(ids), dtype=np.int64),
            "text_sha": text_sha,
        })
        # the manifest read-modify-write is serialised across PROCESSES
        # (several extract tasks share one store) and written atomically
        with open(os.path.join(self.root, ".manifest.lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                m = self.manifest()
                if len(m):
                    stale = ((m["model"] == str(model))
                             & (m["layer"] == int(layer))
                             & (m["site"] == str(site))
                             & m["id"].isin(set(ids)))
                    m = m[~stale]
                self._write_manifest(pd.concat([m, rows], ignore_index=True))
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
        return shard

    def _select(self, model, layer, site) -> pd.DataFrame:
        m = self.manifest()
        if not len(m):
            return m
        return m[(m["model"] == str(model)) & (m["layer"] == int(layer))
                 & (m["site"] == str(site))]

    def has(self, model, layer, site, ids, texts=None) -> np.ndarray:
        """Which ids are cached. With `texts` (order-aligned), an id
        counts as cached ONLY if its stored text_sha matches.

        REVIEW FIX (wave B1): resume was keyed on id membership alone,
        so a views.parquet rebuild that changed the TEXT under a stable
        view_id left the cache stale and C1 skipped re-extraction — the
        run would then train on vectors of the old text with nothing in
        the results to show it.
        """
        sel = self._select(model, layer, site)
        if texts is None:
            known = set(sel["id"])
            return np.array([str(i) in known for i in ids], dtype=bool)
        sha = dict(zip(sel["id"], sel["text_sha"]))
        return np.array(
            [sha.get(str(i), None) == common.sha16(str(t))
             for i, t in zip(ids, texts)], dtype=bool)

    def stale(self, model, layer, site, ids, texts) -> list:
        """Cached ids whose stored text_sha disagrees with `texts`."""
        sel = self._select(model, layer, site)
        sha = dict(zip(sel["id"], sel["text_sha"]))
        out = []
        for i, t in zip(ids, texts):
            got = sha.get(str(i))
            if got not in (None, "") and got != common.sha16(str(t)):
                out.append(str(i))
        return out

    def get(self, model, layer, site, ids, texts=None) -> np.ndarray:
        """Gather float32 [len(ids), d] in the order of `ids`; KeyError
        listing every missing id. With `texts`, also raises when any
        cached row was embedded from different text (review fix)."""
        ids = [str(i) for i in ids]
        if texts is not None:
            bad = self.stale(model, layer, site, ids, texts)
            if bad:
                raise KeyError(
                    f"EmbStore has {len(bad)} STALE row(s) for "
                    f"({model}, L{layer}, {site}) — text changed since "
                    f"extraction, e.g. {bad[:10]}. Re-extract or clear "
                    f"the store.")
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
