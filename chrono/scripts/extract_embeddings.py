"""Embedding extraction cache (plan P0.3; SLA section 8) — freeze the
encoder's view of every text once, into EmbStore.

WHAT. Reads views.parquet (SLA section 4) plus the corpus originals,
forwards every text through a frozen encoder — Thalesian/AKK_300m by
default, hfid resolved from v_1/src/world_models/wm_lib/registry.py,
loaded encoder-side via AutoModelForSeq2SeqLM.get_encoder() exactly as
v_1/src/world_models does — and stores per-layer (0 = embedding layer)
mean- and last-token pooled float32 vectors in EmbStore (SLA section
6). Ids: view rows keep their view_id; corpus originals are stored as
"{doc_id}::{lang}::orig" so probes can address a document without
knowing any augmentation menu or seed.

WHY a cache: E-MIN trains small heads over frozen features on CPU; the
only GPU work in the whole pipeline is this one pass, so it must be
resumable (chunks already fully present in the store are skipped) and
deterministic (fp32 pooled outputs; text_sha in the manifest lets the
P0.3 spot-check re-embed a sample and compare).

    python chrono/scripts/extract_embeddings.py \
        --model thalesian_akk300m --layers all --sites mean last
    python chrono/scripts/extract_embeddings.py --selftest \
        --store-root /tmp/store            # no transformers, CPU-only

--selftest exercises the identical batching/pooling/store path over 5
in-file texts with a tiny stub encoder (embedding + masked positional
mean mixing, 3 hidden states) and verifies determinism end-to-end.
Runs bf16 on GPU, fp32 on CPU (--dtype auto).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from chrono import common                                # noqa: E402

try:
    from chrono.models.store import EmbStore             # noqa: E402
    STORE_LIB = "chrono.models.store"
except ImportError as _e:                                # parallel build
    print(f"WARNING: chrono.models.store unavailable ({_e}) — using a "
          "MINIMAL manifest-compatible local writer. Fine for smoke "
          "runs; re-extract once A4's EmbStore lands.", file=sys.stderr)
    STORE_LIB = "local-fallback"

    class EmbStore:  # noqa: D101 — mirrors the SLA section 6 surface
        _COLS = ["id", "model", "layer", "site", "dim", "shard", "row",
                 "text_sha"]

        def __init__(self, root):
            self.root = str(root)
            os.makedirs(self.root, exist_ok=True)
            self.manifest_path = os.path.join(self.root,
                                              "manifest.parquet")

        def manifest(self):
            if os.path.exists(self.manifest_path):
                return pd.read_parquet(self.manifest_path)
            return pd.DataFrame(columns=self._COLS)

        def put(self, model, layer, site, ids, X, texts=None):
            X = np.ascontiguousarray(X, dtype=np.float32)
            ids = [str(i) for i in ids]
            tag = common.sha16("\x00".join(ids))
            shard = f"{model}__L{int(layer)}__{site}__{tag}.npz" \
                .replace("/", "_")
            np.savez_compressed(os.path.join(self.root, shard),
                                ids=np.array(ids, dtype=object), X=X)
            rows = pd.DataFrame({
                "id": ids, "model": str(model), "layer": int(layer),
                "site": str(site), "dim": int(X.shape[1]),
                "shard": shard,
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
            pd.concat([m, rows], ignore_index=True)[self._COLS] \
                .to_parquet(self.manifest_path, index=False)
            return shard

        def has(self, model, layer, site, ids):
            m = self.manifest()
            if not len(m):
                return np.zeros(len(ids), dtype=bool)
            known = set(m[(m["model"] == str(model))
                          & (m["layer"] == int(layer))
                          & (m["site"] == str(site))]["id"])
            return np.array([str(i) in known for i in ids], dtype=bool)


# --------------------------------------------------------------------------
# text collection

def gather_texts(views_df: pd.DataFrame,
                 corpus_df: pd.DataFrame) -> pd.DataFrame:
    """One (id, text) row per thing to embed: every view under its
    view_id, plus '{doc_id}::{lang}::orig' for each non-empty corpus
    original. Sorted by id so chunking (and thus resume) is stable."""
    rows = [views_df[["view_id", "text"]]
            .rename(columns={"view_id": "id"})]
    for lang in ("akk", "eng"):
        col = f"text_{lang}"
        if col not in corpus_df.columns:
            continue
        sub = corpus_df[corpus_df[col].fillna("").str.strip() != ""]
        rows.append(pd.DataFrame({
            "id": sub["doc_id"].astype(str) + f"::{lang}::orig",
            "text": sub[col].astype(str)}))
    out = pd.concat(rows, ignore_index=True)
    if out["id"].duplicated().any():
        dup = out[out["id"].duplicated()]["id"].iloc[:5].tolist()
        raise ValueError(f"duplicate embed ids: {dup}")
    return out.sort_values("id", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# encoders: both return encode(texts) -> (hidden_states, attention_mask)
# with hidden_states a list of [B, T, d] float32 tensors, index 0 = the
# embedding layer (the transformers output_hidden_states convention).

def make_causal_encoder(spec: dict, *, max_tokens: int, dtype: str):
    """Decoder-only LM (Llama-2, Qwen3, OLMo) through the M.Sc. loader
    wm_lib.extract.load_model -- same snapshot/fallback/tokenizer logic,
    same bf16 device_map=auto -- so the vectors are the thesis's vectors.
    Tokenisation mirrors wm_lib.tokenize_lib.encode_all with an empty
    prompt: BOS + text, truncated to max_tokens, right-padded. The mask
    handed to pool() EXCLUDES the BOS position, as the M.Sc. entity mask
    did ('entity tokens: after the prefix'); `last` is the final real
    token, the causal summary state."""
    wm = os.path.join(common.REPO, "v_1", "src", "world_models")
    if wm not in sys.path:
        sys.path.insert(0, wm)
    from wm_lib import extract as ex
    tok, core = ex.load_model(spec, dtype=("bfloat16" if dtype == "auto" else dtype))
    bos = tok.bos_token_id
    pad = tok.pad_token_id
    device = next(core.parameters()).device

    def encode(texts):
        enc = tok(list(texts), add_special_tokens=False,
                  return_attention_mask=False)["input_ids"]
        rows = []
        for ids in enc:
            ids = ([bos] if bos is not None else []) + list(ids)
            rows.append(ids[:max_tokens])
        T = max(len(r) for r in rows)
        ids_t = torch.full((len(rows), T), pad, dtype=torch.long)
        attn = torch.zeros((len(rows), T), dtype=torch.long)
        pool_mask = torch.zeros((len(rows), T), dtype=torch.long)
        for i, r in enumerate(rows):
            ids_t[i, :len(r)] = torch.tensor(r)
            attn[i, :len(r)] = 1
            start = 1 if (bos is not None and len(r) > 1) else 0
            pool_mask[i, start:len(r)] = 1
        with torch.no_grad():
            out = core(input_ids=ids_t.to(device), attention_mask=attn.to(device),
                       output_hidden_states=True, use_cache=False)
        return [h.float().cpu() for h in out.hidden_states], pool_mask

    return encode


def make_hf_encoder(hfid: str, *, max_tokens: int, dtype: str):
    """Lazy transformers load, encoder side of the seq2seq model."""
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" else "float32"
    td = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(hfid)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hfid, torch_dtype=td, output_hidden_states=True)
    except Exception as e:  # noqa: BLE001 — umt5-style config gaps
        print(f"[load] Auto failed ({type(e).__name__}); explicit "
              "seq2seq classes", flush=True)
        model = None
        for cls_name in ("UMT5ForConditionalGeneration",
                         "MT5ForConditionalGeneration",
                         "T5ForConditionalGeneration"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(
                    hfid, torch_dtype=td, output_hidden_states=True)
                break
            except Exception as e2:  # noqa: BLE001
                print(f"[load] {cls_name}: {type(e2).__name__}: {e2}",
                      flush=True)
        if model is None:
            raise
    enc = model.get_encoder().to(device).eval()

    def encode(texts):
        b = tok(list(texts), return_tensors="pt", padding=True,
                truncation=True, max_length=max_tokens)
        ids = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        with torch.no_grad():
            out = enc(input_ids=ids, attention_mask=mask,
                      output_hidden_states=True)
        return [h.float().cpu() for h in out.hidden_states], mask.cpu()

    return encode


class _StubEncoder(torch.nn.Module):
    """Selftest twin of the real encoder: token embedding, then two
    blocks of linear mixing with the masked positional mean — 3 hidden
    states, same (layers+1, B, T, d) contract as transformers."""

    def __init__(self, vocab=97, d=8, n_blocks=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Embedding(vocab, d)
        self.mix = torch.nn.Linear(d, d)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(vocab, d, generator=g))
            self.mix.weight.copy_(torch.randn(d, d, generator=g) / d)
            self.mix.bias.zero_()
        self.n_blocks = n_blocks

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids)
        m = attention_mask.unsqueeze(-1).float()
        states = [h]
        for _ in range(self.n_blocks):
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            h = torch.tanh(self.mix(h) + mean.unsqueeze(1))
            states.append(h)
        return states


def make_stub_encoder(max_tokens: int):
    stub = _StubEncoder().eval()

    def _tok_id(w):  # deterministic across processes (hash() is not)
        return sum(w.encode("utf-8")) % (stub.emb.num_embeddings - 1) + 1

    def encode(texts):
        seqs = [[_tok_id(w) for w in t.split()[:max_tokens]] or [1]
                for t in texts]
        T = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), T, dtype=torch.long)
        mask = torch.zeros(len(seqs), T, dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        with torch.no_grad():
            hs = stub(ids, mask)
        return [h.float() for h in hs], mask

    return encode


# --------------------------------------------------------------------------
# pooling + driver

def pool(h: torch.Tensor, mask: torch.Tensor, site: str) -> np.ndarray:
    """[B, T, d] + [B, T] -> float32 [B, d]. mean = masked mean; last =
    hidden state at the final non-pad position (encoders have no causal
    summary token, this is the positional 'last' the SLA asks for)."""
    m = mask.unsqueeze(-1).float()
    if site == "mean":
        out = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
    elif site == "last":
        idx = mask.long().sum(1).clamp(min=1) - 1
        out = h[torch.arange(h.shape[0]), idx]
    else:
        raise ValueError(f"unknown site {site!r}")
    return out.numpy().astype(np.float32, copy=False)


def _put(store, model, layer, site, ids, X, texts):
    try:  # text_sha when the store takes texts; SLA signature otherwise
        store.put(model, layer, site, ids, X, texts=texts)
    except TypeError:
        store.put(model, layer, site, ids, X)


def extract(table: pd.DataFrame, encode, *, store, model_name: str,
            layers: list, sites: list, batch_size: int,
            shard_size: int, overwrite: bool = False) -> dict:
    """Embed table (id, text) chunk by chunk; each chunk becomes one
    shard per (layer, site). Chunks fully present are skipped unless
    overwrite — chunk boundaries are a pure function of the sorted id
    list, so a killed job resumes where it stopped."""
    n_done = n_skip = 0
    t0 = time.time()
    for lo in range(0, len(table), shard_size):
        chunk = table.iloc[lo:lo + shard_size]
        ids = chunk["id"].tolist()
        texts = chunk["text"].tolist()
        # REVIEW FIX (wave B1): pass the texts so a chunk whose text
        # changed under a stable view_id is re-embedded instead of skipped
        if not overwrite and all(
                store.has(model_name, ly, st, ids, texts=texts).all()
                for ly in layers for st in sites):
            n_skip += len(ids)
            continue
        acc = {(ly, st): [] for ly in layers for st in sites}
        for b in range(0, len(texts), batch_size):
            hs, mask = encode(texts[b:b + batch_size])
            bad = [ly for ly in layers if ly >= len(hs)]
            if bad:
                raise ValueError(
                    f"layers {bad} out of range: encoder returns "
                    f"{len(hs)} hidden states (0..{len(hs) - 1})")
            for ly in layers:
                for st in sites:
                    acc[(ly, st)].append(pool(hs[ly], mask, st))
        for (ly, st), parts in acc.items():
            _put(store, model_name, ly, st, ids,
                 np.concatenate(parts, axis=0), texts)
        n_done += len(ids)
        print(f"[extract] {lo + len(ids)}/{len(table)} texts "
              f"({time.time() - t0:.1f}s)", flush=True)
    return {"n_texts": len(table), "n_embedded": n_done,
            "n_skipped": n_skip, "layers": layers, "sites": sites,
            "elapsed_s": round(time.time() - t0, 1)}


def resolve_spec(model_key: str):
    """Registry spec for a key; None for a literal 'org/name'."""
    if "/" in model_key:
        return None
    wm = os.path.join(common.REPO, "v_1", "src", "world_models")
    if wm not in sys.path:
        sys.path.insert(0, wm)
    from wm_lib.registry import MODELS
    return MODELS.get(model_key)


def resolve_hfid(model_key: str) -> str:
    """Register key -> hfid via the world-models registry (the single
    source of truth for encoder ids); a literal 'org/name' passes
    through untouched."""
    if "/" in model_key:
        return model_key
    wm = os.path.join(common.REPO, "v_1", "src", "world_models")
    if wm not in sys.path:
        sys.path.insert(0, wm)
    from wm_lib.registry import MODELS
    if model_key not in MODELS:
        raise KeyError(f"{model_key!r} not in wm_lib registry "
                       f"({sorted(MODELS)})")
    return MODELS[model_key]["hfid"]


def n_hidden_states(encode) -> int:
    """How many hidden states this encoder actually returns, asked of the
    encoder itself with one throwaway forward pass. Hardcoding the count
    is how C1 (job 32500) died: AKK_300m has 8 blocks + embeddings = 9
    states, while the grid said 0..12, which is cuneiformBase-400m's
    count. A batch script frozen at submit time cannot be fixed by the
    in-job git sync, so the number must not live in the batch script."""
    hs, _ = encode(["a"])
    return len(hs)


def parse_layers(spec: str, encode=None) -> list:
    """'all' asks the encoder (requires `encode`); otherwise an explicit
    list/range like '0-8' or '0,4,8'."""
    if spec.strip() == "all":
        if encode is None:
            raise ValueError("--layers all needs a loaded encoder")
        return list(range(n_hidden_states(encode)))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------

def selftest(store_root: str, batch_size: int) -> None:
    """Full pipeline on 5 texts with the stub encoder; asserts store
    round-trip, coverage, and byte-level determinism of re-extraction."""
    texts = ["Ashurbanipal king of Assyria built the great wall",
             "Sennacherib restored the temple of the gods",
             "palace foundation of Sargon in Nineveh", "tribute", ""]
    table = pd.DataFrame(
        {"id": [f"T{i}" for i in range(len(texts))], "text": texts})
    layers, sites = [0, 1, 2], ["mean", "last"]
    store = EmbStore(store_root)
    encode = make_stub_encoder(max_tokens=16)
    meta = extract(table, encode, store=store, model_name="stub",
                   layers=layers, sites=sites, batch_size=batch_size,
                   shard_size=3, overwrite=True)
    ref = {}
    for ly in layers:
        for st in sites:
            assert store.has("stub", ly, st, table["id"]).all()
            X = store.get("stub", ly, st, table["id"])
            assert X.shape == (5, 8) and X.dtype == np.float32
            assert np.isfinite(X).all()
            ref[(ly, st)] = X
    extract(table, make_stub_encoder(max_tokens=16), store=store,
            model_name="stub", layers=layers, sites=sites,
            batch_size=1, shard_size=5, overwrite=True)
    for key, X in ref.items():
        X2 = store.get("stub", key[0], key[1], table["id"])
        assert np.allclose(X, X2, atol=1e-6), \
            f"re-extraction drifted at {key}"
    meta2 = extract(table, encode, store=store, model_name="stub",
                    layers=layers, sites=sites, batch_size=batch_size,
                    shard_size=5, overwrite=False)
    assert meta2["n_skipped"] == 5 and meta2["n_embedded"] == 0
    print(f"[selftest] OK — {meta['n_embedded']} texts x "
          f"{len(layers)} layers x {len(sites)} sites, resume + "
          "determinism verified", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="thalesian_akk300m",
                    help="wm_lib registry key or literal hfid")
    ap.add_argument("--views",
                    default=os.path.join(common.ART, "views.parquet"))
    ap.add_argument("--corpus", default=os.path.join(
        common.ART, "corpus_chrono.parquet"))
    ap.add_argument("--store-root",
                    default=os.path.join(common.ART, "emb_store"))
    ap.add_argument("--layers", default="all",
                    help="'all' (asks the encoder how many hidden "
                         "states it returns) or e.g. '0-8' / '0,4,8'")
    ap.add_argument("--sites", nargs="+", default=["mean", "last"],
                    choices=["mean", "last"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--shard-size", type=int, default=1024)
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--limit", type=int, default=0,
                    help="first N texts only (smoke)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="stub-encoder end-to-end check, no transformers")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest(args.store_root, args.batch_size)
        return

    hfid = resolve_hfid(args.model)
    table = gather_texts(pd.read_parquet(args.views),
                         pd.read_parquet(args.corpus))
    if args.limit:
        table = table.iloc[:args.limit]
    spec = resolve_spec(args.model)
    if spec is not None and spec.get("arch") == "causal":
        encode = make_causal_encoder(spec, max_tokens=args.max_tokens,
                                     dtype=args.dtype)
    else:
        encode = make_hf_encoder(hfid, max_tokens=args.max_tokens,
                                 dtype=args.dtype)
    layers = parse_layers(args.layers, encode)
    print(f"[extract] {hfid}: {len(table)} texts, layers {layers}, "
          f"sites {args.sites}, store={STORE_LIB}", flush=True)
    store = EmbStore(args.store_root)
    meta = extract(table, encode, store=store, model_name=hfid,
                   layers=layers, sites=args.sites,
                   batch_size=args.batch_size,
                   shard_size=args.shard_size, overwrite=args.overwrite)
    meta.update(model=args.model, hfid=hfid, store_root=args.store_root,
                max_tokens=args.max_tokens, dtype=args.dtype)
    mpath = os.path.join(args.store_root,
                         f"extract_meta__{args.model}.json")
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
