"""Chrono-Barlow trainer (plan P3.1; SLA section 6) — E-MIN's engine.

WHAT. Config-driven training of an AdapterHead over FROZEN features:
each step draws two views per document from the views table (branch A
from menu_a, branch B from the milder menu_b), embeds them (EmbStore
cache for kind 'emb'; char_wb 2-5 TfidfVectorizer fitted on view texts
for the LOCAL SMOKE PATH 'tfidf'), pushes branch A through the online
head and branch B through its EMA twin, and minimizes

    L = bt(p_a, p_b) + lambda_rank * softrank(s_a) + lambda_var * var(s_a)

with softrank pairs drawn from disjoint reign-proxy intervals only
(chrono.losses.make_order_pairs). Everything downstream reads what this
writes: a per-doc scores parquet (run_id, doc_id, condition, s) with
condition='orig', and results rows via chrono.common.append_results.

WHY frozen features + adapter: the science question is whether a small
head can extract a confound-resistant lateness axis from an encoder's
existing geometry — not whether a big model can memorize 1,187 docs.
Scores s are LATENESS (larger = later, astronomical t; SLA section 1).

    python chrono/scripts/train_cjb.py \
        --config chrono/configs/emin_tfidf_smoke.yaml --seed 0

Losses come from chrono.losses (A3's library); if that import fails
(parallel build), private shims in chrono.models._fallback_losses keep
the smoke path alive — with a loud warning.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from chrono import common                                # noqa: E402
from chrono.models.heads import AdapterHead, EmaTwin     # noqa: E402
from chrono.models.store import EmbStore                 # noqa: E402

try:
    from chrono.losses import (bt_loss, make_order_pairs,  # noqa: E402
                               soft_spearman, softrank_loss,
                               variance_loss)
    LOSS_LIB = "chrono.losses"
except ImportError as _e:
    # REVIEW FIX (wave B1): a transitive import error used to swap the
    # real loss library for stubs with only a stderr line as evidence —
    # a whole cluster run could be scored on shims. Opt in explicitly.
    if os.environ.get("CHRONO_ALLOW_FALLBACK_LOSSES") != "1":
        raise ImportError(
            f"chrono.losses failed to import ({_e}). Set "
            "CHRONO_ALLOW_FALLBACK_LOSSES=1 to run on the private "
            "fallback shims — never do this for a science run.") from _e
    print(f"WARNING: running on FALLBACK loss shims ({_e})",
          file=sys.stderr)
    from chrono.models._fallback_losses import (         # noqa: E402
        bt_loss, make_order_pairs, soft_spearman, softrank_loss,
        variance_loss)
    LOSS_LIB = "chrono.models._fallback_losses"

CONFIG_KEYS = {"run_name", "features", "views", "loss", "train",
               "eval_split"}
METRIC_TEMP = 0.05      # near-hard soft-Spearman for reporting
TFIDF_MAX_FEATURES = 5000


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = CONFIG_KEYS - set(cfg)
    if missing:
        raise ValueError(f"config {path} missing keys {sorted(missing)}")
    return cfg


def git_sha() -> str:
    """Read HEAD from .git files directly — running `git` is forbidden
    for parallel builders (SLA section 0)."""
    git = os.path.join(common.REPO, ".git")
    try:
        head = open(os.path.join(git, "HEAD")).read().strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(None, 1)[1]
        p = os.path.join(git, *ref.split("/"))
        if os.path.exists(p):
            return open(p).read().strip()[:12]
        for line in open(os.path.join(git, "packed-refs")):
            if line.strip().endswith(ref):
                return line.split()[0][:12]
    except OSError:
        pass
    return "unknown"


def _chain_str(chain) -> str:
    return ",".join(chain)


def _safe_spearman(a, b) -> float:
    """spearmanr, but NaN (silently) on degenerate input — e.g. a fold
    whose test docs all share one composition year."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def build_features(cfg: dict, views_df: pd.DataFrame, store=None,
                   fit_doc_ids=None):
    """Feature matrix over ALL rows of views_df (row-aligned).

    kind 'tfidf': char_wb 2-5 tfidf. REVIEW FIX (wave B1): when
    `fit_doc_ids` is given the vectorizer is FITTED ON THOSE DOCS' VIEWS
    ONLY and merely transforms the rest — otherwise vocabulary and idf
    are learned from held-out text and every held-out battery cell is
    quietly transductive, biasing chrono against honestly cross-fitted
    baselines. kind 'emb': EmbStore lookup by view_id; extraction is
    unsupervised so the question does not arise there.
    Returns float32 [n_views, d] (dense).
    """
    feats = cfg["features"]
    kind = feats["kind"]
    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                              max_features=TFIDF_MAX_FEATURES,
                              dtype=np.float32)
        texts = views_df["text"].astype(str).tolist()
        if fit_doc_ids is None:
            X = vec.fit_transform(texts)
        else:
            keep = views_df["doc_id"].isin(set(fit_doc_ids)).to_numpy()
            if not keep.any():
                raise ValueError("fit_doc_ids matched no views")
            vec.fit([t for t, k in zip(texts, keep) if k])
            X = vec.transform(texts)
        return np.asarray(X.todense(), dtype=np.float32)
    if kind == "emb":
        if store is None:
            raise ValueError("features.kind='emb' needs an EmbStore")
        return store.get(feats["model"], feats["layer"], feats["site"],
                         views_df["view_id"].tolist())
    raise ValueError(f"unknown features.kind: {kind!r}")


def _view_index(views_df, doc_ids, menu, seeds):
    """doc_id -> list of views_df ROW POSITIONS whose augs chain is in
    `menu` and seed in `seeds`."""
    want_augs = {_chain_str(c) for c in menu}
    want_seeds = {int(s) for s in seeds}
    ok = (views_df["augs"].isin(want_augs)
          & views_df["seed"].astype(int).isin(want_seeds))
    idx = {d: [] for d in doc_ids}
    for pos, (doc, hit) in enumerate(zip(views_df["doc_id"], ok)):
        if hit and doc in idx:
            idx[doc].append(pos)
    return idx


def _condition_scores(head, X, views_df, doc_ids):
    """Per-doc lateness for EVERY augmentation condition present.

    REVIEW FIX (wave B1): this used to emit condition='orig' only, so
    chrono.eval.battery — whose whole point is the condition x split grid
    of P3.4 — could never return more than one row per split. The
    robustness claim of the plan ("degrades <= half as much as PLS under
    name-masking / formula-removal") was therefore not computable from
    anything chrono wrote. Every augs chain in views_df becomes one
    condition ('' -> 'orig'), scored as the mean s over that doc's views
    of that chain. Returns {condition: np.ndarray aligned to doc_ids}.
    """
    head.eval()
    with torch.no_grad():
        _, s_all, _ = head(torch.as_tensor(X))
    s_all = s_all.numpy()
    head.train()
    doc_col = views_df["doc_id"].to_numpy()
    augs_col = views_df["augs"].to_numpy()
    out = {}
    for chain in pd.unique(augs_col):
        cond = "orig" if chain == "" else str(chain)
        by_doc = {}
        for p in np.flatnonzero(augs_col == chain):
            by_doc.setdefault(doc_col[p], []).append(p)
        vals = []
        for d in doc_ids:
            ps = by_doc.get(d)
            vals.append(float(s_all[ps].mean()) if ps else np.nan)
        arr = np.array(vals, dtype=np.float64)
        if cond == "orig" and not np.isfinite(arr).all():
            miss = [d for d, v in zip(doc_ids, arr) if not np.isfinite(v)]
            raise ValueError(f"docs without an orig view: {miss[:10]}")
        out[cond] = arr
    return out


def train(cfg: dict, corpus_df: pd.DataFrame, views_df: pd.DataFrame, *,
          store=None, ruler_table: pd.DataFrame | None = None,
          fold: int | None = None, split: dict | None = None,
          out_dir: str | None = None, write: bool = True,
          log_every: int = 50) -> dict:
    """Run the P3.1 loop; returns {run_id, metrics, scores, loss_curve,
    scores_path}. `write` gates BOTH the scores parquet and the results
    rows (tests run write=False). With `split` (a splits/<name>.json
    dict) and `fold`, training restricts to that fold's train docs and
    test_spearman is reported on its test docs."""
    seed = int(cfg["train"]["seed"])
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)

    views_df = views_df.reset_index(drop=True)
    corpus_df = corpus_df.reset_index(drop=True)
    have_views = set(views_df["doc_id"])
    docs = corpus_df[corpus_df["doc_id"].isin(have_views)]
    train_docs = docs
    if split is not None and fold is not None:
        keep = set(split["folds"][fold]["train"])
        train_docs = docs[docs["doc_id"].isin(keep)]
    train_docs = train_docs.reset_index(drop=True)
    doc_ids = train_docs["doc_id"].tolist()
    n = len(doc_ids)
    if n < 2:
        raise ValueError("need >= 2 training docs with views")

    if ruler_table is None:
        gt = train_docs.groupby("ruler")["t"]
        ruler_table = pd.DataFrame({
            "ruler": gt.min().index, "t_min": gt.min().values,
            "t_max": gt.max().values, "proxy": True,
            "n_docs": gt.size().values})

    # featurizer sees TRAIN docs only whenever we are inside a fold
    X = build_features(cfg, views_df, store,
                       fit_doc_ids=(doc_ids if (split is not None
                                                and fold is not None)
                                    else None))
    idx_a = _view_index(views_df, doc_ids, cfg["views"]["menu_a"],
                        cfg["views"]["seeds"])
    idx_b = _view_index(views_df, doc_ids, cfg["views"]["menu_b"],
                        cfg["views"]["seeds"])
    empty = [d for d in doc_ids if not idx_a[d] or not idx_b[d]]
    if empty:
        raise ValueError(f"{len(empty)} docs have no view in a menu, "
                         f"e.g. {empty[:5]}")

    resample_pairs = bool(cfg.get("train", {}).get("resample_pairs", True))

    def draw_pairs(epoch):
        """Order constraints for one epoch.

        REVIEW FIX (wave B1): a single frozen draw is 2,428 constraints
        in which 20% of docs never appear and the median ruler-pair
        contributes ONE pair — the quota is min(m, n_i, n_j) and most of
        the 40 rulers are long-tail. Redrawing per epoch is the plan's
        combinatorial-supervision promise (plan section 2) actually
        delivered: measured coverage goes 80.1% -> 99.9% after five
        redraws and 100% by ten, at no extra GPU cost (the pair sampler
        is pure numpy over doc ids). resample_pairs=false restores the
        frozen-draw behaviour for ablations.
        """
        sd = seed if not resample_pairs else int(seed) * 100003 + epoch
        pr, mg, _ = make_order_pairs(train_docs, ruler_table,
                                     per_ruler_pair=21, seed=sd)
        return (torch.as_tensor(np.asarray(pr), dtype=torch.long),
                torch.as_tensor(np.asarray(mg, dtype=np.float32)))

    pairs, margins = draw_pairs(0)

    lcfg, tcfg = cfg["loss"], cfg["train"]
    head = AdapterHead(d_in=X.shape[1])
    twin = EmaTwin(head, momentum=0.996)
    opt = torch.optim.Adam(head.parameters(), lr=float(tcfg["lr"]))
    batch = int(tcfg["batch"])
    epochs = int(tcfg["epochs"])
    t_train = train_docs["t"].to_numpy(dtype=float)

    loss_curve = []
    for epoch in range(epochs):
        if resample_pairs and epoch:
            pairs, margins = draw_pairs(epoch)
        order = g.permutation(n)
        ep_loss, nb = 0.0, 0
        for lo in range(0, n, batch):
            take = order[lo:lo + batch]
            rows_a = [idx_a[doc_ids[i]][g.integers(
                len(idx_a[doc_ids[i]]))] for i in take]
            rows_b = [idx_b[doc_ids[i]][g.integers(
                len(idx_b[doc_ids[i]]))] for i in take]
            xa = torch.as_tensor(X[rows_a])
            xb = torch.as_tensor(X[rows_b])
            _, s_a, p_a = head(xa)
            _, _, p_b = twin(xb)
            loss = bt_loss(p_a, p_b,
                           lambda_offdiag=float(lcfg["lambda_offdiag"]))
            # pairs live in train-doc positions; keep those fully inside
            # this batch and remap to batch-local positions
            loc = torch.full((n,), -1, dtype=torch.long)
            loc[torch.as_tensor(take)] = torch.arange(len(take))
            li, lj = loc[pairs[:, 0]], loc[pairs[:, 1]]
            m = (li >= 0) & (lj >= 0)
            if m.any():
                loss = loss + float(lcfg["lambda_rank"]) * softrank_loss(
                    s_a, torch.stack([li[m], lj[m]], 1), margins[m],
                    temp=float(lcfg["temp"]))
            loss = loss + float(lcfg["lambda_var"]) * variance_loss(s_a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            twin.update()
            ep_loss += float(loss.detach())
            nb += 1
        loss_curve.append(ep_loss / nb)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch + 1}/{epochs} loss {loss_curve[-1]:.4f}",
                  file=sys.stderr)

    # ---- score every doc (condition 'orig') + metrics ----------------
    all_ids = docs["doc_id"].tolist()
    cond_scores = _condition_scores(head, X, views_df, all_ids)
    s_all = cond_scores["orig"]
    s_by_doc = dict(zip(all_ids, s_all))
    s_train = np.array([s_by_doc[d] for d in doc_ids])
    rho_soft = float(soft_spearman(
        torch.tensor(s_train), torch.tensor(t_train),
        temp=METRIC_TEMP))
    rho_hard = _safe_spearman(s_train, t_train)
    metrics = {"train_soft_spearman": rho_soft,
               "train_spearman": rho_hard,
               "final_loss": loss_curve[-1]}
    if split is not None and fold is not None:
        want = list(split["folds"][fold]["test"])
        te = [d for d in want if d in s_by_doc]
        if len(te) != len(want):                    # review fix: no silent
            miss = [d for d in want if d not in s_by_doc][:10]
            raise KeyError(
                f"{len(want) - len(te)} test docs have no score "
                f"(views/EmbStore incomplete), e.g. {miss}")
        t_by_doc = dict(zip(docs["doc_id"], docs["t"]))
        metrics["test_spearman"] = _safe_spearman(
            [s_by_doc[d] for d in te], [t_by_doc[d] for d in te])

    split_name = cfg["eval_split"] or "all"
    run_id = f"{cfg['run_name']}-s{seed}" + \
        (f"-f{fold}" if fold is not None else "")
    # `fit` is scoring PROVENANCE, consumed by whoever reads the parquet:
    # 'full' = head (and featurizer) saw every doc, so held-out battery
    # cells are transductive and must be labelled as such; 'oof' = this
    # run trained on one fold's train docs only, so its test docs are
    # honest and per-fold runs can be concatenated into a pooled-OOF
    # Series for chrono.eval.pooled_rho.
    fit_tag = "oof" if (split is not None and fold is not None) else "full"
    test_set = (set(split["folds"][fold]["test"])
                if (split is not None and fold is not None) else set())
    frames = []
    for cond, arr in cond_scores.items():
        frames.append(pd.DataFrame({
            "run_id": run_id, "doc_id": all_ids, "condition": cond,
            "s": arr, "fit": fit_tag,
            "fold": (-1 if fold is None else int(fold)),
            # REVIEW FIX: is_test marks the rows a pooled-OOF read-out may
            # use; s_rank is the fold-local rank of s among that fold's
            # test docs, because heads trained on different folds share no
            # scale and raw s cannot be concatenated across folds.
            "is_test": [d in test_set for d in all_ids]}))
    scores = pd.concat(frames, ignore_index=True)
    if test_set:
        m = scores["is_test"].to_numpy()
        scores["s_rank"] = np.nan
        for cond in scores["condition"].unique():
            sel = m & (scores["condition"] == cond).to_numpy()
            v = scores.loc[sel, "s"].to_numpy(dtype=float)
            scores.loc[sel, "s_rank"] = (stats.rankdata(v) / max(len(v), 1))
    else:
        scores["s_rank"] = np.nan

    scores_path = None
    if write:
        out_dir = out_dir or os.path.join(common.ART, "scores")
        os.makedirs(out_dir, exist_ok=True)
        scores_path = os.path.join(out_dir, f"{run_id}.parquet")
        scores.to_parquet(scores_path, index=False)
        sha, gsha = common.config_sha(cfg), git_sha()
        extra = json.dumps({"features": cfg["features"]["kind"],
                            "epochs": epochs, "fold": fold,
                            "loss_lib": LOSS_LIB})
        common.append_results([
            dict(run_id=run_id, git_sha=gsha, config_sha=sha, seed=seed,
                 split=split_name, metric=k, value=v,
                 n=len(te) if k == "test_spearman" else n, extra=extra)
            for k, v in metrics.items()])

    return {"run_id": run_id, "metrics": metrics, "scores": scores,
            "loss_curve": loss_curve, "scores_path": scores_path}


def _load_split(name: str) -> dict:
    with open(os.path.join(common.ART, "splits", f"{name}.json")) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="overrides train.seed")
    ap.add_argument("--fold", type=int, default=None,
                    help="fold of eval_split to hold out")
    ap.add_argument("--corpus", default=os.path.join(
        common.ART, "corpus_chrono.parquet"))
    ap.add_argument("--views", default=os.path.join(
        common.ART, "views.parquet"))
    ap.add_argument("--ruler-table", default=os.path.join(
        common.ART, "ruler_table.parquet"))
    ap.add_argument("--store-root", default=os.path.join(
        common.ART, "emb_store"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    corpus = pd.read_parquet(args.corpus)
    if os.path.exists(args.views):
        views = pd.read_parquet(args.views)
    else:  # smoke fallback: build exactly the views the config needs
        from chrono.augment.engine import build_views
        menu = [list(c) for c in
                {tuple(c) for c in (cfg["views"]["menu_a"]
                                    + cfg["views"]["menu_b"])} | {()}]
        views = build_views(corpus, sorted(menu),
                            cfg["views"]["seeds"])
    ruler_table = pd.read_parquet(args.ruler_table) \
        if os.path.exists(args.ruler_table) else None
    store = EmbStore(args.store_root) \
        if cfg["features"]["kind"] == "emb" else None
    split = _load_split(cfg["eval_split"]) if cfg["eval_split"] else None
    fold = args.fold if split is not None else None
    if split is not None and fold is None:
        fold = 0

    res = train(cfg, corpus, views, store=store,
                ruler_table=ruler_table, fold=fold, split=split)
    print(json.dumps({"run_id": res["run_id"],
                      "metrics": res["metrics"],
                      "scores_path": res["scores_path"]}, indent=2))


if __name__ == "__main__":
    main()
