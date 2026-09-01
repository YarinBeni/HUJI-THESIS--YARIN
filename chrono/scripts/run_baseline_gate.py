"""P0.4 baseline gate — ridge + PLS probes on cached embeddings.

WHAT. For each (model, layer, site) in the grid: read the corpus
originals' frozen features from EmbStore (ids '{doc_id}::{lang}::orig',
as written by extract_embeddings.py), cross-fit a probe over the frozen
gkf_ruler folds (fit on train rulers, score held-out rulers), and read
the scores out through the SLA section 7 protocol: gkf_rho per fold,
mc_balanced_rho over the out-of-fold scoring, placebo_rho as the leak
detector. Appends results rows (chrono.common.append_results) and
prints + writes a gate verdict block.

WHY cross-fitted: mc draws never refit (SLA section 7) — they need ONE
frozen scoring of every doc. Scoring each doc from the gkf fold that
holds its ruler out gives that frozen vector with zero ruler leakage.

GATE NUMBERS MUST BE RE-PINNED (plan addendum). The plan's legacy gate
(Thalesian "L11" rho = 0.41 +/- 0.02; Qwen3-8B L16 0.36; TF-IDF ~0.29
-- note the plan's L11 cannot be AKK_300m, which only has layers 0..8;
it is either cuneiformBase-400m or a typo, one more reason to re-pin)
was pinned on the 1,202-fragment / 41-ruler frame; the current eligible
corpus is 1,187 / 40 rulers, and the pairwise harness
(v_1/src/phase2/pairs/RESULTS.md) reports macro accuracy with
twin/floor context, not rho. So this script never hard-asserts 0.41:
pass the re-pinned value via --gate-rho to get a PASS/FAIL verdict,
otherwise the verdict prints UNPINNED with the comparison numbers.

    python chrono/scripts/run_baseline_gate.py \
        --model Thalesian/AKK_300m --layers 0-8 --sites mean last
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

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
          "MINIMAL manifest-compatible local reader.", file=sys.stderr)
    STORE_LIB = "local-fallback"

    class EmbStore:  # noqa: D101 — read side of SLA section 6 only
        def __init__(self, root):
            self.root = str(root)
            self.manifest_path = os.path.join(root, "manifest.parquet")

        def _select(self, model, layer, site):
            m = pd.read_parquet(self.manifest_path)
            return m[(m["model"] == str(model))
                     & (m["layer"] == int(layer))
                     & (m["site"] == str(site))]

        def get(self, model, layer, site, ids):
            sel = self._select(model, layer, site)
            where = {r.id: (r.shard, int(r.row))
                     for r in sel.itertuples()}
            missing = [i for i in ids if str(i) not in where]
            if missing:
                raise KeyError(f"EmbStore missing {missing[:5]}...")
            cache, out = {}, []
            for i in ids:
                shard, row = where[str(i)]
                if shard not in cache:
                    with np.load(os.path.join(self.root, shard),
                                 allow_pickle=True) as z:
                        cache[shard] = z["X"]
                out.append(cache[shard][row])
            return np.stack(out).astype(np.float32)

try:
    from chrono.eval import (block_placebo_rho, gkf_rho,  # noqa: E402
                             mc_balanced_rho, placebo_rho, pooled_rho)
    EVAL_LIB = "chrono.eval"
except ImportError as _e:                                # parallel build
    print(f"WARNING: chrono.eval unavailable ({_e}) — using MINIMAL "
          "local rho fallbacks; re-run once A5's protocol lands.",
          file=sys.stderr)
    from scipy import stats
    EVAL_LIB = "local-fallback"

    def _rho(s, t):
        return float(stats.spearmanr(s, t).statistic)

    def mc_balanced_rho(scores, corpus_df, split):
        t = corpus_df.set_index("doc_id")["t"]
        return np.array([_rho(scores.loc[f["test"]], t.loc[f["test"]])
                         for f in split["folds"]])

    def gkf_rho(scores_by_fold, corpus_df, split):
        t = corpus_df.set_index("doc_id")["t"]
        return np.array([
            _rho(scores_by_fold[k].loc[f["test"]], t.loc[f["test"]])
            for k, f in enumerate(split["folds"])])

    def placebo_rho(scores, corpus_df, split, seed):
        g = np.random.default_rng(seed)
        t = corpus_df.set_index("doc_id")["t"]
        return np.array([
            _rho(scores.loc[f["test"]],
                 g.permutation(t.loc[f["test"]].to_numpy()))
            for f in split["folds"]])


PROBES = ("ridge", "pls")
ALPHAS = np.logspace(-2, 6, 17)


def _git_sha() -> str:
    """HEAD from .git files directly — parallel builders must never run
    the git binary (SLA section 0)."""
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


def fit_predict(probe: str, Xtr, ytr, Xte, n_components: int):
    """Standardize on train, fit, return (test scores, mean train
    prediction). Lateness = larger predicted t = later; SLA section 1
    comes for free from y = t."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if probe == "ridge":
        model = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
    elif probe == "pls":
        k = max(1, min(n_components, Xtr.shape[1], len(ytr) - 1))
        try:
            model = PLSRegression(n_components=k).fit(Xtr, ytr)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # a numerically degenerate fold (near-constant features,
            # fewer distinct rows than components) must not kill the
            # whole sweep; the cell reads NaN and the report says why
            print(f"[gate] pls fit failed on a fold ({exc}); NaN scores",
                  flush=True)
            nan_te = np.full(len(Xte), np.nan)
            return nan_te, float("nan")
    else:
        raise ValueError(f"unknown probe {probe!r}")
    return (np.asarray(model.predict(Xte)).ravel(),
            float(np.mean(model.predict(Xtr))))


def cross_fit(probe, X_by_doc: pd.DataFrame, t_by_doc: pd.Series,
              gkf: dict, n_components: int):
    """Fit per gkf fold; return ({fold: test Series}, oof Series).

    The pooled out-of-fold scoring subtracts each fold model's mean
    TRAIN prediction before pooling: fold probes are fit on different
    ruler subsets, so a raw intercept encodes the complement's mean t
    and pooling raw predictions turns held-out-ruler mean shifts into a
    spurious anti-signal for UNinformative features (observed: pure
    noise scoring mc rho ~ -0.93). The offset is train-side only; the
    per-fold gkf rho is shift-invariant either way."""
    by_fold, oof = {}, {}
    for k, fold in enumerate(gkf["folds"]):
        tr, te = fold["train"], fold["test"]
        s, s_tr_mean = fit_predict(
            probe, X_by_doc.loc[tr].to_numpy(),
            t_by_doc.loc[tr].to_numpy(),
            X_by_doc.loc[te].to_numpy(), n_components)
        by_fold[k] = pd.Series(s, index=pd.Index(te, name="doc_id"))
        oof.update(zip(te, s - s_tr_mean))
    return by_fold, pd.Series(oof, name="s").rename_axis("doc_id")


def evaluate(probe, X_by_doc, corpus, gkf, mc, seed, n_components):
    by_fold, oof = cross_fit(probe, X_by_doc, corpus.set_index(
        "doc_id")["t"], gkf, n_components)
    return {
        # REVIEW FIX (wave B1): gkf folds 0/1 hold a single ruler each and
        # 39/40 rulers carry one year, so per-fold rho is undefined there
        # and averaging the survivors answers a different question. The
        # SLA read-out policy for every non-mc split is POOLED.
        "gkf_pooled": pooled_rho(oof, corpus, gkf),
        "gkf_perfold": gkf_rho(by_fold, corpus, gkf),   # diagnostic only
        "mc": mc_balanced_rho(oof, corpus, mc),
        "placebo": placebo_rho(oof, corpus, mc, seed),        # leak check
        # the honest null: t is block-constant within ruler, so the
        # exchangeable unit is the RULER (~8 per draw), not the document
        "block_placebo": block_placebo_rho(oof, corpus, mc, seed),
    }


def _parse_layers(spec):
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _fmt(v: np.ndarray) -> str:
    return f"{np.nanmean(v):+.3f}±{np.nanstd(v):.3f}"


# The verdict cell is fixed BEFORE looking at any chrono result.
# L8 = the top encoder block of Thalesian/AKK_300m (8 blocks +
# embeddings = 9 hidden states, indices 0..8) — the standard
# read-out point for an encoder, and the M.Sc. PLS convention
# (mean pooling, maximal cleaning). An earlier draft said 11,
# a layer index inherited from the 12-block cuneiformBase-400m;
# it does not exist in this encoder, so the verdict would have
# silently fallen through to the selection-inflated best cell.
APRIORI_LAYER, APRIORI_SITE, APRIORI_PROBE = 8, "mean", "pls"


def verdict_block(rows: list, gate_rho, gate_tol) -> str:
    """rows: dicts with probe/layer/site + rho arrays.

    The verdict is read off the A-PRIORI cell (PLS, layer 8, mean — the
    M.Sc. convention), not off the best of the grid; see the review-fix
    note below.
    """
    lines = ["=" * 72, "P0.4 BASELINE GATE", "=" * 72,
             f"{'probe':<6} {'layer':>5} {'site':<5} {'gkf pooled':>12} "
             f"{'mc rho':>14} {'doc placebo':>14} {'BLOCK null':>14}"]
    for r in rows:
        lines.append(f"{r['probe']:<6} {r['layer']:>5} {r['site']:<5} "
                     f"{np.nanmean(r.get('gkf_pooled', np.nan)):>+12.3f} "
                     f"{_fmt(r['mc']):>14} {_fmt(r['placebo']):>14} "
                     f"{_fmt(r.get('block_placebo', np.array([np.nan]))):>14}")
    # REVIEW FIX (wave B1): the verdict used to be max-over-52-cells
    # (9 layers x 2 sites x 2 probes) compared to a fixed threshold on
    # the same data — with ~40 exchangeable ruler blocks a simulation put
    # the max-of-52 of PURE ruler-block noise at rho 0.72, so that gate
    # could not fail. We now report the a-priori cell as the verdict and
    # show the best cell only as selection-inflated context.
    ap = [r for r in rows
          if r["layer"] == APRIORI_LAYER and r["site"] == APRIORI_SITE
          and r["probe"] == APRIORI_PROBE]
    best = max(rows, key=lambda r: np.nanmean(r["mc"]))
    sel = float(np.nanmean(best["mc"]))
    if ap:
        vcell, vlabel = ap[0], (f"a priori: {APRIORI_PROBE} "
                                f"L{APRIORI_LAYER} {APRIORI_SITE}")
    else:
        # restricted sweep (e.g. the selftest grid): the a-priori cell is
        # not in `rows`, so the verdict falls back to the best cell and
        # says so — a selection-inflated verdict must never look pinned.
        vcell, vlabel = best, (f"a-priori cell absent from this sweep — "
                               f"FELL BACK to best of {len(rows)}, "
                               f"SELECTION-INFLATED")
    b = float(np.nanmean(vcell["mc"]))
    blk = np.concatenate([np.atleast_1d(np.asarray(
        r.get("block_placebo", [np.nan]), dtype=float)) for r in rows])
    blk = blk[np.isfinite(blk)] if np.isfinite(blk).any() else blk
    lines += ["-" * 72,
              f"VERDICT CELL ({vlabel}) — mc rho {b:+.3f}",
              f"best-of-{len(rows)} cell (SELECTION-INFLATED, context "
              f"only): {best['probe']} L{best['layer']} {best['site']} "
              f"mc rho {sel:+.3f}",
              f"block null (ruler-level, the honest reference): "
              f"{np.nanmean(blk):+.3f}±{np.nanstd(blk):.3f} — a max over "
              f"{len(rows)} cells of pure ruler-block noise reaches "
              f"~0.72, so never read the best cell as a result."]
    if gate_rho is None:
        lines += [
            "gate reference: UNPINNED — re-pin from "
            "v_1/src/phase2/pairs/RESULTS.md (plan addendum): the "
            "legacy 0.41±0.02 used the 1,202/41-ruler frame, the "
            "eligible corpus is now 1,187/40.",
            "verdict: UNPINNED (comparison only, no pass/fail)"]
    else:
        d = b - gate_rho
        lines.append(f"gate reference: rho={gate_rho:+.3f} "
                     f"tol={gate_tol:.3f} (delta {d:+.3f})")
        if b >= gate_rho - gate_tol:
            lines.append("verdict: PASS" + (
                "  [above reference band — re-pin the gate]"
                if d > gate_tol else ""))
        else:
            lines.append("verdict: FAIL — do not start later phases "
                         "(plan P0.4)")
    lines.append("=" * 68)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Thalesian/AKK_300m",
                    help="model name as stored in the EmbStore manifest")
    ap.add_argument("--layers", default="0-8")
    ap.add_argument("--sites", nargs="+", default=["mean"],
                    choices=["mean", "last"])
    ap.add_argument("--lang", default="akk", choices=["akk", "eng"],
                    help="which corpus original per doc: "
                         "'{doc_id}::{lang}::orig'")
    ap.add_argument("--probes", nargs="+", default=list(PROBES),
                    choices=list(PROBES))
    ap.add_argument("--corpus", default=os.path.join(
        common.ART, "corpus_chrono.parquet"))
    ap.add_argument("--splits-dir",
                    default=os.path.join(common.ART, "splits"))
    ap.add_argument("--store-root",
                    default=os.path.join(common.ART, "emb_store"))
    # k=2, not an arbitrary 20: the M.Sc. PLS probe on THIS corpus swept
    # k in {1,2,3,5} and k=2 was the selected component count at 7 of the
    # 9 mean-pooled layers (L8 included, its best cell). k=5 already had
    # negative R2 at L8, so 20 components would overfit ~40 ruler blocks
    # and fail the gate for a reason that has nothing to do with the
    # representation. Pinned from the prior work, before seeing any
    # chrono number. Source: v_1/src/linear_probing/results/
    # orcc__probe_pls/pls_results_thalesian_akk300m.json
    ap.add_argument("--pls-components", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-rho", type=float, default=None,
                    help="re-pinned reference rho; omit -> UNPINNED")
    ap.add_argument("--gate-tol", type=float, default=0.02)
    ap.add_argument("--report-out", default=os.path.join(
        common.ART, "baseline_gate_report.txt"))
    args = ap.parse_args(argv)

    corpus = pd.read_parquet(args.corpus)
    with open(os.path.join(args.splits_dir, "gkf_ruler.json")) as f:
        gkf = json.load(f)
    with open(os.path.join(args.splits_dir, "mc_balanced.json")) as f:
        mc = json.load(f)
    store = EmbStore(args.store_root)
    ids = (corpus["doc_id"].astype(str)
           + f"::{args.lang}::orig").tolist()

    cfg = {k: v for k, v in vars(args).items()}
    gsha, csha = _git_sha(), common.config_sha(cfg)
    cells, results = [], []
    missing = []
    for layer in _parse_layers(args.layers):
        for site in args.sites:
            # The grid here is a REQUEST; the encoder decides what exists
            # (extract_embeddings --layers all). A cell with nothing at
            # all in the store is skipped with a note. A cell that is
            # merely INCOMPLETE still raises from store.get -- a few
            # missing documents is a bug, not a smaller encoder.
            if not store.has(args.model, layer, site, ids).any():
                missing.append((layer, site))
                continue
            X = pd.DataFrame(store.get(args.model, layer, site, ids),
                             index=pd.Index(corpus["doc_id"],
                                            name="doc_id"))
            # CLUSTER FIX (job 32723): L0/last is the embedding-layer
            # vector of the final token, and every text ends in the same
            # </s>, so the matrix is one row repeated 1,187 times. Ridge
            # returns a constant (mc rho -0.08 +/- 0.000) and PLS divides
            # by zero inside gesdd. A feature matrix with no variance
            # carries no information about anything; skip it and say so.
            n_var = int((X.std(axis=0) > 1e-8).sum())
            if n_var == 0:
                print(f"[gate] L{layer} {site}: all {X.shape[1]} features "
                      "constant across documents, skipped", flush=True)
                missing.append((layer, site, "constant"))
                continue
            for probe in args.probes:
                r = evaluate(probe, X, corpus, gkf, mc, args.seed,
                             args.pls_components)
                cells.append(dict(probe=probe, layer=layer, site=site,
                                  **r))
                extra = json.dumps({
                    "model": args.model, "layer": layer, "site": site,
                    "probe": probe, "lang": args.lang,
                    "eval_lib": EVAL_LIB, "store_lib": STORE_LIB})
                run_id = (f"p04_gate::{args.model}::L{layer}::{site}"
                          f"::{probe}")
                for split, key in (("gkf_ruler", "gkf_pooled"),
                                   ("mc_balanced", "mc")):
                    v = np.atleast_1d(r[key])
                    for metric, val in (("rho_mean", np.nanmean(v)),
                                        ("rho_sd", np.nanstd(v))):
                        results.append(dict(
                            run_id=run_id, git_sha=gsha,
                            config_sha=csha, seed=args.seed,
                            split=split, metric=metric,
                            value=float(val), n=int(len(v)),
                            extra=extra))
                results.append(dict(
                    run_id=run_id, git_sha=gsha, config_sha=csha,
                    seed=args.seed, split="mc_balanced",
                    metric="placebo_rho_mean",
                    value=float(np.nanmean(r["placebo"])),
                    n=int(len(r["placebo"])), extra=extra))
                print(f"[gate] {probe} L{layer} {site}: "
                      f"mc {_fmt(r['mc'])} gkf_pooled {r['gkf_pooled']:+.3f} "
                      f"placebo {_fmt(r['placebo'])}", flush=True)

    path = common.append_results(results)
    if missing:
        print(f"[gate] {len(missing)} (layer, site) cell(s) absent from "
              f"the store, skipped: {missing}", flush=True)
    # Only when the a-priori cell was ASKED FOR: a deliberately narrow
    # sweep keeps verdict_block's documented "restricted grid" fallback,
    # which already labels its verdict selection-inflated.
    apriori_requested = (APRIORI_LAYER in _parse_layers(args.layers)
                         and APRIORI_SITE in args.sites
                         and APRIORI_PROBE in args.probes)
    if apriori_requested and not any(
            c["layer"] == APRIORI_LAYER and c["site"] == APRIORI_SITE
            and c["probe"] == APRIORI_PROBE for c in cells):
        raise SystemExit(
            f"the a-priori verdict cell (probe={APRIORI_PROBE} "
            f"L{APRIORI_LAYER} site={APRIORI_SITE}) is not in the store; "
            "refusing to print a verdict that would silently fall back "
            "to the selection-inflated best cell. Re-run C1 first.")
    block = verdict_block(cells, args.gate_rho, args.gate_tol)
    print(block, flush=True)
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    with open(args.report_out, "w") as f:
        f.write(block + "\n")
    print(f"[gate] {len(results)} result rows -> {path}; report -> "
          f"{args.report_out}", flush=True)


if __name__ == "__main__":
    main()
