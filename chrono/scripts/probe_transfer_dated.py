"""Does a representation transfer to the DATED royal inscriptions?

The per-corpus period probe in probe_representations.py answered this with
balanced accuracy, and on this corpus that number is not usable: the dated
inscriptions are 924 Neo-Assyrian, 216 Neo-Babylonian, 28 Middle Babylonian,
3 Achaemenid, 1 Hellenistic, while the undated pool that trains the probe has
5 Neo-Babylonian and 52 Middle Babylonian texts. Balanced accuracy over the
surviving classes therefore averaged a class with 924 test docs, a class with
28, and a class with ONE -- which is how those cells came out "below chance".

Two honest read-outs instead, both fit on the undated corpora only and
evaluated on the dated inscriptions the SSL runs never saw:

1. TRANSFER RHO (headline). Regress the embedding on an approximate period
   midpoint year over the undated pool, predict the dated inscriptions, and
   take Spearman against their true year. Continuous, so no class-imbalance
   artefact, and directly comparable with the rho the thesis reports.
2. Classification, but only over classes with >= MIN_TEST test docs AND
   >= MIN_TRAIN training docs, with per-class recall printed next to the
   counts so a degenerate cell is visible instead of silently scored.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono.models.store import EmbStore                        # noqa: E402

# Conventional midpoints (BCE, negative). Approximate on purpose: they only
# have to order the periods, since the read-out is a rank correlation.
MIDPOINT = {"Old Assyrian": -1850, "Old Babylonian": -1750, "Middle Babylonian": -1300,
            "Middle Assyrian": -1225, "Neo-Assyrian": -760, "Neo-Babylonian": -580,
            "Achaemenid": -435, "Hellenistic": -200, "Late Babylonian": -320}
MIN_TEST, MIN_TRAIN = 20, 50


def cells(store: EmbStore) -> list[tuple]:
    m = store.manifest()
    if m.empty:
        return []
    return sorted({(r.model, int(r.layer), r.site) for r in m.itertuples()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="chrono/artifacts_ssl/corpus_all.parquet")
    ap.add_argument("--store-root", default="chrono/artifacts_ssl/emb_store")
    ap.add_argument("--out", default="chrono/reports/ssl/TRANSFER_DATED.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    c = pd.read_parquet(args.corpus)
    tr = c[(c.split != "dated") & c.period_norm.notna() & c.period_norm.isin(MIDPOINT)].reset_index(drop=True)
    te = c[(c.split == "dated") & c.year.notna()].reset_index(drop=True)
    store = EmbStore(args.store_root)

    rows = []
    for model, layer, site in cells(store):
        ids_tr = ("ssl::" + tr.uid.astype(str)).tolist()
        ids_te = ("ssl::" + te.uid.astype(str)).tolist()
        try:
            if not (np.all(store.has(model, layer, site, ids_tr))
                    and np.all(store.has(model, layer, site, ids_te))):
                continue
            Xtr = store.get(model, layer, site, ids_tr).astype(np.float32)
            Xte = store.get(model, layer, site, ids_te).astype(np.float32)
        except Exception as exc:                       # noqa: BLE001
            print(f"[skip] {model} L{layer} {site}: {type(exc).__name__}: {exc}", flush=True)
            continue
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xte)

        y_mid = tr.period_norm.map(MIDPOINT).to_numpy(float)
        pred = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(A, y_mid).predict(B)
        yr = te.year.to_numpy(float)
        rho = spearmanr(pred, yr).statistic
        # 924 of the 1,176 dated inscriptions are Neo-Assyrian, so the overall
        # rho mostly measures "is this Neo-Assyrian or not". The thesis task is
        # the harder one inside a single period, so report that separately.
        m_na = (te.period_norm == "Neo-Assyrian").to_numpy()
        rho_na = spearmanr(pred[m_na], yr[m_na]).statistic if m_na.sum() >= 50 else np.nan

        # classification, honest class filter
        vc_tr, vc_te = tr.period_norm.value_counts(), te.period_norm.value_counts()
        keep = [k for k in vc_te.index if vc_te[k] >= MIN_TEST and vc_tr.get(k, 0) >= MIN_TRAIN]
        acc, per_class = np.nan, ""
        if len(keep) >= 2:
            mtr, mte = tr.period_norm.isin(keep).to_numpy(), te.period_norm.isin(keep).to_numpy()
            le = LabelEncoder().fit(tr.period_norm[mtr])
            clf = LogisticRegression(max_iter=3000, C=0.5).fit(A[mtr], le.transform(tr.period_norm[mtr]))
            yt, yp = le.transform(te.period_norm[mte]), clf.predict(B[mte])
            acc = balanced_accuracy_score(yt, yp)
            rec = recall_score(yt, yp, average=None, labels=np.arange(len(le.classes_)), zero_division=0)
            per_class = ", ".join(f"{cl} {r:.2f} (n={int(vc_te.get(cl, 0))})"
                                  for cl, r in zip(le.classes_, rec) if cl in keep)
        rows.append(dict(model=f"{model}::L{layer}::{site}", n_train=len(tr), n_test=len(te),
                         rho=rho, rho_na=rho_na, classes=len(keep), acc=acc, per_class=per_class))
        print(f"{model} L{layer} {site}: rho={rho:+.3f} rho_NA={rho_na:+.3f} acc={acc:.3f}", flush=True)

    if not rows:
        raise SystemExit("no cell had embeddings for both the undated pool and the dated inscriptions")
    R = pd.DataFrame(rows).sort_values("rho", ascending=False)
    lines = ["# Transfer to the dated royal inscriptions", "",
             f"Fit on {len(tr):,} undated texts (period midpoint as the target), evaluated on "
             f"{len(te):,} dated royal inscriptions with a known year that no SSL run ever saw. "
             "`rho` is Spearman between the predicted year and the true year — the number to read. "
             f"`acc` is balanced accuracy over the periods with >= {MIN_TEST} test and >= {MIN_TRAIN} "
             "training documents, with per-class recall beside it; earlier tables scored classes "
             "with as little as one test document and are superseded here.", "",
             f"`rho within NA` repeats it over the {int((te.period_norm == 'Neo-Assyrian').sum()):,} "
             "Neo-Assyrian inscriptions alone — the same question the thesis asks, with the easy "
             "between-period contrast removed.", "",
             "", "**On `rho within NA`, read the ceiling before the number.** The training target here "
             "is a period MIDPOINT, which is one constant value for every Neo-Assyrian text, so a model "
             "fit on it cannot resolve time inside that period however good its representation is. A "
             "near-zero column is what this target predicts; it is not evidence that the signal is "
             "absent. Within-period dating is answered with real years, by the thesis-protocol read-out "
             "in `EMIN_SSL_RESULT.md` (C18).",
             "| model | transfer rho | rho within NA | classes | bal acc | per-class recall |",
             "|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        a = "" if pd.isna(r.acc) else f"{r.acc:.3f}"
        na = "" if pd.isna(r.rho_na) else f"{r.rho_na:+.3f}"
        lines.append(f"| `{r.model}` | {r.rho:+.3f} | {na} | {r.classes} | {a} | {r.per_class} |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(R)} cells)")


if __name__ == "__main__":
    main()
