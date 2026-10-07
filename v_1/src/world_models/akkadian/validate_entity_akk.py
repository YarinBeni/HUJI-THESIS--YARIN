"""Sanity experiments for the Akkadian name harvester (advisor request).

The harvester chose each king's attested spellings from the FIRST personal
name of his documents. This script does not assume the choice is right or
wrong. It measures it, four ways:

E1  Histogram. For the 12 best-attested kings: the top first-PN spellings in
    their documents, with counts, and which ones the harvester chose.
    Output: a PNG of 12 bar plots + the full table in the report.
E2  Coverage. Per king: the share of his documents whose first PN is one of
    his chosen spellings. Low coverage = the chosen names are not how his
    documents actually open.
E3  Ruler identification (the strong test). Classify every document by WHICH
    king's chosen spellings appear in it (first PN; tie -> most docs). If the
    spellings are right, this must agree with the corpus's ruler metadata.
    Output: accuracy + the worst confusions.
E4  Leak check. For each chosen spelling: in how many OTHER kings' documents
    does it appear anywhere in the text? High = the spelling is not specific.

Writes results/VALIDATE_ENTITY_AKK.md and results/validate_histograms.png.

    python validate_entity_akk.py
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from build_entity_akk import CORPUS, pn_strings, clean, harvest   # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

OUT_MD = os.path.join(_HERE, "results", "VALIDATE_ENTITY_AKK.md")
OUT_PNG = os.path.join(_HERE, "results", "validate_histograms.png")


def main():
    corpus = pd.read_parquet(CORPUS)
    chosen = harvest(corpus)                       # ruler -> [(spelling, n_docs)]
    chosen_set = {ru: {s for s, _ in v} for ru, v in chosen.items()}

    # first PN per document
    first_pn, all_pns = {}, {}
    for r in corpus.itertuples():
        names = pn_strings(str(r.text_akk or ""), str(r.text_akk_masked or ""))
        if names:
            first_pn[r.doc_id] = clean(names[0])
            all_pns[r.doc_id] = [clean(n) for n in names]

    docs = corpus[corpus.doc_id.isin(first_pn)].copy()
    docs["first_pn"] = docs.doc_id.map(first_pn)
    lines = ["# Harvester validation (E1–E4)", "",
             f"{len(docs):,} documents carry at least one [PN]; "
             f"{len(chosen)} kings got a chosen spelling.", ""]

    # ---- E1: histograms ---------------------------------------------------
    top = docs.ruler.value_counts().head(12).index
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), squeeze=False)
    for ax, ru in zip(axes.ravel(), top):
        c = Counter(docs.loc[docs.ruler == ru, "first_pn"]).most_common(8)
        labels = [s[:22] for s, _ in c]
        colors = ["#2a9d8f" if s in chosen_set.get(ru, set()) else "#aaaaaa"
                  for s, _ in c]
        ax.barh(range(len(c))[::-1], [n for _, n in c], color=colors)
        ax.set_yticks(range(len(c))[::-1], labels, fontsize=7)
        ax.set_title(ru, fontsize=9)
    fig.suptitle("First-PN spellings per king (green = chosen by the harvester)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    lines += ["**E1** `validate_histograms.png` — green bars are the harvester's "
              "choices; grey bars are what it rejected.", ""]

    # ---- E2: coverage -----------------------------------------------------
    lines += ["## E2 — coverage per king", "",
              "| king | docs with PN | first PN is a chosen spelling |", "|---|---|---|"]
    for ru in sorted(chosen, key=lambda r: -int((docs.ruler == r).sum())):
        g = docs[docs.ruler == ru]
        cov = g.first_pn.isin(chosen_set[ru]).mean() if len(g) else np.nan
        lines.append(f"| {ru} | {len(g)} | {cov:.0%} |")

    # ---- E3: ruler identification ------------------------------------------
    owner = {}
    for ru, pairs in chosen.items():
        for s, n in pairs:
            if s not in owner or n > owner[s][1]:
                owner[s] = (ru, n)
    pred = docs.first_pn.map(lambda s: owner.get(s, (None,))[0])
    m = pred.notna() & docs.ruler.isin(chosen)
    acc = (pred[m] == docs.ruler[m]).mean()
    conf = (pd.DataFrame({"true": docs.ruler[m], "pred": pred[m]})
            .query("true != pred").groupby(["true", "pred"]).size()
            .sort_values(ascending=False).head(8))
    lines += ["", "## E3 — ruler identification from the chosen spellings", "",
              f"Documents whose first PN is a chosen spelling: {int(m.sum()):,}. "
              f"Predicted king == metadata king: **{acc:.1%}**.", "",
              "Worst confusions (true -> predicted, count):", ""]
    for (t, p), n in conf.items():
        lines.append(f"- {t} -> {p}: {n}")

    # ---- E4: leakage --------------------------------------------------------
    lines += ["", "## E4 — spelling specificity (appears anywhere in other kings' texts)",
              "", "| spelling | king | his docs | other kings' docs |", "|---|---|---|---|"]
    text_of = corpus.set_index("doc_id")["text_akk"].astype(str)
    ruler_of = corpus.set_index("doc_id")["ruler"]
    for ru, pairs in sorted(chosen.items()):
        for s, n in pairs:
            hits = text_of.str.contains(s, regex=False)
            others = int((hits & (ruler_of != ru)).sum())
            lines.append(f"| `{s}` | {ru} | {n} | {others} |")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w").write("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD} and {OUT_PNG}; E3 accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
