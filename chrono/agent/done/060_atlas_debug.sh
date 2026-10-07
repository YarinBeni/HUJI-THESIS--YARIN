# TIMEOUT=900
# Contradiction: the store holds the deep layers, the rerun printed the full
# inventory and "atlas written", yet B_language_alignment.csv still has the
# old 4-layer llama ladder. Look at the actual files on the cluster and run
# one deep-layer fetch exactly as the atlas does it.
ls -la chrono/reports/tier0/atlas/ | head
wc -l chrono/reports/tier0/atlas/B_language_alignment.csv
python3 - <<'PY'
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from chrono.models.store import EmbStore
store = EmbStore("chrono/artifacts_tier0/emb_store")
man = store.manifest()
print("manifest llama mean layers:",
      sorted(man[(man.model=="NousResearch/Llama-2-7b-hf") & (man.site=="mean")].layer.unique()))
views = pd.read_parquet("chrono/artifacts_tier0/views.parquet")
clean = views[views["augs"].fillna("") == ""].drop_duplicates(["doc_id", "lang"])
for lang in ("akk", "eng"):
    ids = clean[clean.lang == lang].view_id.tolist()
    ok = np.all(store.has("NousResearch/Llama-2-7b-hf", 2, "mean", ids))
    print(f"L2 {lang}: all {len(ids)} ids present -> {bool(ok)}")
    if not ok:
        h = store.has("NousResearch/Llama-2-7b-hf", 2, "mean", ids)
        missing = [i for i, v in zip(ids, h) if not v][:5]
        print("  missing examples:", missing)
PY
