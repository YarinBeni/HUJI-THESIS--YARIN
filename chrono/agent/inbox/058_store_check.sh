# TIMEOUT=600
# The atlas rerun saw 16 llama layers in the manifest yet produced outputs
# identical to the 4-layer run. Decisive check: does the store actually HOLD
# the new layers for the clean tier0 view ids?
python3 - <<'PY'
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, ".")
from chrono.models.store import EmbStore
store = EmbStore("chrono/artifacts_tier0/emb_store")
views = pd.read_parquet("chrono/artifacts_tier0/views.parquet")
clean = views[views["augs"].fillna("") == ""].drop_duplicates(["doc_id", "lang"])
ids = clean[clean.lang == "akk"].view_id.head(50).tolist()
for model, Ls in [("NousResearch/Llama-2-7b-hf", [2, 4, 8, 30]),
                  ("allenai/OLMo-2-1124-7B", [3, 15, 32]),
                  ("Qwen/Qwen3-8B", [3, 18, 33])]:
    for L in Ls:
        try:
            h = store.has(model, L, "mean", ids)
            print(f"{model} L{L}: has {int(np.sum(h))}/50")
        except Exception as e:
            print(f"{model} L{L}: {type(e).__name__}: {e}")
PY
