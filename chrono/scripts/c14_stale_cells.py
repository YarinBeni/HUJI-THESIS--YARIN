"""Print the C14 array indices whose S1 table is missing or was written by
the pre-2026-09-03 probe script (empty HELD-OUT section). The cell list must
stay in step with chrono/sbatch/C14_reprobe.sbatch."""
import os, re, sys
def safe(n): return re.sub(r"[^0-9A-Za-z_.-]+", "_", n)
CELLS = [("Thalesian/cuneiformBase-400m",6,"mean"),("Thalesian/cuneiformBase-400m",12,"mean"),
         ("Thalesian/AKK_300m",4,"mean"),("Thalesian/AKK_300m",8,"mean"),
         ("NousResearch/Llama-2-7b-hf",16,"mean"),("NousResearch/Llama-2-7b-hf",24,"mean"),
         ("Qwen/Qwen3-8B",18,"mean"),("Qwen/Qwen3-8B",27,"mean")]
for o in ("barlow","byol","jepa","infonce"):
    for e in ("cunei400m","akk300m","llama2_7b","qwen3_8b"): CELLS.append((f"ssl::ssl_{o}_{e}-s0",0,"h"))
for o in ("barlow","jepa"):
    for s in ("S","M","L","XL"): CELLS.append((f"ssl_e2e::e2e_{o}_{s}-s0",0,"h"))
d = sys.argv[1] if len(sys.argv)>1 else "chrono/reports/ssl"
need = []
for i,(m,l,s) in enumerate(CELLS):
    p = os.path.join(d, f"S1_{safe(m)}_L{l}_{s}.md")
    if not os.path.exists(p): need.append(i); continue
    t = open(p).read()
    blk = t.split("HELD-OUT",1)[-1].split("## ",1)[0]
    rows = [r for r in blk.splitlines() if r.startswith("|") and not set(r) <= set("|- ") and "held out" not in r]
    if not rows: need.append(i)
print(",".join(map(str,need)))
