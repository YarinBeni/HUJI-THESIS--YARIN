#!/bin/bash
# Status for the PLS k-sweep jobs. Both commit_push their own results, so completion is
# visible from the committed files alone — no cluster access needed.
#
# Counts CELLS, not commits: the WAk re-run pushed extra "WAk:" commits for the seven
# cells corrupted by the truncated-write bug, so a commit count overshoots 15 and never
# reads as done. A fragment cell is finished when its mc_group block carries the full
# 11-point pls_per_k; a cell-B file is finished when it carries pls_per_k at all.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
git fetch -q origin main 2>/dev/null
git merge-base --is-ancestor HEAD origin/main 2>/dev/null && git merge -q --ff-only origin/main 2>/dev/null
python3 - <<'EOF'
import glob, json
frag_done = frag_tot = 0
for f in glob.glob('v_1/src/world_models/akkadian/results/probes/*/*.r8.year.*.ridge.json'):
    try:
        d = json.load(open(f)).get('mc_group')
    except Exception:
        print(f"  ! malformed: {f}"); continue
    if not isinstance(d, dict):
        continue
    frag_tot += 1
    if len(d.get('pls_per_k') or {}) >= 11:
        frag_done += 1

ent_done = ent_tot = 0
for f in glob.glob('v_1/src/world_models/akkadian/results/probes_entity/*/*.json'):
    try:
        d = json.load(open(f))
    except Exception:
        print(f"  ! malformed: {f}"); continue
    ent_tot += 1
    bl = str(d.get('best_layer'))
    blk = (d.get('layers') or {}).get(bl) or {}
    if (blk.get('all') or {}).get('pls_per_k'):
        ent_done += 1

print(f"WAk fragments {frag_done}/{frag_tot} cells | WBk entity {ent_done}/{ent_tot} files")
print("DONE" if frag_done == frag_tot and ent_tot and ent_done == ent_tot else "PENDING")
EOF
