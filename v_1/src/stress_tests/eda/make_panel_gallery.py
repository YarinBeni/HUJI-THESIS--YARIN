"""make_panel_gallery.py — build a single self-contained index.html that shows
every embedding panel, grouped cleaning -> reduction, with model tabs. No
dependencies, no server: open embedding_panels/index.html in any browser.

Usage:  python v_1/src/stress_tests/eda/make_panel_gallery.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "e6_clusters" / "embedding_panels"
CLEANINGS = ["maximal", "engtier0"]
REDUCTIONS = ["tsne", "pca", "umap", "pls"]

cards = []
for cl in CLEANINGS:
    for red in REDUCTIONS:
        d = ROOT / cl / red
        if not d.is_dir():
            continue
        imgs = sorted(p.name for p in d.glob("*.png"))
        if not imgs:
            continue
        thumbs = "".join(
            f'<figure><img loading="lazy" src="{cl}/{red}/{n}" '
            f'alt="{n}"><figcaption>{n[:-4]}</figcaption></figure>'
            for n in imgs)
        cards.append(
            f'<section><h2>{cl} &middot; {red.upper()} '
            f'<span class="n">({len(imgs)} models)</span></h2>'
            f'<div class="grid">{thumbs}</div></section>')

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Embedding panels gallery</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f5f7;color:#1d1d1f}}
 header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;z-index:9}}
 header h1{{margin:0;font-size:18px}}
 header p{{margin:4px 0 0;font-size:13px;color:#666}}
 section{{padding:8px 20px 24px}}
 h2{{font-size:15px;border-left:4px solid #0a7;padding-left:8px}}
 h2 .n{{color:#999;font-weight:normal;font-size:12px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}}
 figure{{margin:0;background:#fff;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden}}
 figure img{{width:100%;display:block;cursor:zoom-in}}
 figcaption{{font-size:12px;padding:6px 8px;color:#333;font-weight:600}}
 dialog{{border:none;background:transparent;max-width:96vw;max-height:96vh;padding:0}}
 dialog img{{max-width:96vw;max-height:96vh}}
 dialog::backdrop{{background:rgba(0,0,0,.85)}}
</style></head><body>
<header><h1>Embedding panels &mdash; six views of each map (year / ruler / period / sub-genre / provenance / length)</h1>
<p>ORCC royal inscriptions &middot; grouped cleaning &middot; reduction &middot; click any panel to zoom. See table1_best_models.csv for the top-3 per experiment.</p></header>
{''.join(cards)}
<dialog id="zoom"><img id="zi"></dialog>
<script>
 const dlg=document.getElementById('zoom'),zi=document.getElementById('zi');
 document.querySelectorAll('.grid img').forEach(im=>im.onclick=()=>{{zi.src=im.src;dlg.showModal();}});
 dlg.onclick=()=>dlg.close();
</script>
</body></html>"""

out = ROOT / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({len(cards)} groups)")
