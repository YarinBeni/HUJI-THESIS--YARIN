"""One place that decides output resolution and format for every figure.

The figures were being written at 120-220 dpi, which is fine on screen and unusable in
a talk or a paper: at 120 dpi a 9-inch figure is ~1080 px wide, so an 8 pt footnote
lands at roughly 13 px and turns to mush the moment it is projected or printed.

`save(fig, path)` writes:
  * PNG at FIG_DPI (default 300 — print-quality raster, ~2-3x the old size)
  * PDF alongside it, vector, so text stays sharp at any zoom. This is the one to
    embed in LaTeX / the thesis PDF; the PNG is for the HTML deck.

Override per run:
    FIG_DPI=450 python3 slopegraph.py     # poster
    FIG_PDF=0   python3 slopegraph.py     # skip the vector copy
"""
import os

DPI = int(os.environ.get("FIG_DPI", "300"))
WANT_PDF = os.environ.get("FIG_PDF", "1") != "0"


def save(fig, path, **kw):
    """Write `path` (a .png) at print resolution, plus a sibling .pdf."""
    kw.setdefault("facecolor", "white")
    kw.setdefault("bbox_inches", "tight")
    fig.savefig(path, dpi=DPI, **kw)
    out = [path]
    if WANT_PDF and path.lower().endswith(".png"):
        pdf = path[:-4] + ".pdf"
        try:
            # metadata/dpi are ignored by the PDF backend; text stays as text
            fig.savefig(pdf, **kw)
            out.append(pdf)
        except Exception as e:                                   # noqa: BLE001
            print(f"  (pdf skipped: {type(e).__name__}: {e})")
    print("saved " + " + ".join(os.path.basename(p) for p in out) + f"  [{DPI} dpi]")
    return out
