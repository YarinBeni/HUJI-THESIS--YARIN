"""Architecture diagram for our 37M sign-level MLM, drawn in the style of the
Ithaca/Aeneas Nature figure: input cells, a positional-information node, the
torso with its sub-layer stack and residual arrows, and task heads. Differences
from theirs, which are the point of the figure: a single sign-level input row
(no word row), and a single restoration head (no region or date heads; our year
and place come from probes on the frozen activations, drawn dashed).

    python plot_mlm_arch.py    # -> results/figs/fig_mlm_arch.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "figs", "fig_mlm_arch.png")

INK = "#1c2028"
BLUE = "#2c6fad"          # arrows, feedforward outline
PURPLE = "#9c27b0"        # attention
GREEN = "#2e7d32"         # add & normalize
GREY_FILL = "#d9dce1"     # masked cells
BOX_BG = "#f7f8fa"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "text.color": INK})


def cell(ax, x, y, w, h, text, masked=False, italic=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                fc=GREY_FILL if masked else "white",
                                ec="#9aa0a8", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=15, style="italic" if italic else "normal", color=INK)


def block(ax, x, y, w, h, text, color, fs=13.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.1",
                                fc="white", ec=color, lw=2.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK)


def arrow(ax, p0, p1, color=BLUE, lw=2.0, style="-|>", ls="solid", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(11.6, 8.2), dpi=150)
    ax.set_xlim(0, 116)
    ax.set_ylim(0, 82)
    ax.axis("off")

    # ---------------- inputs ----------------
    ax.text(20, 79.0, "Inputs", fontsize=17, fontweight="bold")
    ax.text(13.5, 73.6, "Signs", fontsize=15, ha="right")
    cw, ch, y0 = 6.4, 4.6, 71.4
    x = 15.5
    sign_cells = [("–", True), ("–", True), ("–", True), None,
                  ("šar", False), None,
                  ("KUR", False), ("aš", False), ("šur", False)]
    xs = []
    for c in sign_cells:
        if c is None:
            x += 2.0
            continue
        cell(ax, x, y0, cw, ch, c[0], masked=c[1], italic=not c[1])
        xs.append(x)
        x += cw
    row_r = x
    ax.text(xs[0] + 1.5 * cw, y0 - 2.3, "[masked name]", fontsize=11,
            color="#6b7484", ha="center")
    ax.text(xs[3] + 0.5 * cw, y0 - 2.3, "\u201ck\u0131ng\u201d".replace("\u0131","i"), fontsize=11,
            color="#6b7484", ha="center")
    ax.text(xs[5] + 1.4 * cw, y0 - 2.3, "\u201cof Assyria\u201d", fontsize=11,
            color="#6b7484", ha="center")

    # ---------------- torso geometry first, so everything can align ----------
    tx, tw = 64.0, 49.0
    ty, th = 18.0, 44.0
    bx, bw = tx + 4.0, tw - 11.0

    # ---------------- positional information ----------------
    px = bx + bw / 2
    py = ty + th + 6.5
    arrow(ax, (row_r + 1, y0 + ch / 2), (px, y0 + ch / 2))
    arrow(ax, (px, y0 - 0.4), (px, py + 2.6))
    ax.add_patch(plt.Circle((px, py), 2.1, fc="white", ec=BLUE, lw=2.0))
    ax.text(px, py, "\u21bb", ha="center", va="center", fontsize=13, color=BLUE)
    ax.text(px - 4.6, py, "positional\ninformation (RoPE)", fontsize=12,
            va="center", ha="right")
    arrow(ax, (px, py - 2.3), (px, ty + th))

    # ---------------- torso ----------------
    ax.text(tx + 1.0, ty + th - 0.5, "Torso", fontsize=17, fontweight="bold", va="bottom")
    ax.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="square,pad=0",
                                fc=BOX_BG, ec="#b9bec7", lw=1.3))
    ax.text(tx + 2.0, ty + th - 3.4, "16\u00d7", fontsize=15, fontweight="bold")

    labels = [("Multi-head self-attention\n8 heads \u00b7 d = 384", PURPLE, 8.4),
              ("Add and normalize (RMSNorm)", GREEN, 5.8),
              ("Feed-forward 384 \u2192 1536 \u2192 384", BLUE, 5.8),
              ("Add and normalize (RMSNorm)", GREEN, 5.8)]
    ys = []
    yb = ty + th - 5.0
    for text, color, h in labels:
        yb -= h + (2.6 if ys else 0.0)
        block(ax, bx, yb, bw, h, text, color, 12)
        ys.append((yb, h))
    for (ya, ha), (ybt, hb) in zip(ys, ys[1:]):
        arrow(ax, (bx + bw / 2, ya), (bx + bw / 2, ybt + hb), lw=1.7)
    arrow(ax, (bx + bw - 1.5, ys[0][0] + ys[0][1] + 2.2),
          (bx + bw + 1.8, ys[1][0] + ys[1][1] / 2), lw=1.5, rad=-0.4)
    arrow(ax, (bx + bw - 0.5, ys[1][0]),
          (bx + bw + 1.8, ys[3][0] + ys[3][1] / 2), lw=1.5, rad=-0.4)

    # ---------------- task head + output ----------------
    ax.text(15.5, 50.0, "Outputs", fontsize=17, fontweight="bold")
    ax.text(38.5, 50.0, "Task head", fontsize=17, fontweight="bold")
    hy = 40.5
    hx, hw = 37.0, 17.0
    block(ax, hx, hy, hw, 5.6, "Feedforward", BLUE, 13.5)
    arrow(ax, (tx - 0.5, hy + 2.8), (hx + hw + 0.6, hy + 2.8))
    ax.text(13.5, hy + 2.8, "Restoration", fontsize=15, ha="right", va="center")
    rx = 15.5
    for spred in ["aš", "šur", "PAB"]:
        cell(ax, rx, hy + 0.4, cw, ch, spred, masked=True)
        rx += cw
    arrow(ax, (hx - 0.4, hy + 2.8), (rx + 0.6, hy + 2.8))
    ax.text(15.5 + 1.5 * cw, hy - 2.2, "predicted masked signs \u00b7 cross-entropy",
            fontsize=11, color="#6b7484", ha="center")

    # ---------------- probe read-out (not a head) ----------------
    dy, dh = 5.5, 5.6
    arrow(ax, (px, ty), (px, dy + dh), color="#8a8f98", lw=1.6, ls=(0, (4, 3)))
    ax.add_patch(FancyBboxPatch((10.0, dy), 93.0, dh,
                                boxstyle="round,pad=0.25,rounding_size=1.1",
                                fc="#fdfaf0", ec="#c8b568", lw=1.6,
                                linestyle=(0, (4, 3))))
    ax.text(10.0 + 46.5, dy + dh / 2, "frozen per-layer activations  \u2192  "
            "linear probes (year \u00b7 place)  \u2014 read-out, not a trained head",
            ha="center", va="center", fontsize=12, color="#6b5a14")

    fig.savefig(OUT, facecolor="white", bbox_inches="tight")
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
