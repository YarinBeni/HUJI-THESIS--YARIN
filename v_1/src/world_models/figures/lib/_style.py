"""Shared visual language for every world-models figure.

THIS FILE IS SLAVED TO THE DECK. The palette, labels, ordering and type scale below are
copied from `world_models/plot_cellA_figs.py`, which is what generated the figures
already embedded in `thesis_story_9.html`. Anything rendered here has to sit next to
those slides, so a colour must mean the same thing in both places.

The earlier palette here broke that in one important way: it coloured uMT5-base purple,
while the deck reserves purple for the CONTROL arms (random-init twins). A reader moving
between a deck slide and one of these figures would have read "purple = untrained" on
one and "purple = the multilingual encoder" on the other. It also re-used each trained
arm's own colour for its random twin, distinguishing them by dash alone, which does not
survive projection.

The deck's scheme, now used here too:

  BLUES   Qwen3 family + gpt-oss, light -> dark with size
  GREENS  Llama-2 family, light -> dark with size
  WARM    the three translation encoders (uMT5 -> AKK-300M -> cuneiform-400M)
  PURPLE  random-init controls, always dashed
  BLACK   TF-IDF floor, always dotted

Type scale is also the deck's (base 14pt, not 8pt) — these figures are projected.
"""
import numpy as np

# --- palette: identical to plot_cellA_figs.COLORS ---------------------------------
COL = {
    # blues: decoder family 1
    "qwen3_1b7": "#7cc0f8", "qwen3_8b": "#2b8ae8", "qwen3_32b": "#1252b3",
    "gpt_oss_120b": "#0a2a5e",
    # greens: decoder family 2
    "llama2_7b": "#7fd39b", "llama2_13b": "#25a35c", "llama2_70b": "#0a5c31",
    # teal: OLMo is a family of one here, and teal is unused by the blue/green/warm
    # families, so it reads as "a new arm" rather than as a member of one
    "olmo2_7b": "#0f8b8d",
    # warm: the three translation encoders
    "umt5_base": "#f0b429", "thalesian_akk300m": "#f2711c",
    "thalesian_cunei400m": "#a03706",
    # purples + black: controls
    "llama2_7b_random": "#c9b6f2", "llama2_13b_random": "#9670e0",
    "llama2_70b_random": "#6b32c9", "random": "#3f1a78",
    "olmo2_7b_random": "#b06ab3",   # purple like every control, distinct from the rest
    "tfidf": "#000000",
}
LAB = {
    "llama2_70b": "Llama-2-70B", "llama2_13b": "Llama-2-13B",
    "llama2_7b": "Llama-2-7B", "gpt_oss_120b": "gpt-oss-120B",
    "olmo2_7b": "OLMo-2-7B", "olmo2_7b_random": "OLMo-2-7B rand*",
    "qwen3_32b": "Qwen3-32B", "qwen3_8b": "Qwen3-8B", "qwen3_1b7": "Qwen3-1.7B",
    "umt5_base": "uMT5-base", "thalesian_cunei400m": "cuneiform-400M",
    "thalesian_akk300m": "AKK-300M", "random": "random Qwen3-8B*",
    "llama2_7b_random": "Llama-2-7B rand*", "llama2_13b_random": "Llama-2-13B rand*",
    "llama2_70b_random": "Llama-2-70B rand*", "tfidf": "TF-IDF floor*",
}
ENC = {"thalesian_akk300m", "thalesian_cunei400m", "umt5_base"}
ORDER = ["llama2_70b", "llama2_13b", "llama2_7b", "olmo2_7b", "gpt_oss_120b", "qwen3_32b",
         "qwen3_8b", "qwen3_1b7", "umt5_base", "thalesian_cunei400m",
         "thalesian_akk300m", "tfidf", "llama2_70b_random", "llama2_13b_random",
         "llama2_7b_random", "olmo2_7b_random", "random"]
IS_CTRL = {"random", "tfidf", "llama2_7b_random", "llama2_13b_random",
           "llama2_70b_random", "olmo2_7b_random"}

#: deck type scale — plot_cellA_figs.py's rcParams, minus the dpi keys, which
#: figures/lib/_save.py owns (300 dpi + vector PDF, vs the deck script's 130).
RC = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 16.5,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13.5,
    "axes.linewidth": 0.9, "lines.linewidth": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight",
}


def rc(**over):
    """Apply the deck's type scale. Call once at the top of a figure script."""
    import matplotlib.pyplot as plt
    d = dict(RC)
    d.update(over)
    plt.rcParams.update(d)
    return d


def isr(m):
    return m.endswith("random") or m == "random"


def sty(m):
    """Line style for one arm: controls dashed and thinner, TF-IDF dotted."""
    if m == "tfidf":
        return dict(color=COL["tfidf"], ls=(0, (1.5, 1.6)), lw=2.2)
    return dict(color=COL.get(m, "#888"), ls="--" if isr(m) else "-",
                lw=1.5 if isr(m) else 2.2)


def star(ax, x, y, m):
    """Mark an arm's best point — the deck uses these to call out the peak layer."""
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return
    i = int(np.nanargmax(y))
    ax.plot(np.asarray(x)[i], y[i], marker="*", ms=17, color=COL.get(m, "#888"),
            mec="k", mew=.6, zorder=6)


def r2_axis(ax):
    """The deck's symlog R2 axis: linear within +-0.1, log outside, so the deep
    negative tail of a failing arm fits without squashing the 0-1 band."""
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.7)
    ax.set_yticks([-10, -1, 0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["-10", "-1", "0", ".25", ".5", ".75", "1"])
    ax.set_ylim(-1.8, 1.25)
