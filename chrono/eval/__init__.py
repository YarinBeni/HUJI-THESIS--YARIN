"""chrono.eval — the evaluation protocol (SLA §7), one import surface.

Everything is computed against the corpus t column (astronomical years,
larger = later) on frozen SLA §3 splits; years are never re-derived
here. Consumers write `from chrono.eval import mc_balanced_rho, ...`.
"""
from chrono.eval.calibration import coverage, mean_width, winkler_score
from chrono.eval.protocol import gkf_rho, mc_balanced_rho, placebo_rho
from chrono.eval.robustness import BATTERY_COLS, battery

__all__ = ["mc_balanced_rho", "gkf_rho", "placebo_rho", "battery",
           "BATTERY_COLS", "coverage", "mean_width", "winkler_score"]
