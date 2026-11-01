"""Loss library public surface (SLA section 5): torch terms from core,
the numpy calibrator, and the weak-label pair generator."""
from chrono.losses.calibrate import MonotoneCalibrator
from chrono.losses.core import (bt_loss, cka_loss, graph_smoothness, hsic_loss,
                                interval_nll, soft_spearman,
                                softrank_loss, variance_loss)
from chrono.losses.pairs import make_order_pairs

__all__ = ["bt_loss", "softrank_loss", "soft_spearman", "variance_loss",
           "hsic_loss", "cka_loss", "graph_smoothness", "interval_nll",
           "MonotoneCalibrator", "make_order_pairs"]
