"""Model surface (SLA section 6): embedding store, adapter head, EMA
twin. The trainer lives in chrono/scripts/train_cjb.py."""
from chrono.models.heads import AdapterHead, EmaTwin
from chrono.models.store import EmbStore

__all__ = ["EmbStore", "AdapterHead", "EmaTwin"]
