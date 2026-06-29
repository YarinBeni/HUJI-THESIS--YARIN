"""Unambiguous loader for v_1/src/geodesic/utils.py.

There are two `utils.py` on the project (linear_probing/ and geodesic/). Putting
both dirs on sys.path makes `import utils` resolve to whichever is first —
linear_probing/utils.py imports torch and lacks find_acts_dir, so a plain
`from utils import find_acts_dir` breaks. Load geodesic/utils.py by explicit file
path to avoid the collision (it needs only numpy/scipy/sklearn, no torch).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_GEO = Path(__file__).resolve().parents[2] / "geodesic" / "utils.py"
_spec = importlib.util.spec_from_file_location("geodesic_utils", _GEO)
geodesic_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geodesic_utils)

find_acts_dir = geodesic_utils.find_acts_dir
load_layer = geodesic_utils.load_layer
available_layers = geodesic_utils.available_layers
isomap_1d = geodesic_utils.isomap_1d
