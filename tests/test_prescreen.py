import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import prescreen  # noqa: E402


def test_class_fractions_and_keep_rules():
    arr = np.array([[60] * 8 + [50, 50]] * 10)                       # 80 % kahl, 20 % bebaut
    frac = prescreen.class_fractions(arr)
    assert abs(frac["bare"] - 0.8) < 1e-9 and abs(frac["built"] - 0.2) < 1e-9
    assert prescreen.keep_tile(frac) == (True, "ok")
    assert prescreen.keep_tile({"built": 0.4, "bare": 0.6})[0] is False
    assert prescreen.keep_tile({"water": 0.7, "bare": 0.3})[0] is False
    assert prescreen.keep_tile({"tree": 0.3, "crop": 0.3, "bare": 0.4})[0] is False     # grün zusammen > 50 %
    assert prescreen.keep_tile({})[0] is True                                           # ohne Daten: behalten
    assert prescreen.keep_tile({"built": 0.4}, {"max_built": 0.5})[0] is True


def test_grid_tiles_cover_region_without_gaps():
    rows = prescreen.grid_tiles(-69.6, -23.6, -69.4, -23.4, 2700, prefix="g")
    assert rows and all(r["status"] == "ok" and r["radius_m"] == 1350 for r in rows)
    assert len({r["site_id"] for r in rows}) == len(rows)
    xs = sorted({round(r["center_x"]) for r in rows})
    assert all(abs((b - a) - 2700) < 1 for a, b in zip(xs, xs[1:]))
