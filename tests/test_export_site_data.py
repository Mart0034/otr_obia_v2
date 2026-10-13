import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import export_site_data as ex  # noqa: E402


def test_snap_is_cell_centre_and_coarse():
    lon, lat = ex.snap(-69.5731, -23.4519)
    assert (lon, lat) == (-69.575, -23.455)                      # Mitte der 0,01°-Zelle
    assert ex.snap(-69.5799, -23.4599) == (lon, lat)             # gleiche Zelle
    assert ex.snap(-69.5799, -23.4601) == (lon, round(lat - 0.01, 5))   # Nachbarzelle in Breite
    assert abs(lon - (-69.5731)) <= 0.005 + 1e-9 and abs(lat - (-23.4519)) <= 0.005 + 1e-9


def test_cells_merge_points_and_never_keep_exact_coordinates():
    df = pd.DataFrame({"lon": [-69.5731, -69.5702, -69.4010], "lat": [-23.4519, -23.4501, -23.2001],
                       "size": ["s", "s", "m"]})
    fc = ex.cells(df, ["size"])
    assert sorted(f["properties"]["n"] for f in fc["features"]) == [1, 2]
    for f in fc["features"]:
        for c in f["geometry"]["coordinates"]:
            assert round(c * 1000) / 1000 == c or len(str(abs(c)).split(".")[1]) <= 3   # höchstens 3 Nachkommastellen
    assert {f["properties"]["region"] for f in fc["features"]} == {"atacama"}


def test_how_found_and_stats():
    assert ex.how_found({"ai_dump_p": 0.85}) == "ai" and ex.how_found({"ai_dump_p": 0.2}) == "human"
    assert ex.how_found({"ai_dump_p": None}) == "human"
    d = pd.DataFrame({"round": ["v4"] * 4, "review": ["dump", "clean", "clean", "partial"],
                      "ai_dump_p": [0.9, 0.05, 0.8, 0.1], "ai_decision": ["accepted", "spot_ok", "spot_miss", "auto_clean"]})
    st = ex.stats_from(d)
    assert st["reviewed"] == 4 and st["confirmed"] == 2
    assert st["ai_high"]["n"] == 2 and st["ai_high"]["confirmed"] == 1
    assert st["spot_checks"] == 2 and st["spot_misses"] == 0            # spot_miss zählt nur, wenn heute dump/partial
