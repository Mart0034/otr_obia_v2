import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import osm_places as op  # noqa: E402


def test_classify_priority_and_small_industrial_dropped():
    polys = gpd.GeoDataFrame(
        {"landuse": ["industrial", "industrial", "landfill", None], "man_made": [None, None, None, "works"],
         "amenity": [None, None, None, None], "shop": [None] * 4},
        geometry=[box(0, 0, 200, 200), box(1000, 0, 1010, 10), box(2000, 0, 2100, 100), box(3000, 0, 3300, 300)], crs=32719)
    pts = gpd.GeoDataFrame({"shop": ["tyres"], "amenity": [None], "man_made": [None]}, geometry=[Point(5000, 5000)], crs=32719)
    out = op.build(polys, pts, point_buffer_m=100, min_area_m2=5000)
    assert list(out["kind"]) == ["waste", "tyres", "works", "industrial"]        # kleine Industriefläche fällt weg, Reihenfolge nach Art
    assert out.loc[1, "area_m2"] > 30000                                          # Punkt wurde zur Fläche
