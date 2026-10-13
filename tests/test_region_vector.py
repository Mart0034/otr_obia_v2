import os
import sys

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import region_vector  # noqa: E402


def test_split_big_cuts_large_polygons_into_grid_pieces():
    small = box(0, 0, 1000, 1000)
    big = box(10000, 10000, 16000, 12000)            # 6 km x 2 km
    gdf = gpd.GeoDataFrame(geometry=[small, big], crs="EPSG:32719")
    out = region_vector.split_big(gdf, piece_m=2700, big_side_m=3500)
    areas = sorted(out.area_m2)
    assert abs(out.area_m2.sum() - (small.area + big.area)) < 1e5          # nichts Wesentliches geht verloren
    assert len(out) > 2 and max(areas) <= 2700 * 2700 + 1                    # Stücke höchstens eine Rasterkachel groß
    assert any(abs(a - small.area) < 1 for a in areas)                       # kleines Polygon bleibt ganz
