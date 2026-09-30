"""Tests for compute_is_poi() and the POI parsing: a purely informational
is_poi column (never touches dump_pred/dump_proba) for segments near an OSM
point (amenity/shop/tourism) or inside a retail/commercial area - places
where OSM has a business but no building outline. Validated on the real data:
0 of 36 real dumps affected, ~0.3% of borderline false positives caught."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from fetch_osm_poi import _element_to_geometry
from otr_obia_pipeline import compute_is_poi


def _result(mine_ids, geoms):
    return gpd.GeoDataFrame(pd.DataFrame({"mine_id": mine_ids}), geometry=geoms, crs="EPSG:32719")


def _write_poi(path, geoms):
    gpd.GeoDataFrame({"kind": ["amenity/shop"] * len(geoms)}, geometry=geoms,
                     crs="EPSG:32719").to_file(path, driver="GPKG")


def test_segment_near_a_poi_point_is_flagged(tmp_path):
    result = _result(["m"], [box(0, 0, 10, 10)])
    _write_poi(tmp_path / "m.gpkg", [Point(25, 5)])  # 15 m from the segment edge
    assert compute_is_poi(result, str(tmp_path), buffer_m=20)["is_poi"].tolist() == [True]


def test_segment_far_from_any_poi_is_not_flagged(tmp_path):
    result = _result(["m"], [box(0, 0, 10, 10)])
    _write_poi(tmp_path / "m.gpkg", [Point(200, 5)])
    assert compute_is_poi(result, str(tmp_path), buffer_m=20)["is_poi"].tolist() == [False]


def test_segment_inside_a_commercial_area_is_flagged(tmp_path):
    result = _result(["m"], [box(0, 0, 10, 10)])
    _write_poi(tmp_path / "m.gpkg", [box(-50, -50, 50, 50)])
    assert compute_is_poi(result, str(tmp_path))["is_poi"].tolist() == [True]


def test_works_with_a_non_zero_based_index(tmp_path):
    result = _result(["m"] * 3, [box(i * 100, 0, i * 100 + 10, 10) for i in range(3)])
    result.index = [500, 501, 502]
    _write_poi(tmp_path / "m.gpkg", [Point(205, 5)])
    out = compute_is_poi(result, str(tmp_path))
    assert out["is_poi"].tolist() == [False, False, True]


def test_none_dir_and_missing_file_leave_everything_false(tmp_path):
    result = _result(["m", "other"], [box(0, 0, 10, 10), box(0, 0, 10, 10)])
    _write_poi(tmp_path / "m.gpkg", [Point(5, 5)])
    assert not compute_is_poi(result, None)["is_poi"].any()
    out = compute_is_poi(result, str(tmp_path))
    assert out["is_poi"].tolist() == [True, False]


def test_never_modifies_the_prediction_columns(tmp_path):
    result = _result(["m"], [box(0, 0, 10, 10)])
    result["dump_pred"] = 1
    result["dump_proba"] = 0.9
    _write_poi(tmp_path / "m.gpkg", [Point(5, 5)])
    out = compute_is_poi(result, str(tmp_path))
    assert out["dump_pred"].tolist() == [1] and out["dump_proba"].tolist() == [0.9]


def test_parsing_node_way_and_open_way():
    node = {"type": "node", "lat": 1.0, "lon": 2.0, "tags": {"amenity": "restaurant"}}
    geom, kind = _element_to_geometry(node)
    assert geom.equals(Point(2.0, 1.0)) and kind == "amenity/shop"
    ring = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    way = {"type": "way", "tags": {"landuse": "retail"},
           "geometry": [{"lat": y, "lon": x} for x, y in ring]}
    geom, kind = _element_to_geometry(way)
    assert geom.area == 1.0 and kind == "landuse"
    open_way = dict(way, geometry=way["geometry"][:-1])
    assert _element_to_geometry(open_way) is None
