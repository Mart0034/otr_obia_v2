"""Tests for compute_is_building(): a purely informational is_building
column (never touches dump_pred/dump_proba) so a segment overlapping or
touching a mapped OSM building can be filtered out by hand in QGIS.
Validated against the real 36 labeled dumps (see README): never affects a
real dump, catches ~1.6% of false positives on average - much more at
well-OSM-mapped sites, close to nothing at sparsely-mapped ones."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from otr_obia_pipeline import compute_is_building


def _result(rows, geoms):
    df = pd.DataFrame(rows, columns=["mine_id"])
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:32719")


def _write_buildings(path, geoms, crs="EPSG:32719"):
    gpd.GeoDataFrame(geometry=geoms, crs=crs).to_file(path, driver="GPKG")


def test_segment_overlapping_a_building_is_flagged(tmp_path):
    result = _result([("mine_1",)], [box(0, 0, 10, 10)])
    _write_buildings(tmp_path / "mine_1.gpkg", [box(2, 2, 4, 4)])

    out = compute_is_building(result, str(tmp_path))

    assert out["is_building"].tolist() == [True]


def test_segment_touching_a_building_segment_is_also_flagged(tmp_path):
    # segment A overlaps the building, segment B only touches segment A
    result = _result(
        [("mine_1",), ("mine_1",)],
        [box(0, 0, 10, 10), box(10, 0, 20, 10)],
    )
    _write_buildings(tmp_path / "mine_1.gpkg", [box(2, 2, 4, 4)])

    out = compute_is_building(result, str(tmp_path))

    assert out["is_building"].tolist() == [True, True]


def test_far_segment_is_not_flagged(tmp_path):
    result = _result(
        [("mine_1",), ("mine_1",)],
        [box(0, 0, 10, 10), box(1000, 1000, 1010, 1010)],
    )
    _write_buildings(tmp_path / "mine_1.gpkg", [box(2, 2, 4, 4)])

    out = compute_is_building(result, str(tmp_path))

    assert out["is_building"].tolist() == [True, False]


def test_no_osm_buildings_dir_flags_nothing_and_does_not_crash():
    result = _result([("mine_1",)], [box(0, 0, 10, 10)])

    out = compute_is_building(result, None)

    assert out["is_building"].tolist() == [False]


def test_mine_with_no_matching_file_is_not_flagged(tmp_path):
    result = _result([("mine_1",), ("mine_2",)], [box(0, 0, 10, 10), box(0, 0, 10, 10)])
    _write_buildings(tmp_path / "mine_1.gpkg", [box(2, 2, 4, 4)])
    # no mine_2.gpkg written at all

    out = compute_is_building(result, str(tmp_path))

    assert out["is_building"].tolist() == [True, False]


def test_empty_buildings_file_flags_nothing(tmp_path):
    result = _result([("mine_1",)], [box(0, 0, 10, 10)])
    _write_buildings(tmp_path / "mine_1.gpkg", [])

    out = compute_is_building(result, str(tmp_path))

    assert out["is_building"].tolist() == [False]


def test_never_modifies_dump_pred_or_proba(tmp_path):
    df = pd.DataFrame([{"mine_id": "mine_1", "dump_pred": 1, "dump_proba": 0.9}])
    result = gpd.GeoDataFrame(df, geometry=[box(0, 0, 10, 10)], crs="EPSG:32719")
    _write_buildings(tmp_path / "mine_1.gpkg", [box(2, 2, 4, 4)])

    out = compute_is_building(result, str(tmp_path))

    assert out["dump_pred"].tolist() == [1]
    assert out["dump_proba"].tolist() == [0.9]
    assert out["is_building"].tolist() == [True]
