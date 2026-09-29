"""Tests for compute_road_density(): purely informational road_density_150m
and is_road_grid columns (never touch dump_pred/dump_proba). Idea: bare
proximity to any road barely separates real dumps from false positives
(11% of the 36 real dumps sit within 12m of a road - a single access
track is normal), but a dense GRID of roads is the signature of a
facility's parking/vehicle area, not a single dump. Validated against the
real data (see README): grid_threshold_m=500 affects 0/36 real dumps
while catching ~6.5% of borderline false positives."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

from otr_obia_pipeline import compute_road_density


def _result(rows, geoms):
    df = pd.DataFrame(rows, columns=["mine_id"])
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:32719")


def _write_roads(path, lines, crs="EPSG:32719"):
    gpd.GeoDataFrame(geometry=lines, crs=crs).to_file(path, driver="GPKG")


def test_dense_grid_is_flagged(tmp_path):
    # a segment at the origin, surrounded by a grid of roads well past
    # the 500m default threshold
    result = _result([("mine_1",)], [box(-5, -5, 5, 5)])
    grid = [LineString([(-100, y), (100, y)]) for y in range(-100, 101, 20)]
    grid += [LineString([(x, -100), (x, 100)]) for x in range(-100, 101, 20)]
    _write_roads(tmp_path / "mine_1.gpkg", grid)

    out = compute_road_density(result, str(tmp_path))

    assert out["road_density_150m"].iloc[0] > 500
    assert out["is_road_grid"].iloc[0] == True  # noqa: E712


def test_single_nearby_road_is_not_flagged(tmp_path):
    # a single road passing near the segment - not a grid
    result = _result([("mine_1",)], [box(-5, -5, 5, 5)])
    _write_roads(tmp_path / "mine_1.gpkg", [LineString([(-200, 0), (200, 0)])])

    out = compute_road_density(result, str(tmp_path))

    assert out["is_road_grid"].iloc[0] == False  # noqa: E712
    assert 0 < out["road_density_150m"].iloc[0] < 500


def test_no_nearby_roads_gives_zero_density(tmp_path):
    result = _result([("mine_1",)], [box(-5, -5, 5, 5)])
    _write_roads(tmp_path / "mine_1.gpkg", [LineString([(10000, 10000), (10001, 10001)])])

    out = compute_road_density(result, str(tmp_path))

    assert out["road_density_150m"].iloc[0] == 0.0
    assert out["is_road_grid"].iloc[0] == False  # noqa: E712


def test_no_osm_roads_dir_gives_zero_and_does_not_crash():
    result = _result([("mine_1",)], [box(-5, -5, 5, 5)])

    out = compute_road_density(result, None)

    assert out["road_density_150m"].tolist() == [0.0]
    assert out["is_road_grid"].tolist() == [False]


def test_mine_with_no_matching_file_gets_zero(tmp_path):
    result = _result([("mine_1",), ("mine_2",)], [box(-5, -5, 5, 5), box(-5, -5, 5, 5)])
    _write_roads(tmp_path / "mine_1.gpkg", [LineString([(-200, 0), (200, 0)])])

    out = compute_road_density(result, str(tmp_path))

    assert out[out.mine_id == "mine_2"]["road_density_150m"].tolist() == [0.0]


def test_custom_threshold_is_respected(tmp_path):
    result = _result([("mine_1",)], [box(-5, -5, 5, 5)])
    _write_roads(tmp_path / "mine_1.gpkg", [LineString([(-200, 0), (200, 0)])])

    out = compute_road_density(result, str(tmp_path), grid_threshold_m=50)

    assert out["is_road_grid"].iloc[0] == True  # noqa: E712


def test_never_modifies_dump_pred_or_proba(tmp_path):
    df = pd.DataFrame([{"mine_id": "mine_1", "dump_pred": 1, "dump_proba": 0.9}])
    result = gpd.GeoDataFrame(df, geometry=[box(-5, -5, 5, 5)], crs="EPSG:32719")
    _write_roads(tmp_path / "mine_1.gpkg", [LineString([(-200, 0), (200, 0)])])

    out = compute_road_density(result, str(tmp_path))

    assert out["dump_pred"].tolist() == [1]
    assert out["dump_proba"].tolist() == [0.9]
