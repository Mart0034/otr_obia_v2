"""Tests for site_context: inside a mine / near one / open desert."""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from candidate_sites import sites_from_segments, write_sites_gpkg
from site_context import add_context, load_operations, main

X0, Y0 = 400000.0, 7400000.0


def _sites():
    boxes = [box(X0 + 10, Y0 + 10, X0 + 30, Y0 + 30),        # inside the operation
             box(X0 + 300, Y0 + 10, X0 + 320, Y0 + 30),       # ~200 m east of its edge -> near
             box(X0 + 5000, Y0, X0 + 5020, Y0 + 20)]          # far away -> desert
    return gpd.GeoDataFrame({"site_id": ["a", "b", "c"], "review": ["", "clean", ""], "note": ["", "x", ""]},
                            geometry=boxes, crs="EPSG:32719")


def _ops():
    return gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + 100, Y0 + 100)], crs="EPSG:32719")


def test_three_classes_and_distance():
    out = add_context(_sites(), _ops(), near_m=250)
    assert out["context"].tolist() == ["inside_op", "near_op", "open_desert"]
    assert out.loc[0, "op_dist_m"] == 0 and out.loc[1, "op_dist_m"] == pytest.approx(200, abs=1)


def test_near_threshold_is_respected():
    assert add_context(_sites(), _ops(), near_m=100)["context"].tolist()[1] == "open_desert"


def test_no_operations_means_open_desert_and_empty_sites_are_fine():
    assert add_context(_sites(), None)["context"].eq("open_desert").all()
    assert add_context(_sites().iloc[0:0], _ops()).empty


def test_point_operations_become_small_areas_and_other_crs_is_converted(tmp_path):
    pts = gpd.GeoDataFrame(geometry=[Point(X0 + 20, Y0 + 20)], crs="EPSG:32719").to_crs(4326)
    path = str(tmp_path / "nodes.geojson")
    pts.to_file(path, driver="GeoJSON")
    ops = load_operations([path], "EPSG:32719")
    assert ops.geometry.iloc[0].geom_type == "Polygon" and ops.geometry.iloc[0].area > 1e4
    assert add_context(_sites(), ops)["context"].tolist()[0] == "inside_op"


def test_cli_keeps_reviews_and_the_drawing_layer(tmp_path):
    src, ops_path, out = str(tmp_path / "s.gpkg"), str(tmp_path / "o.gpkg"), str(tmp_path / "out.gpkg")
    write_sites_gpkg(_sites(), src)
    _ops().to_file(ops_path, driver="GPKG")
    assert main(["--sites", src, "--out", out, "--operations", ops_path]) == 0
    res = gpd.read_file(out, layer="candidate_sites")
    assert res["review"].tolist() == ["", "clean", ""] and res["context"].tolist()[0] == "inside_op"
    assert len(gpd.read_file(out, layer="drawn_polygons")) == 0


def test_sites_from_segments_adds_context(tmp_path):
    cells = [(c, 0, 0.9) for c in range(5)]
    seg = gpd.GeoDataFrame({"mine_id": "m1", "dump_proba": [c[2] for c in cells]},
                           geometry=[box(X0 + c * 10, Y0, X0 + c * 10 + 10, Y0 + 10) for c, _, _ in cells],
                           crs="EPSG:32719")
    ops_path = str(tmp_path / "o.gpkg")
    _ops().to_file(ops_path, driver="GPKG")
    sites = sites_from_segments(seg, str(tmp_path / "l.gpkg"), threshold=0.5, operations=ops_path)
    assert sites["context"].tolist() == ["inside_op"]
    assert "context" in gpd.read_file(str(tmp_path / "l.gpkg"), layer="candidate_sites").columns
