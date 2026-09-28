"""Tests for compute_shape_features(): the 'shape of the patch' features
that the docstring promised but the original code never computed."""
import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box

from otr_obia_pipeline import compute_shape_features


def test_square_area_perimeter_and_compactness():
    gdf = gpd.GeoDataFrame(
        {"segment_id": [1]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:32632"
    )
    result = compute_shape_features(gdf)
    assert result.loc[0, "shape_area"] == 100.0
    assert result.loc[0, "shape_perimeter"] == 40.0
    # A square is never as "round" as a circle -> compactness strictly below 1
    assert 0 < result.loc[0, "shape_compactness"] < 1.0


def test_larger_segment_has_larger_area():
    gdf = gpd.GeoDataFrame(
        {"segment_id": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(0, 0, 20, 20)],
        crs="EPSG:32632",
    )
    result = compute_shape_features(gdf)
    assert result.loc[1, "shape_area"] > result.loc[0, "shape_area"]


def test_rectangle_fills_its_own_bounding_rectangle():
    # a building-like segment: axis-aligned rectangle, area == its own
    # minimum rotated rectangle's area -> rectangularity exactly 1.0
    gdf = gpd.GeoDataFrame(
        {"segment_id": [1]}, geometry=[box(0, 0, 10, 6)], crs="EPSG:32632"
    )
    result = compute_shape_features(gdf)
    assert result.loc[0, "shape_rectangularity"] == pytest.approx(1.0)


def test_ragged_shape_has_low_rectangularity():
    # a jagged, tire-pile-like outline: much smaller area than the
    # rectangle needed to enclose it -> low rectangularity
    ragged = Polygon([(0, 0), (2, 3), (0, 5), (3, 6), (1, 9), (5, 8), (5, 2), (2, 1)])
    gdf = gpd.GeoDataFrame({"segment_id": [1]}, geometry=[ragged], crs="EPSG:32632")
    result = compute_shape_features(gdf)
    assert result.loc[0, "shape_rectangularity"] < 0.6


def test_circle_has_the_expected_rectangularity():
    # a circle's minimum bounding rectangle is a square of side 2r ->
    # rectangularity = pi/4 regardless of radius
    circle = Point(0, 0).buffer(10, quad_segs=64)
    gdf = gpd.GeoDataFrame({"segment_id": [1]}, geometry=[circle], crs="EPSG:32632")
    result = compute_shape_features(gdf)
    assert result.loc[0, "shape_rectangularity"] == pytest.approx(3.14159265 / 4, abs=1e-3)
