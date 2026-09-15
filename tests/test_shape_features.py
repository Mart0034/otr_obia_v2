"""Tests for compute_shape_features(): the 'shape of the patch' features
that the docstring promised but the original code never computed."""
import geopandas as gpd
from shapely.geometry import box

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
