"""Tests for filter_to_mine_boundary(): each mine's Sentinel-2 export
includes a 500m buffer of surrounding terrain that can never contain a
tire dump. Checked against the real dataset: 16.2% of all false positives
fell outside the real mine boundary, entirely in that buffer zone."""
import geopandas as gpd
import pytest
from shapely.geometry import box

from otr_obia_pipeline import CONFIG, build_dataset, filter_to_mine_boundary


def _segments():
    return gpd.GeoDataFrame(
        {"segment_id": [1, 2, 3]},
        geometry=[
            box(0, 0, 10, 10),        # fully inside the mine boundary
            box(100, 100, 110, 110),  # far outside -> should be dropped
            box(9, 9, 15, 15),        # partially overlapping -> kept
        ],
        crs="EPSG:32719",
    )


def test_segments_outside_boundary_are_dropped():
    boundary = gpd.GeoDataFrame(
        {"mine_id": ["mine_001"]}, geometry=[box(0, 0, 12, 12)], crs="EPSG:32719"
    )
    boundary["_mine_id_str"] = boundary["mine_id"]

    result = filter_to_mine_boundary(_segments(), boundary, "mine_001")

    assert set(result["segment_id"]) == {1, 3}


def test_buffer_extends_the_boundary():
    boundary = gpd.GeoDataFrame(
        {"mine_id": ["mine_001"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:32719"
    )
    boundary["_mine_id_str"] = boundary["mine_id"]

    # segment 2 is centered at (105,105), box(100,100,110,110) - still too
    # far even with a modest buffer, but confirms buffer_m is applied
    # without crashing and doesn't accidentally include something at 0.
    result_no_buffer = filter_to_mine_boundary(_segments(), boundary, "mine_001", buffer_m=0)
    result_big_buffer = filter_to_mine_boundary(_segments(), boundary, "mine_001", buffer_m=200)

    assert set(result_no_buffer["segment_id"]) == {1, 3}
    assert set(result_big_buffer["segment_id"]) == {1, 2, 3}


def test_no_boundary_for_this_mine_is_a_no_op():
    boundary = gpd.GeoDataFrame(
        {"mine_id": ["mine_999"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32719"
    )
    boundary["_mine_id_str"] = boundary["mine_id"]

    result = filter_to_mine_boundary(_segments(), boundary, "mine_001")

    assert set(result["segment_id"]) == {1, 2, 3}


def test_none_boundary_gdf_is_a_no_op():
    result = filter_to_mine_boundary(_segments(), None, "mine_001")
    assert set(result["segment_id"]) == {1, 2, 3}


def test_build_dataset_raises_clear_error_for_missing_boundary_file(tmp_path):
    labels_gdf = gpd.GeoDataFrame({"mine_id": []}, geometry=[], crs="EPSG:32719")
    labels_path = tmp_path / "labels.gpkg"
    labels_gdf.to_file(labels_path, driver="GPKG")

    cfg = dict(CONFIG)
    cfg.update({
        "imagery_dir": str(tmp_path / "imagery"),  # doesn't need to exist yet
        "labels_path": str(labels_path),
        "mine_id_field": "mine_id",
        "mine_boundary_path": str(tmp_path / "does_not_exist.gpkg"),
    })

    with pytest.raises(FileNotFoundError, match="mine_boundary_path"):
        build_dataset(cfg)
