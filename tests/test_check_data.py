"""Tests for check_data.py: catching data problems before the full
pipeline runs, using small made-up mines instead of real imagery."""
import geopandas as gpd
from shapely.geometry import box

from check_data import check_data
from otr_obia_pipeline import CONFIG


def _base_cfg(imagery_dir, labels_path):
    cfg = dict(CONFIG)
    cfg.update({
        "imagery_dir": str(imagery_dir),
        "labels_path": str(labels_path),
        "mine_id_field": "mine_id",
    })
    return cfg


def test_passes_when_mine_ids_match(tmp_path, write_synthetic_raster):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()
    write_synthetic_raster(imagery_dir / "1.tif", seed=1)

    labels_gdf = gpd.GeoDataFrame(
        {"mine_id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32632"
    )
    labels_path = tmp_path / "labels.gpkg"
    labels_gdf.to_file(labels_path, driver="GPKG")

    assert check_data(_base_cfg(imagery_dir, labels_path)) is True


def test_fails_when_mine_ids_dont_match(tmp_path, write_synthetic_raster):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()
    write_synthetic_raster(imagery_dir / "1.tif", seed=1)

    labels_gdf = gpd.GeoDataFrame(
        {"mine_id": [999]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32632"
    )
    labels_path = tmp_path / "labels.gpkg"
    labels_gdf.to_file(labels_path, driver="GPKG")

    assert check_data(_base_cfg(imagery_dir, labels_path)) is False


def test_fails_when_labels_file_missing(tmp_path, write_synthetic_raster):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()
    write_synthetic_raster(imagery_dir / "1.tif", seed=1)

    assert check_data(_base_cfg(imagery_dir, tmp_path / "missing.gpkg")) is False


def test_fails_when_no_imagery_found(tmp_path):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()

    labels_gdf = gpd.GeoDataFrame(
        {"mine_id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32632"
    )
    labels_path = tmp_path / "labels.gpkg"
    labels_gdf.to_file(labels_path, driver="GPKG")

    assert check_data(_base_cfg(imagery_dir, labels_path)) is False
