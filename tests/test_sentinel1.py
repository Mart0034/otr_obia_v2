"""Tests for the optional Sentinel-1 radar features: scene selection and
median/dB combination in fetch_sentinel1.py, and how the pipeline attaches
the radar bands to each segment (or leaves them empty when a file is missing)."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from fetch_sentinel1 import combine_scenes, select_evenly
from otr_obia_pipeline import CONFIG, build_dataset, load_s1_for_mine


def _write_s1_like(s2_path, s1_path, vv_db=-12.0, vh_db=-20.0, shape=None):
    with rasterio.open(s2_path) as ref:
        profile = {
            "driver": "GTiff", "count": 2, "dtype": "float32", "crs": ref.crs,
            "transform": ref.transform,
            "height": shape[0] if shape else ref.height,
            "width": shape[1] if shape else ref.width,
        }
    h, w = profile["height"], profile["width"]
    with rasterio.open(s1_path, "w", **profile) as dst:
        dst.write(np.stack([np.full((h, w), vv_db), np.full((h, w), vh_db)]).astype(np.float32))


def _cfg(tmp_path, imagery_dir, s1_dir):
    labels_path = tmp_path / "labels.gpkg"
    gpd.GeoDataFrame({"mine_id": []}, geometry=[], crs="EPSG:32632").to_file(
        labels_path, driver="GPKG"
    )
    cfg = dict(CONFIG)
    cfg.update({
        "imagery_dir": str(imagery_dir),
        "labels_path": str(labels_path),
        "n_jobs": 1,
        "s1_dir": str(s1_dir),
    })
    return cfg


def test_select_evenly_spreads_scenes_over_the_period():
    start = datetime(2023, 1, 1)
    items = [SimpleNamespace(datetime=start + timedelta(days=12 * i)) for i in range(30)]

    chosen = select_evenly(items, 4)

    assert len(chosen) == 4
    assert chosen[0].datetime == items[0].datetime
    assert chosen[-1].datetime == items[-1].datetime


def test_select_evenly_keeps_everything_when_few_scenes():
    items = [SimpleNamespace(datetime=datetime(2023, 1, d)) for d in (3, 1, 2)]
    assert [it.datetime.day for it in select_evenly(items, 12)] == [1, 2, 3]


def test_combine_scenes_takes_median_in_db_and_ignores_empty_pixels():
    a = np.array([[0.1, 0.0]], dtype=np.float32)
    b = np.array([[0.1, 0.0]], dtype=np.float32)
    c = np.array([[1.0, np.nan]], dtype=np.float32)

    out = combine_scenes([a, b, c])

    assert out[0, 0] == pytest.approx(-10.0)  # median 0.1 -> -10 dB
    assert np.isnan(out[0, 1])  # no valid scene at all -> stays empty


def test_pipeline_adds_radar_features_per_segment(tmp_path, write_synthetic_raster):
    imagery_dir, s1_dir = tmp_path / "imagery", tmp_path / "s1"
    imagery_dir.mkdir()
    s1_dir.mkdir()
    s2 = write_synthetic_raster(imagery_dir / "mine_001.tif", seed=1)
    _write_s1_like(s2, s1_dir / "mine_001.tif", vv_db=-12.0, vh_db=-20.0)

    features, _ = build_dataset(_cfg(tmp_path, imagery_dir, s1_dir))

    assert features["s1_vv_mean"].to_numpy() == pytest.approx(-12.0)
    assert features["s1_vh_vv_mean"].to_numpy() == pytest.approx(-8.0)
    assert "s1_vh_std" in features.columns


def test_missing_radar_file_gives_empty_columns_not_a_crash(tmp_path, write_synthetic_raster):
    imagery_dir, s1_dir = tmp_path / "imagery", tmp_path / "s1"
    imagery_dir.mkdir()
    s1_dir.mkdir()
    write_synthetic_raster(imagery_dir / "mine_001.tif", seed=1)

    with pytest.warns(RuntimeWarning):  # nanmean over all-NaN radar bands
        features, _ = build_dataset(_cfg(tmp_path, imagery_dir, s1_dir))

    assert "s1_vv_mean" in features.columns
    assert features["s1_vv_mean"].isna().all()


def test_radar_file_on_a_different_grid_is_rejected(tmp_path, write_synthetic_raster):
    s2 = write_synthetic_raster(tmp_path / "mine_001.tif", seed=1, size=40)
    s1_dir = tmp_path / "s1"
    s1_dir.mkdir()
    _write_s1_like(s2, s1_dir / "mine_001.tif", shape=(30, 30))

    with pytest.raises(ValueError, match="Raster"):
        load_s1_for_mine(s2, str(s1_dir), (40, 40))
