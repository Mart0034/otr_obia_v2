"""Tests for the multi-date Sentinel-2 features: shadows move with the sun
over the year, tire dumps don't. Covers the reflectance conversion, cloud
masking, the per-pixel statistics and how the pipeline attaches them."""
from datetime import datetime
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from fetch_s2_timeseries import _dedupe_by_date, temporal_stats, to_reflectance, valid_mask
from otr_obia_pipeline import CONFIG, build_dataset


def test_reflectance_accounts_for_the_2022_offset():
    dn = np.array([1500, 0], dtype=np.uint16)
    new = to_reflectance(dn, "05.10")
    old = to_reflectance(dn, "03.01")
    assert new[0] == pytest.approx(0.05)
    assert old[0] == pytest.approx(0.15)
    assert np.isnan(new[1])  # DN 0 = no data


def test_cloud_mask_drops_clouds_but_keeps_terrain_shadow():
    scl = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, np.nan], dtype=np.float32)
    assert valid_mask(scl).tolist() == [
        False, False, True, False, True, True, False, False, False, True, False,
    ]


def test_dump_stays_steady_while_shadowed_slope_swings():
    dump = [0.08, 0.08, 0.08, 0.08]      # always dark
    slope = [0.30, 0.30, 0.28, 0.05]     # only dark in winter
    stack = np.array([dump, slope], dtype=np.float32).T.reshape(4, 1, 2)

    cv, min_ratio = temporal_stats(stack)

    assert cv[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert min_ratio[0, 0] == pytest.approx(1.0)
    assert cv[0, 1] > 0.4
    assert min_ratio[0, 1] < 0.25


def test_pixels_with_too_few_valid_dates_stay_empty():
    stack = np.array([0.2, np.nan, np.nan, 0.2], dtype=np.float32).reshape(4, 1, 1)
    cv, min_ratio = temporal_stats(stack, min_valid=3)
    assert np.isnan(cv[0, 0]) and np.isnan(min_ratio[0, 0])


def test_same_day_from_two_tiles_is_kept_once():
    items = [
        SimpleNamespace(datetime=datetime(2023, 3, 1, 14, 37), id="tileA"),
        SimpleNamespace(datetime=datetime(2023, 3, 1, 14, 37), id="tileB"),
        SimpleNamespace(datetime=datetime(2023, 3, 11, 14, 37), id="tileA"),
    ]
    assert len(_dedupe_by_date(items)) == 2


def test_pipeline_adds_temporal_features_per_segment(tmp_path, write_synthetic_raster):
    imagery_dir, s2t_dir = tmp_path / "imagery", tmp_path / "s2t"
    imagery_dir.mkdir()
    s2t_dir.mkdir()
    s2 = write_synthetic_raster(imagery_dir / "mine_001.tif", seed=1)
    with rasterio.open(s2) as ref:
        profile = {"driver": "GTiff", "count": 2, "dtype": "float32", "crs": ref.crs,
                   "transform": ref.transform, "height": ref.height, "width": ref.width}
    with rasterio.open(s2t_dir / "mine_001.tif", "w", **profile) as dst:
        dst.write(np.stack([np.full((40, 40), 0.1), np.full((40, 40), 0.9)]).astype(np.float32))

    labels_path = tmp_path / "labels.gpkg"
    gpd.GeoDataFrame({"mine_id": []}, geometry=[], crs="EPSG:32632").to_file(
        labels_path, driver="GPKG"
    )
    cfg = dict(CONFIG)
    cfg.update({"imagery_dir": str(imagery_dir), "labels_path": str(labels_path),
                "n_jobs": 1, "s2t_dir": str(s2t_dir)})

    features, _ = build_dataset(cfg)

    assert features["s2t_bright_cv_mean"].to_numpy() == pytest.approx(0.1)
    assert features["s2t_bright_min_ratio_mean"].to_numpy() == pytest.approx(0.9)
