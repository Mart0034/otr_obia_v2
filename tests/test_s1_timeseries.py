"""Tests for the multi-date Sentinel-1 features: a geometrically clear
surface (road, building edge) scatters radar differently depending on
look angle, so its backscatter swings between acquisitions; a disordered
tire pile scatters diffusely and stays comparatively constant. Covers the
per-pixel temporal statistic and how the pipeline attaches it."""
import geopandas as gpd
import numpy as np
import pytest
import rasterio

from fetch_sentinel1_timeseries import temporal_cv
from otr_obia_pipeline import CONFIG, build_dataset


def test_dump_stays_steady_while_oriented_surface_swings():
    dump = [10.0, 10.0, 10.0, 10.0, 10.0]     # diffuse scatter, look-angle independent
    road = [4.0, 12.0, 3.0, 11.0, 4.0]        # oriented surface, swings with look angle
    stack = np.array([dump, road], dtype=np.float32).T.reshape(5, 1, 2)

    cv = temporal_cv(stack, min_valid=4)

    assert cv[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert cv[0, 1] > 0.4


def test_pixels_with_too_few_valid_scenes_stay_empty():
    stack = np.array([5.0, np.nan, np.nan, 5.0], dtype=np.float32).reshape(4, 1, 1)
    cv = temporal_cv(stack, min_valid=3)
    assert np.isnan(cv[0, 0])


def test_non_positive_values_are_treated_as_invalid():
    # 0.0 and -1.0 are not physically valid linear backscatter and must be
    # excluded from the count of valid scenes, same as NaN.
    stack = np.array([5.0, 0.0, -1.0, 6.0], dtype=np.float32).reshape(4, 1, 1)
    cv = temporal_cv(stack, min_valid=3)
    assert np.isnan(cv[0, 0])  # only 2 positive values -> not enough

    stack2 = np.array([5.0, 0.0, -1.0, 6.0, 5.5], dtype=np.float32).reshape(5, 1, 1)
    cv2 = temporal_cv(stack2, min_valid=3)
    assert not np.isnan(cv2[0, 0])  # 3 positive values -> enough


def test_pipeline_adds_s1_temporal_features_per_segment(tmp_path, write_synthetic_raster):
    imagery_dir, s1t_dir = tmp_path / "imagery", tmp_path / "s1t"
    imagery_dir.mkdir()
    s1t_dir.mkdir()
    s2 = write_synthetic_raster(imagery_dir / "mine_001.tif", seed=1)
    with rasterio.open(s2) as ref:
        profile = {"driver": "GTiff", "count": 2, "dtype": "float32", "crs": ref.crs,
                   "transform": ref.transform, "height": ref.height, "width": ref.width}
    with rasterio.open(s1t_dir / "mine_001.tif", "w", **profile) as dst:
        dst.write(np.stack([np.full((40, 40), 0.2), np.full((40, 40), 0.7)]).astype(np.float32))

    labels_path = tmp_path / "labels.gpkg"
    gpd.GeoDataFrame({"mine_id": []}, geometry=[], crs="EPSG:32632").to_file(
        labels_path, driver="GPKG"
    )
    cfg = dict(CONFIG)
    cfg.update({"imagery_dir": str(imagery_dir), "labels_path": str(labels_path),
                "n_jobs": 1, "s1t_dir": str(s1t_dir)})

    features, _ = build_dataset(cfg)

    assert features["s1t_vv_cv_mean"].to_numpy() == pytest.approx(0.2)
    assert features["s1t_vh_cv_mean"].to_numpy() == pytest.approx(0.7)
