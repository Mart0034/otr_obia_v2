"""Tests for the optional Copernicus DEM terrain features: slope, position
relative to the surroundings (TPI) and north/south facing, plus how the
pipeline attaches them to each segment."""
import geopandas as gpd
import numpy as np
import pytest
import rasterio

from fetch_dem import mosaic_tiles
from otr_obia_pipeline import CONFIG, build_dataset, load_dem_for_mine, terrain_features

PX = 10.0


def _plane(rows=40, cols=40, rise_per_row=0.0, rise_per_col=0.0):
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float32)
    return 1000 + rise_per_row * r + rise_per_col * c


def test_flat_ground_has_no_slope_no_tpi_and_no_facing():
    out = terrain_features(_plane(), PX, PX, 31)
    assert out[..., 0] == pytest.approx(0.0)  # slope
    assert out[..., 1] == pytest.approx(0.0)  # tpi
    assert out[..., 2] == pytest.approx(0.0)  # northness


def test_ground_falling_to_the_north_faces_north():
    # rows run south -> elevation rising with the row means the ground falls
    # toward the north: a 10% grade (1 m per 10 m pixel)
    out = terrain_features(_plane(rise_per_row=1.0), PX, PX, 31)
    assert out[..., 0] == pytest.approx(np.degrees(np.arctan(0.1)), abs=1e-4)
    assert out[..., 2] == pytest.approx(np.sin(np.arctan(0.1)), abs=1e-4)


def test_ground_falling_to_the_south_faces_south_and_counts_negative():
    out = terrain_features(_plane(rise_per_row=-1.0), PX, PX, 31)
    assert (out[..., 2] < 0).all()


def test_east_facing_slope_is_neither_north_nor_south():
    out = terrain_features(_plane(rise_per_col=1.0), PX, PX, 31)
    assert out[..., 0].mean() > 5
    assert out[..., 2] == pytest.approx(0.0, abs=1e-6)


def test_hilltop_has_positive_tpi_and_valley_negative():
    r, c = np.mgrid[0:41, 0:41]
    hill = 1000 + 50 * np.exp(-((r - 20) ** 2 + (c - 20) ** 2) / 50.0)
    out = terrain_features(hill.astype(np.float32), PX, PX, 31)
    assert out[20, 20, 1] > 10
    assert out[20, 20, 0] == pytest.approx(0.0, abs=1e-6)  # flat right on the top

    out_valley = terrain_features((2000 - hill).astype(np.float32), PX, PX, 31)
    assert out_valley[20, 20, 1] < -10


def test_missing_elevation_stays_missing_without_spoiling_neighbours():
    elev = _plane(rise_per_row=1.0)
    elev[0, 0] = np.nan
    out = terrain_features(elev, PX, PX, 31)
    assert np.isnan(out[0, 0]).all()
    assert np.isfinite(out[20, 20]).all()


def test_mosaic_fills_each_part_from_the_tile_that_covers_it():
    north = np.array([[5.0, 5.0], [np.nan, np.nan]], dtype=np.float32)
    south = np.array([[np.nan, np.nan], [7.0, 7.0]], dtype=np.float32)
    assert mosaic_tiles([north, south]).tolist() == [[5.0, 5.0], [7.0, 7.0]]


def _write_dem_like(s2_path, dem_path, elevation):
    with rasterio.open(s2_path) as ref:
        profile = {"driver": "GTiff", "count": 1, "dtype": "float32", "crs": ref.crs,
                   "transform": ref.transform, "height": elevation.shape[0],
                   "width": elevation.shape[1]}
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(elevation[np.newaxis].astype(np.float32))


def test_pipeline_adds_terrain_features_per_segment(tmp_path, write_synthetic_raster):
    imagery_dir, dem_dir = tmp_path / "imagery", tmp_path / "dem"
    imagery_dir.mkdir()
    dem_dir.mkdir()
    s2 = write_synthetic_raster(imagery_dir / "mine_001.tif", seed=1)
    _write_dem_like(s2, dem_dir / "mine_001.tif", _plane(rise_per_row=1.0))

    labels_path = tmp_path / "labels.gpkg"
    gpd.GeoDataFrame({"mine_id": []}, geometry=[], crs="EPSG:32632").to_file(
        labels_path, driver="GPKG"
    )
    cfg = dict(CONFIG)
    cfg.update({"imagery_dir": str(imagery_dir), "labels_path": str(labels_path),
                "n_jobs": 1, "dem_dir": str(dem_dir)})

    features, _ = build_dataset(cfg)

    assert features["dem_slope_mean"].to_numpy() == pytest.approx(np.degrees(np.arctan(0.1)), abs=1e-3)
    assert {"dem_tpi_mean", "dem_northness_mean", "dem_slope_std"} <= set(features.columns)
    assert "dem_elevation_mean" not in features.columns


def test_dem_on_a_different_grid_is_rejected(tmp_path, write_synthetic_raster):
    s2 = write_synthetic_raster(tmp_path / "mine_001.tif", seed=1, size=40)
    dem_dir = tmp_path / "dem"
    dem_dir.mkdir()
    _write_dem_like(s2, dem_dir / "mine_001.tif", _plane(rows=30, cols=30))

    with pytest.raises(ValueError, match="Raster"):
        load_dem_for_mine(s2, str(dem_dir), (40, 40), 31)
