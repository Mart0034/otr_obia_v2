"""Regression test: if one mine's processing fails (corrupt file, resource
limit, etc.), build_dataset() must keep the other mines' results instead of
discarding everything. This happened for real: a 3-mine run lost a
successfully-completed mine's ~13 minutes of work because one of the other
two mines hit a memory error."""
import geopandas as gpd
import pytest
from shapely.geometry import box

from otr_obia_pipeline import CONFIG, build_dataset


def test_one_failing_mine_does_not_discard_the_others(tmp_path, write_synthetic_raster, monkeypatch):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()
    write_synthetic_raster(imagery_dir / "1.tif", seed=1)
    write_synthetic_raster(imagery_dir / "2.tif", seed=2)
    write_synthetic_raster(imagery_dir / "3.tif", seed=3)

    labels_gdf = gpd.GeoDataFrame({"mine_id": []}, geometry=[], crs="EPSG:32632")
    labels_path = tmp_path / "labels.gpkg"
    labels_gdf.to_file(labels_path, driver="GPKG")

    cfg = dict(CONFIG)
    cfg.update({
        "imagery_dir": str(imagery_dir),
        "labels_path": str(labels_path),
        "mine_id_field": "mine_id",
        "n_segments_per_mine": 20,
        "target_segment_px": None,
        "n_jobs": 1,  # deterministic order for the monkeypatch below
    })

    import otr_obia_pipeline as pipe
    real_process_one_mine = pipe._process_one_mine

    def flaky_process_one_mine(path, *args, **kwargs):
        if "2.tif" in path:
            raise MemoryError("simulated failure for mine 2")
        return real_process_one_mine(path, *args, **kwargs)

    monkeypatch.setattr(pipe, "_process_one_mine", flaky_process_one_mine)

    feature_df, polygons_gdf = build_dataset(cfg)

    assert set(feature_df["mine_id"]) == {"1", "3"}
    assert len(feature_df) > 0


def test_all_mines_failing_raises_clear_error(tmp_path, write_synthetic_raster, monkeypatch):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()
    write_synthetic_raster(imagery_dir / "1.tif", seed=1)

    labels_gdf = gpd.GeoDataFrame({"mine_id": []}, geometry=[], crs="EPSG:32632")
    labels_path = tmp_path / "labels.gpkg"
    labels_gdf.to_file(labels_path, driver="GPKG")

    cfg = dict(CONFIG)
    cfg.update({
        "imagery_dir": str(imagery_dir),
        "labels_path": str(labels_path),
        "mine_id_field": "mine_id",
        "n_jobs": 1,
    })

    import otr_obia_pipeline as pipe

    def always_fails(path, *args, **kwargs):
        raise MemoryError("simulated total failure")

    monkeypatch.setattr(pipe, "_process_one_mine", always_fails)

    with pytest.raises(RuntimeError, match="Alle Minen sind fehlgeschlagen"):
        build_dataset(cfg)
