"""Tests for the --use-cache option: reusing a previous run's segment
dataset instead of rebuilding it, since that's the expensive step
(~25 minutes on the real 138-mine dataset)."""
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import box

from otr_obia_pipeline import CONFIG, main


def _make_cfg(tmp_path, imagery_dir, write_synthetic_raster):
    imagery_dir.mkdir()
    write_synthetic_raster(imagery_dir / "1.tif", seed=1)
    write_synthetic_raster(imagery_dir / "2.tif", seed=2)

    labels_gdf = gpd.GeoDataFrame(
        {"mine_id": [1]}, geometry=[box(500050, 5599850, 500150, 5599950)],
        crs="EPSG:32632",
    )
    labels_path = tmp_path / "labels.gpkg"
    labels_gdf.to_file(labels_path, driver="GPKG")

    cfg = dict(CONFIG)
    cfg.update({
        "imagery_dir": str(imagery_dir),
        "labels_path": str(labels_path),
        "mine_id_field": "mine_id",
        "n_segments_per_mine": 20,
        "target_segment_px": None,
        "min_overlap_ratio": 0.3,
        "output_dir": str(tmp_path / "output"),
        "use_cache": True,
        "n_jobs": 1,
    })
    return cfg


def test_second_run_reuses_cache_instead_of_rebuilding(tmp_path, write_synthetic_raster):
    cfg = _make_cfg(tmp_path, tmp_path / "imagery", write_synthetic_raster)

    with patch("otr_obia_pipeline.build_dataset", wraps=__import__(
        "otr_obia_pipeline"
    ).build_dataset) as spy:
        main(cfg)
        assert spy.call_count == 1  # first run: no cache yet, must build

        main(cfg)
        assert spy.call_count == 1  # second run: cache exists, must NOT rebuild


def test_extra_imagery_dir_scores_only_the_new_tile(tmp_path, write_synthetic_raster):
    cfg = _make_cfg(tmp_path, tmp_path / "imagery", write_synthetic_raster)
    main(cfg)  # builds the cache from the two known mines

    extra = tmp_path / "extra"
    extra.mkdir()
    write_synthetic_raster(extra / "3.tif", seed=3)
    cfg2 = dict(cfg, extra_imagery_dir=str(extra))
    main(cfg2)

    out = gpd.read_file(tmp_path / "output" / "segments_classified.gpkg")
    assert set(out["mine_id"].astype(str)) == {"3"}
    assert len(out) > 0


def test_extra_imagery_dir_without_cache_is_an_error(tmp_path, write_synthetic_raster):
    import pytest
    cfg = _make_cfg(tmp_path, tmp_path / "imagery", write_synthetic_raster)
    extra = tmp_path / "extra"
    extra.mkdir()
    write_synthetic_raster(extra / "3.tif", seed=3)
    with pytest.raises(ValueError, match="Zwischenspeicher"):
        main(dict(cfg, extra_imagery_dir=str(extra)))


def test_extra_tiles_are_scored_but_never_trained_on(tmp_path, write_synthetic_raster):
    from unittest.mock import patch
    import otr_obia_pipeline as pipe
    cfg = _make_cfg(tmp_path, tmp_path / "imagery", write_synthetic_raster)
    main(cfg)
    extra = tmp_path / "extra"
    extra.mkdir()
    write_synthetic_raster(extra / "3.tif", seed=3)

    seen = {}
    real = pipe.train_and_evaluate

    def spy(feature_df, polygons_gdf, cfg_):
        seen["train_mines"] = set(feature_df["mine_id"].astype(str))
        return real(feature_df, polygons_gdf, cfg_)

    with patch("otr_obia_pipeline.train_and_evaluate", side_effect=spy):
        main(dict(cfg, extra_imagery_dir=str(extra)))
    assert "3" not in seen["train_mines"] and seen["train_mines"]
