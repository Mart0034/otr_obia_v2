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


def _extra_setup(tmp_path, write_synthetic_raster):
    cfg = _make_cfg(tmp_path, tmp_path / "imagery", write_synthetic_raster)
    main(cfg)
    extra = tmp_path / "extra"
    extra.mkdir()
    write_synthetic_raster(extra / "3.tif", seed=3)
    return cfg, extra


def test_labeled_tiles_are_trained_on_with_their_labels(tmp_path, write_synthetic_raster):
    from unittest.mock import patch
    import otr_obia_pipeline as pipe
    cfg, extra = _extra_setup(tmp_path, write_synthetic_raster)
    labels = gpd.GeoDataFrame({"mine_id": [3]}, geometry=[box(500050, 5599850, 500150, 5599950)],
                              crs="EPSG:32632")
    labels_path = tmp_path / "extra_labels.gpkg"
    labels.to_file(labels_path, driver="GPKG")

    seen = {}
    real = pipe.train_and_evaluate

    def spy(feature_df, polygons_gdf, cfg_):
        seen["mines"] = set(feature_df["mine_id"].astype(str))
        seen["pos3"] = int(feature_df.loc[feature_df["mine_id"].astype(str) == "3", "label"].sum())
        return real(feature_df, polygons_gdf, cfg_)

    with patch("otr_obia_pipeline.train_and_evaluate", side_effect=spy):
        main(dict(cfg, labeled_imagery_dir=str(extra), extra_labels_path=str(labels_path)))
    assert "3" in seen["mines"] and seen["pos3"] > 0


def test_ignore_zone_drops_only_negatives_near_the_points(tmp_path):
    import numpy as np
    import pandas as pd
    from otr_obia_pipeline import drop_ignored_negatives
    polys = gpd.GeoDataFrame(
        {"mine_id": ["m"] * 4, "segment_id": [1, 2, 3, 4]},
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10), box(900, 0, 910, 10)],
        crs="EPSG:32719")
    feats = pd.DataFrame({"mine_id": ["m"] * 4, "segment_id": [1, 2, 3, 4],
                          "label": [1, 0, 0, 0], "x": np.arange(4.0)})
    pts = tmp_path / "pts.gpkg"
    gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1).centroid], crs="EPSG:32719").to_file(pts, driver="GPKG")

    kept, kept_poly = drop_ignored_negatives(feats, polys, str(pts), radius_m=35)
    # seg 1 is positive (kept), seg 2 is a negative ~20 m away (dropped),
    # seg 3 is ~40 m away and seg 4 is far (both outside the 35 m zone, kept)
    assert kept["segment_id"].tolist() == [1, 3, 4]
    assert kept_poly["segment_id"].tolist() == [1, 3, 4]


def test_labeled_and_scored_tiles_are_handled_separately(tmp_path, write_synthetic_raster):
    from unittest.mock import patch
    import otr_obia_pipeline as pipe
    cfg, labeled = _extra_setup(tmp_path, write_synthetic_raster)
    scored = tmp_path / "scored"
    scored.mkdir()
    write_synthetic_raster(scored / "4.tif", seed=4)
    labels = gpd.GeoDataFrame({"mine_id": [3]}, geometry=[box(500050, 5599850, 500150, 5599950)],
                              crs="EPSG:32632")
    labels_path = tmp_path / "labels3.gpkg"
    labels.to_file(labels_path, driver="GPKG")
    seen = {}
    real = pipe.train_and_evaluate

    def spy(feature_df, polygons_gdf, cfg_):
        seen["mines"] = set(feature_df["mine_id"].astype(str))
        return real(feature_df, polygons_gdf, cfg_)

    with patch("otr_obia_pipeline.train_and_evaluate", side_effect=spy):
        main(dict(cfg, labeled_imagery_dir=str(labeled), extra_labels_path=str(labels_path),
                  extra_imagery_dir=str(scored)))
    assert "3" in seen["mines"] and "4" not in seen["mines"]
    out = gpd.read_file(tmp_path / "output" / "segments_classified.gpkg")
    assert set(out["mine_id"].astype(str)) == {"4"}


def test_apply_extra_labels_adds_positives_without_touching_others():
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from otr_obia_pipeline import apply_extra_labels

    def polys(mine):
        return gpd.GeoDataFrame({"mine_id": mine, "segment_id": [1, 2, 3, 4]},
                                geometry=[box(i * 10, 0, i * 10 + 10, 10) for i in range(4)], crs="EPSG:32719")

    polygons = pd.concat([polys("m1"), polys("m2")], ignore_index=True)
    polygons = gpd.GeoDataFrame(polygons, crs="EPSG:32719")
    feats = pd.DataFrame({"mine_id": ["m1"] * 4 + ["m2"] * 4, "segment_id": [1, 2, 3, 4] * 2,
                          "label": [0, 0, 0, 1, 0, 0, 0, 0]})
    labels = gpd.GeoDataFrame({"mine_id": ["m1"]}, geometry=[box(0, 0, 20, 10)], crs="EPSG:32719")  # covers segs 1+2
    out = apply_extra_labels(feats, polygons, labels, min_overlap_ratio=0.1)
    assert out.loc[out.mine_id == "m1", "label"].tolist() == [1, 1, 0, 1]     # old positive kept
    assert out.loc[out.mine_id == "m2", "label"].sum() == 0                    # other mine untouched
    assert feats["label"].sum() == 1                                           # input not modified


def test_drop_ignored_negatives_accepts_several_files(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, box
    from otr_obia_pipeline import drop_ignored_negatives

    polygons = gpd.GeoDataFrame({"mine_id": "m", "segment_id": [1, 2, 3]},
                                geometry=[box(i * 1000, 0, i * 1000 + 10, 10) for i in range(3)], crs="EPSG:32719")
    feats = pd.DataFrame({"mine_id": "m", "segment_id": [1, 2, 3], "label": [0, 0, 0]})
    for name, x in (("a.geojson", 5), ("b.geojson", 1005)):
        gpd.GeoDataFrame(geometry=[Point(x, 5)], crs="EPSG:32719").to_crs(4326).to_file(tmp_path / name, driver="GeoJSON")
    kept, _ = drop_ignored_negatives(feats, polygons, f"{tmp_path / 'a.geojson'},{tmp_path / 'b.geojson'}", 50)
    assert kept["segment_id"].tolist() == [3]


def test_reviewed_imagery_dir_scores_everything_but_trains_only_on_reviewed_parts(tmp_path, write_synthetic_raster):
    from unittest.mock import patch
    import otr_obia_pipeline as pipe

    cfg = _make_cfg(tmp_path, tmp_path / "imagery", write_synthetic_raster)
    main(cfg)  # cache from mines 1 and 2

    rev = tmp_path / "rev"
    rev.mkdir()
    write_synthetic_raster(rev / "3.tif", seed=3)
    # a confirmed dump and a confirmed clean site, both in tile 3
    add = gpd.GeoDataFrame({"mine_id": [3]}, geometry=[box(500050, 5599850, 500150, 5599950)], crs="EPSG:32632")
    neg = gpd.GeoDataFrame({"mine_id": [3]}, geometry=[box(500350, 5599550, 500450, 5599650)], crs="EPSG:32632")
    add.to_file(tmp_path / "add.gpkg", driver="GPKG")
    neg.to_file(tmp_path / "neg.gpkg", driver="GPKG")
    cfg2 = dict(cfg, reviewed_imagery_dir=str(rev), add_labels_path=str(tmp_path / "add.gpkg"),
                hard_negatives_path=str(tmp_path / "neg.gpkg"), hard_negative_factor=3)
    seen = {}
    real = pipe.train_and_evaluate

    def spy(train_feat, *a, **k):
        seen["tile3"] = train_feat[train_feat["mine_id"].astype(str) == "3"]
        seen["n_mines"] = train_feat["mine_id"].nunique()
        return real(train_feat, *a, **k)

    with patch.object(pipe, "train_and_evaluate", spy):
        main(cfg2)

    t3 = seen["tile3"]
    assert len(t3) > 0 and seen["n_mines"] == 3
    assert (t3["label"] == 1).sum() > 0                         # reviewed dump trains
    assert len(t3) < 30                                          # but not the whole tile (20+ unreviewed segments dropped)
    out = gpd.read_file(tmp_path / "output" / "segments_classified.gpkg")
    assert set(out["mine_id"].astype(str)) == {"3"}              # the whole reviewed tile is still scored and exported
    assert len(out) > len(t3)
