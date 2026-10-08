"""Tests for area_recall(): the segment-size-independent yardstick (flag the
top q% of each mine's area, count known dump polygons it touches)."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from eval_segment_size import area_recall


def _grid(n=10, size=10):
    geoms, ids = [], []
    for i in range(n):
        geoms.append(box(i * size, 0, (i + 1) * size, size))
        ids.append(i)
    return gpd.GeoDataFrame({"mine_id": "m", "segment_id": ids}, geometry=geoms, crs="EPSG:32719")


def test_dump_in_the_top_segment_is_caught_at_small_q():
    poly = _grid()
    oof = pd.DataFrame({"mine_id": "m", "segment_id": range(10), "dump_proba": [0.1] * 3 + [0.9] + [0.1] * 6})
    labels = gpd.GeoDataFrame({"mine_id": ["m"]}, geometry=[box(31, 2, 35, 6)], crs="EPSG:32719")
    res = area_recall(oof, poly, labels, qs=(10, 50))
    assert res[10] == (1.0, 1.0)   # top 10% of the area = segment 3 = the dump


def test_dump_in_a_low_scored_segment_is_missed_until_q_is_large():
    poly = _grid()
    oof = pd.DataFrame({"mine_id": "m", "segment_id": range(10), "dump_proba": [0.9] + [0.1] * 9})
    labels = gpd.GeoDataFrame({"mine_id": ["m"]}, geometry=[box(81, 2, 85, 6)], crs="EPSG:32719")
    res = area_recall(oof, poly, labels, qs=(10, 100))
    assert res[10][0] == 0.0 and res[100][0] == 1.0


def test_result_does_not_depend_on_segment_size():
    # same ground, same dump, cut into 10 big or 20 small segments, dump segment scored highest
    labels = gpd.GeoDataFrame({"mine_id": ["m"]}, geometry=[box(31, 2, 34, 6)], crs="EPSG:32719")
    coarse = _grid(10, 10)
    fine = _grid(20, 5)
    oof_c = pd.DataFrame({"mine_id": "m", "segment_id": range(10), "dump_proba": [0.9 if i == 3 else 0.1 for i in range(10)]})
    oof_f = pd.DataFrame({"mine_id": "m", "segment_id": range(20), "dump_proba": [0.9 if i == 6 else 0.1 for i in range(20)]})
    assert area_recall(oof_c, coarse, labels, qs=(10,))[10][0] == area_recall(oof_f, fine, labels, qs=(5, 10))[10][0] == 1.0


def test_rows_without_predictions_are_ignored():
    poly = _grid()
    oof = pd.DataFrame({"mine_id": "m", "segment_id": range(10), "dump_proba": [None] * 5 + [0.9] + [0.1] * 4})
    labels = gpd.GeoDataFrame({"mine_id": ["m"]}, geometry=[box(51, 2, 55, 6)], crs="EPSG:32719")
    assert area_recall(oof, poly, labels, qs=(20,))[20][0] == 1.0


def test_load_run_adds_the_polygons_of_extra_tiles(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from eval_segment_size import load_run

    pd.DataFrame({"mine_id": ["m"], "segment_id": [1], "dump_proba": [0.5]}).to_pickle(tmp_path / "oof_predictions.pkl")
    mk = lambda mid: gpd.GeoDataFrame({"mine_id": [mid], "segment_id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32719")
    mk("m").to_file(tmp_path / "polygons_cache.gpkg", driver="GPKG")
    assert len(load_run(str(tmp_path))[1]) == 1
    mk("tile").to_file(tmp_path / "polygons_extra.gpkg", driver="GPKG")
    assert sorted(load_run(str(tmp_path))[1]["mine_id"]) == ["m", "tile"]
