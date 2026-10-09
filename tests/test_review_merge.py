"""Merging review rounds, and training only on the reviewed parts of partly-reviewed tiles."""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from merge_review_outputs import merge
from otr_obia_pipeline import restrict_to_reviewed


def _gdf(rows, crs="EPSG:32719"):
    return gpd.GeoDataFrame({"mine_id": [r[0] for r in rows]}, geometry=[r[1] for r in rows], crs=crs)


def test_merge_appends_new_rounds_and_removes_duplicates(tmp_path):
    base, new = tmp_path / "base", tmp_path / "new"
    base.mkdir(); new.mkdir()
    a, b, c = box(0, 0, 1, 1), box(5, 5, 6, 6), box(9, 9, 10, 10)
    _gdf([("t1", a)]).to_file(base / "labels_reviewed_dump_all.gpkg", driver="GPKG")
    _gdf([("t1", b)]).to_file(base / "negatives_all_clean.gpkg", driver="GPKG")
    gpd.GeoDataFrame(geometry=[c], crs="EPSG:32719").to_file(base / "ignore_reviewed_all.geojson", driver="GeoJSON")  # old: no mine_id
    _gdf([("t1", a), ("t2", c)]).to_file(new / "labels_reviewed_dump.gpkg", driver="GPKG")      # a is a duplicate
    _gdf([("t3", b)]).to_file(new / "negatives_reviewed_clean.gpkg", driver="GPKG")
    # no ignore file in the new round -> fine
    _gdf([("known", box(20, 20, 21, 21))]).to_file(tmp_path / "eval.gpkg", driver="GPKG")
    report, tiles = merge(str(base), "all", "all3", [str(new)], str(tmp_path / "eval.gpkg"), str(tmp_path / "tiles.txt"))
    assert report["dump"] == (1, 2) and report["neg"] == (1, 2) and report["ign"] == (1, 1)
    assert report["eval"] == (1, 3)
    assert tiles == ["t1", "t2", "t3"]
    assert len(gpd.read_file(base / "labels_reviewed_dump_all3.gpkg")) == 2
    assert (tmp_path / "tiles.txt").read_text().split() == ["t1", "t2", "t3"]


def test_unreviewed_negatives_of_partly_reviewed_tiles_do_not_train():
    feat = pd.DataFrame({
        "mine_id": ["q", "q", "q", "q", "m", "m"], "segment_id": [0, 1, 2, 3, 0, 1],
        "label": [1, 0, 0, 0, 0, 0]})
    polys = gpd.GeoDataFrame(feat[["mine_id", "segment_id"]].copy(),
                             geometry=[box(i, 0, i + 1, 1) for i in range(6)], crs="EPSG:32719")
    hard = np.array([False, True, False, False, False, False])      # q/1 is a confirmed clean site
    f, p, h = restrict_to_reviewed(feat, polys, hard, ["q"])
    kept = set(zip(f["mine_id"], f["segment_id"]))
    assert kept == {("q", 0), ("q", 1), ("m", 0), ("m", 1)}          # positive, hard negative, normal mine untouched
    assert len(p) == len(f) == len(h) and h.sum() == 1
    assert set(zip(p["mine_id"], p["segment_id"])) == kept
