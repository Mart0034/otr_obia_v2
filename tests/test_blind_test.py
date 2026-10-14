import json
import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import blind_test as bt  # noqa: E402


def test_choose_folds_balances_and_covers_every_tile():
    truth = pd.DataFrame({"mine_id": ["a"] * 10 + ["b"] * 6 + ["c"] * 5 + ["d"] * 4 + ["e"] * 1})
    groups = bt.choose_folds(truth, 3, seed=1)
    assert sorted(sum(groups, [])) == ["a", "b", "c", "d", "e"]
    loads = [int(truth["mine_id"].isin(g).sum()) for g in groups]
    assert max(loads) - min(loads) <= 6


def test_score_counts_places_found_by_lists(tmp_path):
    truth = gpd.GeoDataFrame({"mine_id": ["t1", "t1", "t2"]},
                             geometry=[box(0, 0, 30, 30), box(40, 0, 70, 30), box(5000, 0, 5030, 30)], crs=32719)
    truth.to_file(tmp_path / "truth.gpkg", driver="GPKG")
    sites = gpd.GeoDataFrame({"site_id": ["S1"]}, geometry=[box(-10, -10, 80, 40)], crs=32719)
    sites.to_file(tmp_path / "list.gpkg", driver="GPKG", layer="candidate_sites")
    a = type("A", (), {"out": str(tmp_path), "fold": 0, "lists": [str(tmp_path / "list.gpkg")]})
    bt.score(a)
    res = json.load(open(tmp_path / "score.json"))
    assert res["hidden_places"] == 2                                         # die ersten beiden Polygone sind ein Ort
    assert res["lists"]["list.gpkg"]["found_places"] == 1 and res["lists"]["list.gpkg"]["recall"] == 0.5
