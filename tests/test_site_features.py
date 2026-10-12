"""Tests for eval_site_features: which segment features separate real dumps from dirt-heap false alarms."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from eval_site_features import auc_with_ci, combined_cv_auc, feature_auc_table, site_feature_table

X0, Y0 = 400000.0, 7400000.0


def test_auc_with_ci_perfect_none_and_reversed():
    a, lo, hi = auc_with_ci(np.arange(10, 20), np.arange(0, 10))
    assert a == 1.0 and lo > 0.9
    rng = np.random.default_rng(1)
    a2, lo2, hi2 = auc_with_ci(rng.normal(0, 1, 40), rng.normal(0, 1, 40))
    assert lo2 < 0.5 < hi2                                       # no separation: the interval contains 0.5
    assert auc_with_ci(np.arange(0, 10), np.arange(10, 20))[0] == 0.0
    assert np.isnan(auc_with_ci([1, 2], [3, 4, 5])[0])           # too few samples


def test_feature_table_marks_direction_and_certainty():
    rng = np.random.default_rng(0)
    pos = pd.DataFrame({"slope": rng.normal(2, 0.5, 20), "noise": rng.normal(0, 1, 20)})
    neg = pd.DataFrame({"slope": rng.normal(8, 0.5, 40), "noise": rng.normal(0, 1, 40)})
    table = pd.concat([pos, neg], ignore_index=True)
    is_pos = pd.Series([True] * 20 + [False] * 40)
    res = feature_auc_table(table, is_pos).set_index("merkmal")
    assert res.loc["slope", "richtung"] == "Halde niedriger" and res.loc["slope", "sicher"]
    assert res.loc["slope", "auc"] < 0.05 and not res.loc["noise", "sicher"]
    assert list(feature_auc_table(table, is_pos)["merkmal"])[0] == "slope"               # sorted by strength
    cv, _ = combined_cv_auc(table, is_pos.values, repeats=2)
    assert cv > 0.9


def test_site_features_are_averaged_over_the_segments_inside_each_site(tmp_path):
    cells = [(c, 0) for c in range(6)]
    polys = gpd.GeoDataFrame({"mine_id": "t1", "segment_id": range(6)},
                             geometry=[box(X0 + c * 10, Y0, X0 + c * 10 + 10, Y0 + 10) for c, _ in cells], crs="EPSG:32719")
    feats = pd.DataFrame({"mine_id": "t1", "segment_id": range(6), "label": 0, "dump_proba": 0.5,
                          "dem_slope": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]})
    pd.to_pickle((feats, polys), tmp_path / "t1.abc.pkl")
    sites = gpd.GeoDataFrame({"site_id": ["A", "B"], "mine_id": ["t1", "t1"]},
                             geometry=[box(X0, Y0, X0 + 20, Y0 + 10), box(X0 + 40, Y0, X0 + 60, Y0 + 10)], crs="EPSG:32719")
    t = site_feature_table(sites, str(tmp_path))
    assert t.loc["A", "dem_slope"] == pytest.approx(2.0) and t.loc["B", "dem_slope"] == pytest.approx(10.0)
    assert "dump_proba" not in t.columns and "label" not in t.columns
    assert t.loc["A", "n_seg_features"] == 2
