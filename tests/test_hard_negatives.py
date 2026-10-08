"""Tests for hard-negative marking, training-time repetition and the false-alarm report."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from eval_false_alarms import false_alarm_share
from otr_obia_pipeline import mark_hard_negatives, train_and_evaluate


def _polys(mine, n=4):
    return gpd.GeoDataFrame({"mine_id": mine, "segment_id": range(1, n + 1)},
                            geometry=[box(i * 10, 0, i * 10 + 10, 10) for i in range(n)], crs="EPSG:32719")


def test_mark_hard_negatives_needs_half_overlap_and_label_zero():
    polygons = _polys("m1")
    feats = pd.DataFrame({"mine_id": "m1", "segment_id": [1, 2, 3, 4], "label": [0, 0, 1, 0]})
    neg = gpd.GeoDataFrame({"mine_id": ["m1"]}, geometry=[box(0, 0, 25, 10)], crs="EPSG:32719")
    # covers segment 1 and 2 fully, segment 3 by half (label 1 -> never a hard negative), 4 not at all
    mask = mark_hard_negatives(feats, polygons, neg, min_overlap=0.5)
    assert mask.tolist() == [True, True, False, False]


def _toy(n_mines=8, per_mine=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_mines):
        x = rng.normal(size=per_mine)
        label = (x > 1.3).astype(int)
        rows.append(pd.DataFrame({"mine_id": f"m{m}", "segment_id": range(1, per_mine + 1),
                                  "f1": x, "f2": rng.normal(size=per_mine), "label": label}))
    feats = pd.concat(rows, ignore_index=True)
    polygons = pd.concat([_polys(f"m{m}", per_mine) for m in range(n_mines)], ignore_index=True)
    return feats, gpd.GeoDataFrame(polygons, crs="EPSG:32719")


def test_training_with_hard_negatives_keeps_one_oof_row_per_segment(tmp_path):
    feats, polygons = _toy()
    cfg = {"n_estimators": 20, "random_state": 0, "output_dir": str(tmp_path)}
    hard = (feats["label"] == 0).values & (feats["f1"].values > 0.5)          # some ordinary negatives
    _, _, scores = train_and_evaluate(feats, polygons, cfg, hard_neg_mask=hard, hard_neg_factor=5)
    oof = pd.read_pickle(tmp_path / "oof_predictions.pkl")
    assert len(oof) == len(feats) and oof["dump_proba"].notna().all()
    assert not oof.duplicated(["mine_id", "segment_id"]).any()


def test_false_alarm_share_counts_flagged_clean_area():
    polygons = _polys("m1", 10)
    scores = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    oof = pd.DataFrame({"mine_id": "m1", "segment_id": range(1, 11), "dump_proba": scores})
    neg = gpd.GeoDataFrame({"mine_id": ["m1", "m1"]},
                           geometry=[box(0, 0, 10, 10), box(50, 0, 60, 10)], crs="EPSG:32719")   # seg 1 and seg 6
    res, mean_score = false_alarm_share(oof, polygons, neg, qs=(10, 50))
    assert res[10] == (pytest.approx(0.5), pytest.approx(0.5))     # top 10% = segment 1 only -> half of the clean area
    assert mean_score == pytest.approx(0.5)                        # (0.9 + 0.1) / 2
