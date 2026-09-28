"""Tests for report_threshold_sweep(): the out-of-fold detection-rate vs.
false-positive trade-off at several classification thresholds, now a
permanent part of the pipeline's own output instead of a one-off script."""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from otr_obia_pipeline import report_threshold_sweep


def _result(rows, geoms=None):
    df = pd.DataFrame(rows, columns=["mine_id", "segment_id", "label", "dump_proba"])
    if geoms is None:
        return df
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:32719")


def test_higher_threshold_reduces_positives_and_can_lose_detections():
    result = _result([
        ("mine_1", 1, 1, 0.55),  # true dump, proba just above 0.5
        ("mine_1", 2, 0, 0.55),  # false positive, proba just above 0.5
        ("mine_2", 1, 1, 0.95),  # true dump, high-confidence proba
        ("mine_2", 2, 0, 0.10),
    ])

    sweep = report_threshold_sweep(result, thresholds=(0.5, 0.9))

    row_05 = sweep[sweep["threshold"] == 0.5].iloc[0]
    row_09 = sweep[sweep["threshold"] == 0.9].iloc[0]
    assert row_05["mines_detected"] == 2
    assert row_05["n_pred_positive"] == 3  # both true dumps + the false positive
    # raising the threshold drops mine_1's borderline detection but keeps
    # mine_2's high-confidence one, and reduces false positives
    assert row_09["mines_detected"] == 1
    assert row_09["n_pred_positive"] == 1


def test_no_dump_mines_returns_none():
    result = _result([
        ("mine_1", 1, 0, 0.9),
        ("mine_1", 2, 0, 0.1),
    ])

    assert report_threshold_sweep(result) is None


def test_require_neighbor_below_drops_isolated_medium_confidence_segments():
    # mine_1: two adjacent medium-confidence false positives support each
    # other (kept), one isolated medium-confidence false positive far away
    # (dropped), and the true dump is a single HIGH-confidence segment with
    # no neighbor at all (kept regardless - this is the mine_012 case).
    result = _result(
        [
            ("mine_1", 1, 0, 0.55),  # medium-confidence, has neighbor -> kept
            ("mine_1", 2, 0, 0.55),  # medium-confidence, has neighbor -> kept
            ("mine_1", 3, 0, 0.55),  # medium-confidence, isolated -> dropped
            ("mine_1", 4, 1, 0.90),  # true dump, isolated but high-confidence -> kept
        ],
        geoms=[
            box(0, 0, 10, 10),
            box(10, 0, 20, 10),   # touches segment 1
            box(1000, 0, 1010, 10),  # far from everything
            box(2000, 0, 2010, 10),  # far from everything, but high confidence
        ],
    )

    sweep = report_threshold_sweep(result, thresholds=(0.5,), require_neighbor_below=0.6)

    row = sweep.iloc[0]
    assert row["n_pred_positive"] == 3  # segments 1, 2, 4 - segment 3 dropped
    assert row["mines_detected"] == 1  # the single-segment true dump still counts
