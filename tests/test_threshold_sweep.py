"""Tests for report_threshold_sweep(): the out-of-fold detection-rate vs.
false-positive trade-off at several classification thresholds, now a
permanent part of the pipeline's own output instead of a one-off script."""
import numpy as np
import pandas as pd

from otr_obia_pipeline import report_threshold_sweep


def _feature_df(rows):
    return pd.DataFrame(rows, columns=["mine_id", "segment_id", "label"])


def test_higher_threshold_reduces_positives_and_can_lose_detections():
    feature_df = _feature_df([
        ("mine_1", 1, 1),  # true dump, proba just above 0.5
        ("mine_1", 2, 0),  # false positive, proba just above 0.5
        ("mine_2", 1, 1),  # true dump, high-confidence proba
        ("mine_2", 2, 0),
    ])
    proba = np.array([0.55, 0.55, 0.95, 0.1])

    sweep = report_threshold_sweep(feature_df, proba, thresholds=(0.5, 0.9))

    row_05 = sweep[sweep["threshold"] == 0.5].iloc[0]
    row_09 = sweep[sweep["threshold"] == 0.9].iloc[0]
    assert row_05["mines_detected"] == 2
    assert row_05["n_pred_positive"] == 3  # both true dumps + the false positive
    # raising the threshold drops mine_1's borderline detection but keeps
    # mine_2's high-confidence one, and reduces false positives
    assert row_09["mines_detected"] == 1
    assert row_09["n_pred_positive"] == 1


def test_no_dump_mines_returns_none():
    feature_df = _feature_df([
        ("mine_1", 1, 0),
        ("mine_1", 2, 0),
    ])
    proba = np.array([0.9, 0.1])

    assert report_threshold_sweep(feature_df, proba) is None
