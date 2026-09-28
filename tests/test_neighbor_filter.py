"""Tests for apply_neighbor_filter(): a low-confidence positive segment
needs an adjacent positive segment to survive, but a high-confidence one
never does. The high-confidence exemption exists because real dumps found
in the wild are sometimes a single segment (mine_012's only known dump is
exactly this: one high-confidence segment with no neighbor at all) - a
plain "requires at least 2 segments" rule would have deleted that hit."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from otr_obia_pipeline import apply_neighbor_filter


def _result(rows, geoms):
    df = pd.DataFrame(rows, columns=["mine_id", "dump_proba", "dump_pred"])
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:32719")


def test_isolated_low_confidence_segment_is_dropped():
    result = _result(
        [("mine_1", 0.55, 1)],
        [box(0, 0, 10, 10)],
    )

    out = apply_neighbor_filter(result, high_confidence=0.6)

    assert out["dump_pred"].tolist() == [0]


def test_isolated_high_confidence_segment_is_kept():
    result = _result(
        [("mine_1", 0.90, 1)],
        [box(0, 0, 10, 10)],
    )

    out = apply_neighbor_filter(result, high_confidence=0.6)

    assert out["dump_pred"].tolist() == [1]


def test_low_confidence_segment_with_a_touching_neighbor_is_kept():
    result = _result(
        [("mine_1", 0.55, 1), ("mine_1", 0.55, 1)],
        [box(0, 0, 10, 10), box(10, 0, 20, 10)],  # share an edge
    )

    out = apply_neighbor_filter(result, high_confidence=0.6)

    assert out["dump_pred"].tolist() == [1, 1]


def test_low_confidence_segment_supported_only_by_a_negative_neighbor_is_dropped():
    result = _result(
        [("mine_1", 0.55, 1), ("mine_1", 0.55, 0)],
        [box(0, 0, 10, 10), box(10, 0, 20, 10)],
    )

    out = apply_neighbor_filter(result, high_confidence=0.6)

    assert out["dump_pred"].tolist() == [0, 0]


def test_neighbor_support_does_not_cross_mine_boundaries():
    # two touching segments, but they belong to different mines -> no support
    result = _result(
        [("mine_1", 0.55, 1), ("mine_2", 0.55, 1)],
        [box(0, 0, 10, 10), box(10, 0, 20, 10)],
    )

    out = apply_neighbor_filter(result, high_confidence=0.6)

    assert out["dump_pred"].tolist() == [0, 0]


def test_already_negative_segments_are_left_alone():
    result = _result(
        [("mine_1", 0.1, 0)],
        [box(0, 0, 10, 10)],
    )

    out = apply_neighbor_filter(result, high_confidence=0.6)

    assert out["dump_pred"].tolist() == [0]


def test_empty_result_is_a_no_op():
    result = _result([], [])
    out = apply_neighbor_filter(result, high_confidence=0.6)
    assert out.empty
