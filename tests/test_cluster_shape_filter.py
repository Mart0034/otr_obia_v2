"""Tests for apply_cluster_shape_filter(): a low-confidence positive
segment needs its connected cluster of touching positive segments to be
reasonably blob-shaped, not a long thin chain (a road or cliff edge).

This is deliberately separate from apply_neighbor_filter(): a chain of
segments running along a road all touch each other, so every one of them
already has a supporting neighbor and would survive that filter untouched.
What's wrong with a chain isn't isolation, it's shape - which is what
cluster compactness measures instead."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from otr_obia_pipeline import apply_cluster_shape_filter

# A straight chain of N touching 10x10 segments has compactness pi*N/(N+1)^2
# (rectangle N*10 x 10): N=25 -> ~0.12, safely under the 0.15 default.
# N=4 -> ~0.50, well above it - short chains are NOT automatically "thin".
CHAIN_LEN = 25


def _result(rows, geoms):
    df = pd.DataFrame(rows, columns=["mine_id", "dump_proba", "dump_pred"])
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:32719")


def _chain(n, size=10):
    # a long, one-segment-wide, straight chain (worst case for compactness)
    return [box(i * size, 0, (i + 1) * size, size) for i in range(n)]


def _blob(n_side, size=10):
    # a roughly square block of touching segments (compact, dump-like)
    return [
        box(x * size, y * size, (x + 1) * size, (y + 1) * size)
        for x in range(n_side) for y in range(n_side)
    ]


def test_low_confidence_chain_is_dropped():
    geoms = _chain(CHAIN_LEN)
    rows = [("mine_1", 0.55, 1) for _ in geoms]
    result = _result(rows, geoms)

    out = apply_cluster_shape_filter(result, high_confidence=0.6, min_compactness=0.15)

    assert (out["dump_pred"] == 0).all()


def test_short_chain_is_not_thin_enough_to_be_dropped():
    # a short chain isn't automatically "road-like" - compactness has to
    # actually be low, not just "more than one segment in a line"
    geoms = _chain(4)
    rows = [("mine_1", 0.55, 1) for _ in geoms]
    result = _result(rows, geoms)

    out = apply_cluster_shape_filter(result, high_confidence=0.6, min_compactness=0.15)

    assert (out["dump_pred"] == 1).all()


def test_low_confidence_compact_blob_is_kept():
    geoms = _blob(3)  # 3x3 block, compact
    rows = [("mine_1", 0.55, 1) for _ in geoms]
    result = _result(rows, geoms)

    out = apply_cluster_shape_filter(result, high_confidence=0.6, min_compactness=0.15)

    assert (out["dump_pred"] == 1).all()


def test_high_confidence_segment_in_a_chain_is_kept_regardless_of_shape():
    geoms = _chain(CHAIN_LEN)
    rows = [("mine_1", 0.55, 1) for _ in geoms]
    rows[10] = ("mine_1", 0.95, 1)  # one high-confidence segment in the middle of the chain
    result = _result(rows, geoms)

    out = apply_cluster_shape_filter(result, high_confidence=0.6, min_compactness=0.15)

    expected = [0] * CHAIN_LEN
    expected[10] = 1
    assert out["dump_pred"].tolist() == expected


def test_isolated_single_segment_is_unaffected_by_this_filter():
    # a lone segment forms a trivially "compact" cluster of one - this
    # filter has nothing to say about isolation, that's apply_neighbor_filter's job
    result = _result([("mine_1", 0.55, 1)], [box(0, 0, 10, 10)])

    out = apply_cluster_shape_filter(result, high_confidence=0.6, min_compactness=0.15)

    assert out["dump_pred"].tolist() == [1]


def test_chain_support_does_not_cross_mine_boundaries():
    # two separate mines, each with its own thin chain - shape is judged
    # per mine, not by (incorrectly) merging geometry across mines
    geoms_1 = _chain(CHAIN_LEN)
    geoms_2 = _chain(CHAIN_LEN)
    rows = [("mine_1", 0.55, 1) for _ in geoms_1] + [("mine_2", 0.55, 1) for _ in geoms_2]
    result = _result(rows, geoms_1 + geoms_2)

    out = apply_cluster_shape_filter(result, high_confidence=0.6, min_compactness=0.15)

    assert (out["dump_pred"] == 0).all()


def test_already_negative_segments_are_left_alone():
    result = _result([("mine_1", 0.1, 0)], [box(0, 0, 10, 10)])
    out = apply_cluster_shape_filter(result, high_confidence=0.6, min_compactness=0.15)
    assert out["dump_pred"].tolist() == [0]


def test_empty_result_is_a_no_op():
    result = _result([], [])
    out = apply_cluster_shape_filter(result, high_confidence=0.6, min_compactness=0.15)
    assert out.empty
