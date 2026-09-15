"""Tests for label_segments(): deciding whether a patch counts as a
tire dump based on how much of it overlaps a digitized dump polygon."""
import geopandas as gpd
from shapely.geometry import box

from otr_obia_pipeline import compute_shape_features, label_segments

DUMP = box(0, 0, 10, 10)


def _make_segments():
    gdf = gpd.GeoDataFrame(
        {"mine_id": ["1", "1", "1"], "segment_id": [1, 2, 3]},
        geometry=[
            box(0, 0, 10, 10),        # segment 1: identical to the dump -> fully covered
            box(5, 5, 20, 20),        # segment 2: only ~11% overlap -> below threshold
            box(100, 100, 110, 110),  # segment 3: nowhere near the dump
        ],
        crs="EPSG:32632",
    )
    return compute_shape_features(gdf)


def test_fully_covered_segment_is_positive():
    labels_gdf = gpd.GeoDataFrame(geometry=[DUMP], crs="EPSG:32632")
    result = label_segments(_make_segments(), labels_gdf, min_overlap_ratio=0.30)
    labels = result.set_index("segment_id")["label"]
    assert labels[1] == 1


def test_below_overlap_threshold_is_negative():
    labels_gdf = gpd.GeoDataFrame(geometry=[DUMP], crs="EPSG:32632")
    result = label_segments(_make_segments(), labels_gdf, min_overlap_ratio=0.30)
    labels = result.set_index("segment_id")["label"]
    assert labels[2] == 0


def test_far_away_segment_is_negative():
    labels_gdf = gpd.GeoDataFrame(geometry=[DUMP], crs="EPSG:32632")
    result = label_segments(_make_segments(), labels_gdf, min_overlap_ratio=0.30)
    labels = result.set_index("segment_id")["label"]
    assert labels[3] == 0


def test_no_labels_at_all_means_everything_negative():
    empty_labels = gpd.GeoDataFrame(geometry=[], crs="EPSG:32632")
    result = label_segments(_make_segments(), empty_labels, min_overlap_ratio=0.30)
    assert (result["label"] == 0).all()


def test_overlap_columns_exist_for_leakage_exclusion():
    # regression guard: train_and_evaluate() must exclude these columns from
    # the model's inputs, since overlap_ratio directly encodes the label.
    labels_gdf = gpd.GeoDataFrame(geometry=[DUMP], crs="EPSG:32632")
    result = label_segments(_make_segments(), labels_gdf, min_overlap_ratio=0.30)
    assert "overlap_ratio" in result.columns
    assert "overlap_area" in result.columns
