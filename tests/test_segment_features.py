"""Tests for compute_segment_features(): per-patch color and texture stats."""
import warnings

import numpy as np

from otr_obia_pipeline import compute_segment_features


def test_basic_stats_and_texture_columns():
    band_names = ["blue", "green", "dsi"]
    H, W = 10, 10
    arr = np.zeros((H, W, 3), dtype=np.float32)
    arr[:5, :, 0] = 10.0   # blue, top half
    arr[5:, :, 0] = 50.0   # blue, bottom half
    arr[..., 1] = 20.0     # green, constant everywhere
    arr[..., 2] = np.tile(np.arange(W, dtype=np.float32), (H, 1))  # dsi gradient

    segments = np.zeros((H, W), dtype=np.int32)
    segments[:5, :] = 1
    segments[5:, :] = 2

    feats = compute_segment_features(arr, segments, band_names, texture_band="dsi")

    assert set(feats["segment_id"]) == {1, 2}
    by_id = feats.set_index("segment_id")
    assert by_id.loc[1, "blue_mean"] == 10.0
    assert by_id.loc[2, "blue_mean"] == 50.0
    assert by_id.loc[1, "green_std"] == 0.0  # constant band -> zero spread
    for col in ("tex_contrast", "tex_homogeneity", "tex_energy", "tex_correlation"):
        assert col in feats.columns


def test_tiny_segments_are_skipped():
    band_names = ["blue"]
    H, W = 5, 5
    arr = np.ones((H, W, 1), dtype=np.float32)
    segments = np.zeros((H, W), dtype=np.int32)
    segments[0, 0] = 1     # only 1 pixel -> below the minimum segment size
    segments[1:, 1:] = 2   # a large-enough segment

    feats = compute_segment_features(arr, segments, band_names, texture_band="blue")
    assert set(feats["segment_id"]) == {2}


def test_nan_pixels_in_texture_band_dont_crash_or_warn():
    # The real Sentinel-2 exports have no explicit nodata value; cloud-masked
    # pixels are NaN directly. compute_segment_features must not choke on
    # this (previously: NaN got cast straight to uint8, an undefined
    # operation that only surfaced as a silent RuntimeWarning).
    band_names = ["blue", "dsi"]
    H, W = 10, 10
    arr = np.ones((H, W, 2), dtype=np.float32)
    arr[..., 1] = np.tile(np.arange(W, dtype=np.float32), (H, 1))  # dsi gradient
    arr[0:2, 0:2, 1] = np.nan  # a cloud-masked corner, outside our one segment

    segments = np.zeros((H, W), dtype=np.int32)
    segments[3:8, 3:8] = 1  # well clear of the NaN corner

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        feats = compute_segment_features(arr, segments, band_names, texture_band="dsi")

    assert set(feats["segment_id"]) == {1}
    assert not feats.isna().any().any()
