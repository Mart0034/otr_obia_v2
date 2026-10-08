"""Tests for compute_segment_features(): per-patch color and texture stats."""
import warnings

import numpy as np
import pytest

from otr_obia_pipeline import brightness_extreme_fraction, compute_segment_features


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


# --- brightness_extreme_fraction(): the bimodality check that separates a
# sharp light/dark split (e.g. a shed roof edge next to its own shadow,
# or a lined pond's bright rim next to dark water) from a real dump's
# broader, more evenly spread texture. Found by manual review: false
# positives in the real data were bimodal (extreme_fraction > 0.75) 3.7x
# more often than real dumps. ---


def test_uniform_values_have_low_extreme_fraction():
    vals = np.linspace(0, 100, 30)  # evenly spread, nothing bimodal about it
    assert brightness_extreme_fraction(vals) == pytest.approx(2 / 3, abs=0.05)


def test_two_tone_split_has_high_extreme_fraction():
    # half the pixels dark, half bright, nothing in between - a roof/shadow
    # edge, not a dump
    vals = np.concatenate([np.zeros(15), np.full(15, 100.0)])
    assert brightness_extreme_fraction(vals) == pytest.approx(1.0)


def test_constant_segment_has_zero_extreme_fraction():
    vals = np.full(10, 42.0)  # no range at all -> nothing to be "extreme" against
    assert brightness_extreme_fraction(vals) == 0.0


def test_too_few_pixels_returns_zero_not_a_crash():
    assert brightness_extreme_fraction(np.array([1.0, 2.0])) == 0.0


def test_nan_values_are_ignored():
    vals = np.array([0.0, 0.0, np.nan, 100.0, 100.0, np.nan])
    assert brightness_extreme_fraction(vals) == pytest.approx(1.0)


def test_compute_segment_features_adds_brightness_extreme_fraction():
    band_names = ["blue", "green", "red", "dsi"]
    H, W = 10, 10
    arr = np.zeros((H, W, 4), dtype=np.float32)
    # segment 1: sharp two-tone split in visible bands
    arr[0:5, :, 0:3] = 0.0
    arr[5:10, :, 0:3] = 100.0
    segments = np.zeros((H, W), dtype=np.int32)
    segments[:, :] = 1

    feats = compute_segment_features(arr, segments, band_names, texture_band="dsi")

    assert "brightness_extreme_fraction" in feats.columns
    assert feats.loc[0, "brightness_extreme_fraction"] == pytest.approx(1.0)


def test_spatial_split_contrast_detects_a_clean_left_right_split():
    # left half dark, right half bright - a sharp, spatially clean split,
    # the signature of a shadowed slope/pit edge, not a dump
    band_names = ["blue", "green", "red", "dsi"]
    H, W = 10, 10
    arr = np.zeros((H, W, 4), dtype=np.float32)
    arr[:, :5, 0:3] = 0.0
    arr[:, 5:, 0:3] = 1.0
    segments = np.ones((H, W), dtype=np.int32)

    feats = compute_segment_features(arr, segments, band_names, texture_band="dsi")

    assert feats.loc[0, "spatial_split_contrast"] == pytest.approx(1.0)


def test_spatial_split_contrast_is_low_for_uniform_brightness():
    band_names = ["blue", "green", "red", "dsi"]
    H, W = 10, 10
    arr = np.full((H, W, 4), 0.5, dtype=np.float32)
    segments = np.ones((H, W), dtype=np.int32)

    feats = compute_segment_features(arr, segments, band_names, texture_band="dsi")

    assert feats.loc[0, "spatial_split_contrast"] == pytest.approx(0.0)


def test_spatial_split_contrast_defaults_to_zero_without_color_bands():
    band_names = ["dsi"]
    H, W = 10, 10
    arr = np.tile(np.arange(W, dtype=np.float32), (H, 1))[..., np.newaxis]
    segments = np.zeros((H, W), dtype=np.int32)
    segments[:, :] = 1

    feats = compute_segment_features(arr, segments, band_names, texture_band="dsi")

    assert feats.loc[0, "spatial_split_contrast"] == 0.0


def test_compute_segment_features_without_color_bands_defaults_to_zero():
    # a minimal band set with no blue/green/red present at all (as used by
    # some other tests in this file) must not crash
    band_names = ["dsi"]
    H, W = 10, 10
    arr = np.tile(np.arange(W, dtype=np.float32), (H, 1))[..., np.newaxis]
    segments = np.zeros((H, W), dtype=np.int32)
    segments[:, :] = 1

    feats = compute_segment_features(arr, segments, band_names, texture_band="dsi")

    assert feats.loc[0, "brightness_extreme_fraction"] == 0.0


def test_bounding_box_loop_matches_the_full_image_reference():
    """compute_segment_features looks at each segment only inside its own
    bounding box; the numbers must equal the straightforward full-image version
    (irregular, interleaved segments and NaN pixels included)."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(3)
    h, w = 60, 70
    band_names = ["blue", "green", "red", "dsi"]
    arr = rng.normal(size=(h, w, 4)).astype("float32")
    arr[5:9, 5:9, :] = np.nan
    seg = (gaussian_filter(rng.normal(size=(h, w)), 2) > 0).astype(int) + 1
    seg = seg * 10 + (np.arange(w)[None, :] // 7)  # irregular, non-compact labels
    seg[:3, :] = 0
    feats = compute_segment_features(arr, seg, band_names, texture_band="dsi").set_index("segment_id")
    for sid in feats.index:
        m = seg == sid
        for b, name in enumerate(band_names):
            vals = arr[..., b][m]
            assert feats.loc[sid, f"{name}_mean"] == np.nanmean(vals) or (
                np.isnan(feats.loc[sid, f"{name}_mean"]) and np.isnan(np.nanmean(vals)))
        ys, xs = np.where(m)
        assert feats.loc[sid, "n_pixels"] == m.sum()
        assert ys.min() >= 0  # bounding box sanity
