"""Tests for segment_mine()'s size-aware segmentation. This exists because
a fixed n_segments_per_mine produces wildly different segment sizes across
mines whose images range from ~72k to ~5.9M pixels, making most real dump
polygons (often just a few pixels) too small to ever reach a meaningful
overlap ratio with any segment."""
import numpy as np

from otr_obia_pipeline import segment_mine


def _fake_arr(size, n_bands=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((size, size, n_bands), dtype=np.float32)


def test_target_segment_px_scales_with_image_size():
    small = segment_mine(_fake_arr(20, seed=1), n_segments=999, compactness=8,
                          target_segment_px=10)
    large = segment_mine(_fake_arr(80, seed=2), n_segments=999, compactness=8,
                          target_segment_px=10)

    # 20x20=400px / 10 = ~40 segments; 80x80=6400px / 10 = ~640 segments
    small_ids = len(np.unique(small[small > 0]))
    large_ids = len(np.unique(large[large > 0]))
    assert large_ids > small_ids * 5


def test_max_segments_caps_large_images():
    capped = segment_mine(_fake_arr(80, seed=3), n_segments=999, compactness=8,
                           target_segment_px=1, max_segments=50)
    n_ids = len(np.unique(capped[capped > 0]))
    # without the cap, target_segment_px=1 would ask SLIC for ~6400 segments
    assert n_ids <= 60  # SLIC doesn't guarantee the exact count, allow slack


def test_target_segment_px_none_uses_fixed_n_segments():
    segments = segment_mine(_fake_arr(20, seed=4), n_segments=5, compactness=8,
                             target_segment_px=None)
    n_ids = len(np.unique(segments[segments > 0]))
    assert n_ids <= 8  # close to the requested fixed count, not size-derived
