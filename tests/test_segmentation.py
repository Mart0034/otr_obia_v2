"""Tests for segment_mine()'s block-wise mode: with very many segments SLIC's
masked k-means initialisation scales as pixels x segments (hours for a big
mine), so big requests are split into blocks that each get their share."""
import numpy as np

import otr_obia_pipeline as pipe


def _image(h=120, w=120, bands=3, seed=0):
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack(
        [np.sin(xx / (4.0 + b)) + np.cos(yy / (5.0 + 2 * b)) + 0.05 * b for b in range(bands)], axis=-1
    ).astype(np.float32)
    arr[:20, :20, :] = np.nan  # a no-data corner
    return arr


def test_small_requests_use_the_plain_single_pass(monkeypatch):
    called = []
    real = pipe.slic
    monkeypatch.setattr(pipe, "slic", lambda *a, **k: called.append(a[0].shape) or real(*a, **k))
    pipe.segment_mine(_image(), n_segments=30, compactness=8)
    assert len(called) == 1 and called[0][:2] == (120, 120)


def test_big_requests_are_segmented_in_blocks_with_unique_labels(monkeypatch):
    monkeypatch.setattr(pipe, "SLIC_BLOCK_MIN_SEGMENTS", 50)
    monkeypatch.setattr(pipe, "SLIC_BLOCK_PX", 40)
    arr = _image()
    seg = pipe.segment_mine(arr, n_segments=200, compactness=8)
    valid = np.all(np.isfinite(arr), axis=-1)
    assert seg.shape == (120, 120)
    assert (seg[valid] > 0).all()        # every valid pixel belongs to a segment
    assert (seg[~valid] == 0).all()      # no-data stays unlabelled
    labels = np.unique(seg[valid])
    assert len(labels) > 100             # roughly the requested 200 segments, not 9 blocks
    # labels are unique over the whole image: each label forms one connected group inside one block
    for lab in labels[:20]:
        ys, xs = np.where(seg == lab)
        assert ys.max() // 40 == ys.min() // 40 and xs.max() // 40 == xs.min() // 40


def test_target_segment_px_still_controls_the_segment_count_in_block_mode(monkeypatch):
    monkeypatch.setattr(pipe, "SLIC_BLOCK_MIN_SEGMENTS", 50)
    monkeypatch.setattr(pipe, "SLIC_BLOCK_PX", 40)
    arr = _image()
    n_valid = int(np.all(np.isfinite(arr), axis=-1).sum())
    seg = pipe.segment_mine(arr, n_segments=10, compactness=8, target_segment_px=40, max_segments=10**6)
    n = len(np.unique(seg[seg > 0]))
    assert 0.6 * n_valid / 40 < n < 1.5 * n_valid / 40
