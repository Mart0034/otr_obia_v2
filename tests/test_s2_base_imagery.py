"""Tests for building a Sentinel-2 base-imagery tile from scratch, for a new
site that has no existing data/imagery/*.tif from the earlier feasibility
study (see fetch_s2_base_imagery.py). Covers the target-grid math and the
NDVI/NDWI/BSI/DSI formulas, which were reverse-engineered from the real
data/imagery tiles (exact match, error < 1e-7). DSI turned out to be a
linear combination, -(red + nir + swir1) / 3, not a ratio."""
import numpy as np
import pytest

from fetch_s2_base_imagery import compute_dsi, build_grid, compute_indices


def test_build_grid_covers_the_requested_square():
    transform, width, height = build_grid(365950.1, 7367190.3, 1500, "EPSG:32719")

    assert width == 300 and height == 300  # 3000m / 10m pixels
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    assert left == pytest.approx(365950.1 - 1500)
    assert top == pytest.approx(7367190.3 + 1500)
    assert right == pytest.approx(365950.1 + 1500)
    assert bottom == pytest.approx(7367190.3 - 1500)


def test_ndvi_formula():
    nir, red = np.array([0.4]), np.array([0.1])
    ndvi, _, _ = compute_indices(
        np.zeros(1), np.zeros(1), red, nir, np.ones(1), np.ones(1)
    )
    assert ndvi[0] == pytest.approx((0.4 - 0.1) / (0.4 + 0.1))


def test_ndwi_uses_green_and_swir1_not_the_standard_green_nir_formula():
    green, swir1 = np.array([0.3]), np.array([0.2])
    _, ndwi, _ = compute_indices(
        np.zeros(1), green, np.zeros(1), np.full(1, 99.0), swir1, np.ones(1)
    )
    # nir is deliberately set way off (99.0) to prove it's NOT in the formula
    assert ndwi[0] == pytest.approx((0.3 - 0.2) / (0.3 + 0.2))


def test_bsi_formula():
    blue, red, nir, swir1 = np.array([0.1]), np.array([0.2]), np.array([0.3]), np.array([0.25])
    _, _, bsi = compute_indices(blue, np.zeros(1), red, nir, swir1, np.ones(1))
    expected = ((0.25 + 0.2) - (0.3 + 0.1)) / ((0.25 + 0.2) + (0.3 + 0.1))
    assert bsi[0] == pytest.approx(expected)


def test_dsi_is_the_negative_mean_of_red_nir_swir1():
    # Recovered by regressing the real tiles' DSI band on the raw bands
    # (R^2 = 1.000, max error 4e-8).
    assert compute_dsi(0.2, 0.3, 0.4) == pytest.approx(-0.3)
