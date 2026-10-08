"""Tests for the multi-year brightness features (growth of a dark area)."""
import numpy as np
import pytest

from fetch_s2_multiyear import BAND_NAMES, multiyear_features

YEARS = list(range(2018, 2026))     # 8 years


def _stack(pixels):
    """pixels: list of 8-year brightness lists -> (8, 1, n_pixels)"""
    return np.array(pixels, dtype=np.float32).T[:, None, :]


def test_a_dump_appearing_mid_period_gets_negative_delta_and_late_onset():
    new_dump = [0.30, 0.30, 0.30, 0.05, 0.05, 0.05, 0.05, 0.05]
    out = multiyear_features(_stack([new_dump]), YEARS)[:, 0, 0]
    f = dict(zip(BAND_NAMES, out))
    assert f["s2my_bright_early"] == pytest.approx(0.30)
    assert f["s2my_bright_late"] == pytest.approx(0.05)
    assert f["s2my_bright_delta"] == pytest.approx(-0.25)
    assert f["s2my_bright_slope"] < -0.02
    assert f["s2my_dark_years_frac"] == pytest.approx(5 / 8)
    assert f["s2my_dark_onset"] == pytest.approx(3 / 7)


def test_stable_bright_ground_has_no_change_and_no_onset():
    out = multiyear_features(_stack([[0.30] * 8]), YEARS)[:, 0, 0]
    f = dict(zip(BAND_NAMES, out))
    assert f["s2my_bright_delta"] == pytest.approx(0.0)
    assert f["s2my_bright_slope"] == pytest.approx(0.0, abs=1e-6)
    assert f["s2my_dark_years_frac"] == 0.0 and np.isnan(f["s2my_dark_onset"])


def test_old_dark_area_is_dark_from_the_first_year_without_delta():
    out = multiyear_features(_stack([[0.05] * 8]), YEARS)[:, 0, 0]
    f = dict(zip(BAND_NAMES, out))
    assert f["s2my_dark_onset"] == 0.0 and f["s2my_dark_years_frac"] == 1.0
    assert f["s2my_bright_delta"] == pytest.approx(0.0)


def test_pixels_with_too_few_valid_years_stay_empty_and_gaps_are_tolerated():
    nan = np.nan
    too_few = [0.3, nan, nan, 0.05, nan, nan, 0.05, nan]                # 3 valid years < 4
    gaps = [0.30, nan, 0.30, nan, 0.05, 0.05, nan, 0.05]                # 5 valid years
    out = multiyear_features(_stack([too_few, gaps]), YEARS)
    assert np.isnan(out[:, 0, 0]).all()
    f = dict(zip(BAND_NAMES, out[:, 0, 1]))
    assert f["s2my_bright_early"] == pytest.approx(0.30) and f["s2my_bright_late"] == pytest.approx(0.05)
    assert f["s2my_dark_onset"] == pytest.approx((2022 - 2018) / 7)


def test_pipeline_loader_returns_nan_bands_when_the_file_is_missing_and_reads_it_otherwise(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from otr_obia_pipeline import S2MY_BAND_NAMES, load_s2_multiyear_for_mine

    assert S2MY_BAND_NAMES == BAND_NAMES                    # the two lists must stay in sync
    s2_path = str(tmp_path / "imagery" / "mine_001.tif")
    out_dir = tmp_path / "my"
    out_dir.mkdir()
    (tmp_path / "imagery").mkdir()
    missing = load_s2_multiyear_for_mine(s2_path, str(out_dir), (4, 5))
    assert missing.shape == (4, 5, 6) and np.isnan(missing).all()
    data = np.arange(6 * 4 * 5, dtype="float32").reshape(6, 4, 5)
    with rasterio.open(out_dir / "mine_001.tif", "w", driver="GTiff", height=4, width=5, count=6,
                       dtype="float32", crs="EPSG:32719", transform=from_origin(0, 40, 10, 10)) as dst:
        dst.write(data)
    got = load_s2_multiyear_for_mine(s2_path, str(out_dir), (4, 5))
    assert got.shape == (4, 5, 6) and got[2, 3, 4] == data[4, 2, 3]


def test_a_subset_of_the_multiyear_features_can_be_selected(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from otr_obia_pipeline import load_s2_multiyear_for_mine, select_s2my_names

    assert select_s2my_names({}) == BAND_NAMES
    assert select_s2my_names({"s2my_bands": "s2my_bright_late, s2my_bright_early"}) == ["s2my_bright_early", "s2my_bright_late"]
    with pytest.raises(ValueError):
        select_s2my_names({"s2my_bands": "s2my_nonsense"})
    (tmp_path / "imagery").mkdir()
    out_dir = tmp_path / "my"
    out_dir.mkdir()
    data = np.arange(6 * 3 * 3, dtype="float32").reshape(6, 3, 3)
    with rasterio.open(out_dir / "m.tif", "w", driver="GTiff", height=3, width=3, count=6, dtype="float32",
                       crs="EPSG:32719", transform=from_origin(0, 30, 10, 10)) as dst:
        dst.write(data)
    got = load_s2_multiyear_for_mine(str(tmp_path / "imagery" / "m.tif"), str(out_dir), (3, 3),
                                     ["s2my_bright_late", "s2my_dark_onset"])
    assert got.shape == (3, 3, 2)
    assert got[1, 2, 0] == data[1, 1, 2] and got[1, 2, 1] == data[5, 1, 2]       # late = band index 1, onset = 5
