"""Tests for score_points: reading model scores at known coordinates."""
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from score_points import score_points

X0, Y0 = 400000.0, 7400000.0


def _segments():
    cells = [(0, 0.9), (1, 0.5), (2, 0.2), (3, 0.1)]       # four 10 m squares in a row
    return gpd.GeoDataFrame(
        {"mine_id": "t", "dump_proba": [p for _, p in cells]},
        geometry=[box(X0 + c * 10, Y0, X0 + c * 10 + 10, Y0 + 10) for c, _ in cells], crs="EPSG:32719")


def test_score_share_and_neighbourhood():
    pts = gpd.GeoDataFrame({"size": ["big"]}, geometry=[Point(X0 + 15, Y0 + 5)], crs="EPSG:32719")  # in cell 1
    out = score_points(_segments(), pts, radii=(4, 12))
    row = out.iloc[0]
    assert row["seg_proba"] == pytest.approx(0.5)
    assert row["top_share_pct"] == pytest.approx(50.0)          # cells with >= 0.5: 2 of 4
    assert row["max_4m"] == pytest.approx(0.5)
    assert row["max_12m"] == pytest.approx(0.9)                 # reaches the neighbour at 0.9
    assert row["size"] == "big" and row["tile"] == "t"
    assert row["maps_url"].startswith("https://www.google.com/maps/search/?api=1&query=-")


def test_point_outside_any_segment_is_nan():
    pts = gpd.GeoDataFrame(geometry=[Point(X0 + 500, Y0)], crs="EPSG:32719")
    out = score_points(_segments(), pts)
    assert out["seg_proba"].isna().all() and out["tile"].isna().all()
