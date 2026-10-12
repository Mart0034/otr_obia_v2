"""Tests for eval_near_miss: distance of known dumps to the model's flagged areas."""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from eval_near_miss import nearest_distances, summarize

X0, Y0 = 400000.0, 7400000.0


def _gdf(items):
    return gpd.GeoDataFrame({"mine_id": [i[0] for i in items]},
                            geometry=[box(X0 + i[1], Y0 + i[2], X0 + i[3], Y0 + i[4]) for i in items], crs="EPSG:32719")


def test_hit_near_miss_and_unflagged_mine():
    labels = _gdf([("m1", 0, 0, 20, 20),        # overlapped by a flag -> 0
                   ("m1", 300, 0, 320, 20),     # flag 100 m to the east -> 100
                   ("m2", 0, 0, 20, 20)])       # mine m2 has nothing flagged -> NaN
    flagged = _gdf([("m1", 10, 10, 30, 30), ("m1", 420, 0, 440, 20)])
    d = nearest_distances(labels, flagged, same_mine=True)
    assert d[0] == 0 and d[1] == pytest.approx(100) and pd.isna(d[2])


def test_same_mine_ignores_flags_of_other_mines_unless_disabled():
    labels = _gdf([("m1", 0, 0, 20, 20)])
    flagged = _gdf([("m2", 50, 0, 70, 20)])
    assert pd.isna(nearest_distances(labels, flagged, same_mine=True)[0])
    assert nearest_distances(labels, flagged, same_mine=False)[0] == pytest.approx(30)


def test_summary_shares_and_size_classes():
    labels = _gdf([("m1", 0, 0, 10, 10), ("m1", 100, 0, 110, 10), ("m1", 1000, 0, 1100, 100), ("m3", 0, 0, 5, 5)])
    dist = pd.Series([0.0, 200.0, 0.0, float("nan")])
    tab = summarize(labels, dist).set_index("gruppe")
    assert tab.loc["alle", "n"] == 4 and tab.loc["alle", "ohne_markierung"] == 1
    assert tab.loc["alle", "<= 0 m"] == 0.5 and tab.loc["alle", "<= 250 m"] == 0.75
    assert tab.loc[">2000 m2", "n"] == 1 and tab.loc[">2000 m2", "<= 0 m"] == 1.0
    assert tab.loc["<500 m2", "n"] == 3


def test_no_flags_at_all_gives_nan_everywhere():
    labels = _gdf([("m1", 0, 0, 10, 10)])
    assert nearest_distances(labels, labels.iloc[0:0]).isna().all()
