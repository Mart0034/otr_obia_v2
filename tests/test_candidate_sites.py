"""Tests for candidate_sites: merging flagged segments into ranked sites."""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from candidate_sites import build_candidate_sites

X0, Y0 = 400000.0, 7400000.0  # UTM 19S, northern Chile


def _seg(cells, mine="mine_001"):
    """cells: list of (col, row, proba); 10 m squares on a grid."""
    return gpd.GeoDataFrame(
        {"mine_id": mine, "dump_proba": [c[2] for c in cells]},
        geometry=[box(X0 + c * 10, Y0 + r * 10, X0 + c * 10 + 10, Y0 + r * 10 + 10) for c, r, _ in cells],
        crs="EPSG:32719")


def test_adjacent_segments_merge_and_far_ones_stay_apart():
    big = [(c, 0, 0.8) for c in range(5)]            # 5 touching squares
    small = [(40, 0, 0.9)]                           # 300 m away
    sites = build_candidate_sites(_seg(big + small), threshold=0.5)
    assert len(sites) == 2
    assert sites.loc[0, "n_segments"] == 5 and sites.loc[0, "area_m2"] == pytest.approx(500)
    assert sites.loc[1, "n_segments"] == 1


def test_low_scoring_segments_are_ignored_and_rank_follows_area_times_score():
    cells = [(0, 0, 0.55), (1, 0, 0.55), (30, 0, 0.99), (60, 0, 0.2)]
    sites = build_candidate_sites(_seg(cells), threshold=0.5)
    assert len(sites) == 2                            # the 0.2 one is gone
    assert sites["rank"].tolist() == [1, 2]
    assert sites.loc[0, "area_m2"] == pytest.approx(200)   # 2 squares x 0.55 beats 1 x 0.99
    assert sites["score"].is_monotonic_decreasing


def test_gap_within_join_distance_connects_sites():
    cells = [(0, 0, 0.8), (2, 0, 0.8)]               # 10 m gap
    assert len(build_candidate_sites(_seg(cells), threshold=0.5, join_dist_m=20)) == 1
    assert len(build_candidate_sites(_seg(cells), threshold=0.5, join_dist_m=5)) == 2


def test_sites_never_merge_across_mines():
    a, b = _seg([(0, 0, 0.8)], "mine_001"), _seg([(1, 0, 0.8)], "mine_002")
    sites = build_candidate_sites(pd.concat([a, b], ignore_index=True), threshold=0.5, join_dist_m=20)
    assert len(sites) == 2


def test_maps_link_and_coordinates():
    sites = build_candidate_sites(_seg([(0, 0, 0.9)]))
    row = sites.iloc[0]
    assert -25 < row["lat"] < -23 and -71 < row["lon"] < -69
    assert row["maps_url"] == f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}"
    assert row["review"] == "" and row["site_id"] == "S0001"


def test_known_dump_flag_and_empty_result():
    seg = _seg([(0, 0, 0.9), (200, 0, 0.9)])
    labels = gpd.GeoDataFrame(geometry=[box(X0 - 5, Y0 - 5, X0 + 15, Y0 + 15)], crs="EPSG:32719")
    sites = build_candidate_sites(seg, threshold=0.5, labels=labels)
    assert sites.sort_values("rank")["near_known_dump"].sum() == 1
    assert build_candidate_sites(seg, threshold=0.99).empty


def test_size_class_and_info_fractions():
    cells = [(c, r, 0.8) for c in range(20) for r in range(15)]   # 300 squares = 30,000 m2
    seg = _seg(cells)
    seg["is_building"] = False
    seg.loc[seg.index[:30], "is_building"] = True                 # 10% of the area
    sites = build_candidate_sites(seg, threshold=0.5)
    assert sites.loc[0, "size_class"] == "large"
    assert sites.loc[0, "frac_is_building"] == pytest.approx(0.1)


def test_geographic_crs_is_rejected():
    seg = _seg([(0, 0, 0.9)]).to_crs(4326)
    with pytest.raises(ValueError):
        build_candidate_sites(seg)


def test_default_flags_the_top_share_of_each_mine_area():
    cells = [(c, 0, 0.1 + 0.001 * c) for c in range(100)]   # 100 squares, best at the right end
    sites = build_candidate_sites(_seg(cells), area_pct=5.0)   # 5% of 100 squares = 5 squares
    assert sites["n_segments"].sum() == 5
    assert sites["max_proba"].max() == pytest.approx(0.199)
    # each mine is judged on its own area
    two = pd.concat([_seg(cells, "mine_001"), _seg(cells[:20], "mine_002")], ignore_index=True)
    assert build_candidate_sites(two, area_pct=10.0)["n_segments"].sum() == 10 + 2
