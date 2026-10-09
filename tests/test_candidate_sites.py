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


def _two_hills():
    """A 40 x 6 strip of squares whose scores peak at two places far apart, so they
    form one connected blob that should split into two cores."""
    import math
    cells = []
    for c in range(40):
        for r in range(6):
            p = 0.3 + 0.65 * max(math.exp(-((c - 5) ** 2) / 8), math.exp(-((c - 34) ** 2) / 8))
            cells.append((c, r, p))
    return _seg(cells)


def test_big_blob_splits_into_cores_under_the_size_cap():
    seg = _two_hills()
    one = build_candidate_sites(seg, threshold=0.0, max_site_m2=0)
    assert len(one) == 1 and one.loc[0, "area_m2"] == pytest.approx(24000)
    cores = build_candidate_sites(seg, threshold=0.0, max_site_m2=3000, join_dist_m=2)
    assert len(cores) >= 2
    assert (cores["area_m2"] <= 3000 + 1e-6).all()
    assert cores["mean_proba"].min() > one.loc[0, "mean_proba"]      # cores beat the blob average


def test_exclude_known_drops_sites_near_labels_and_reranks():
    cells = [(0, 0, 0.9), (50, 0, 0.8)]
    labels = gpd.GeoDataFrame(geometry=[box(X0 - 5, Y0 - 5, X0 + 15, Y0 + 15)], crs="EPSG:32719")
    sites = build_candidate_sites(_seg(cells), threshold=0.5, labels=labels, exclude_known=True)
    assert len(sites) == 1 and sites.loc[0, "rank"] == 1 and not sites.loc[0, "near_known_dump"]


def test_urban_filter_drops_built_up_sites_and_needs_osm_columns():
    cells = [(0, 0, 0.9), (50, 0, 0.8)]
    seg = _seg(cells)
    seg["is_building"] = [True, False]
    kept = build_candidate_sites(seg, threshold=0.5, max_urban_frac=0.2)
    assert len(kept) == 1 and kept.loc[0, "urban_frac"] == 0.0
    both = build_candidate_sites(seg, threshold=0.5)
    assert sorted(both["urban_frac"]) == [0.0, 1.0]
    # without OSM columns the filter is a no-op (with a warning), not a crash
    assert len(build_candidate_sites(_seg(cells), threshold=0.5, max_urban_frac=0.2)) == 2


def test_gpkg_has_an_empty_polygon_layer_for_hand_drawn_shapes(tmp_path):
    import pyogrio
    from candidate_sites import write_sites_gpkg

    path = str(tmp_path / "sites.gpkg")
    write_sites_gpkg(build_candidate_sites(_seg([(0, 0, 0.9)]), threshold=0.5), path)
    layers = dict(pyogrio.list_layers(path))
    assert layers["candidate_sites"] == "Polygon" and layers["drawn_polygons"] == "Polygon"
    drawn = gpd.read_file(path, layer="drawn_polygons")
    assert drawn.empty and {"kind", "mine_id", "note"} <= set(drawn.columns)
    assert drawn.crs.to_epsg() == 32719


def test_rank_by_peak_score_puts_the_highest_peak_first():
    cells = [(c, 0, 0.60) for c in range(10)] + [(100, 0, 0.99)]    # big mediocre blob, small sharp peak
    by_score = build_candidate_sites(_seg(cells), threshold=0.5, rank_by="score")
    by_peak = build_candidate_sites(_seg(cells), threshold=0.5, rank_by="max_proba")
    assert by_score.loc[0, "n_segments"] == 10 and by_peak.loc[0, "n_segments"] == 1
    with pytest.raises(ValueError):
        build_candidate_sites(_seg(cells), threshold=0.5, rank_by="nonsense")


def test_exclude_areas_hides_already_reviewed_sites():
    cells = [(0, 0, 0.9), (50, 0, 0.8), (100, 0, 0.7)]
    done = gpd.GeoDataFrame(geometry=[box(X0 + 495, Y0 - 5, X0 + 515, Y0 + 15)], crs="EPSG:32719")   # covers cell 50
    sites = build_candidate_sites(_seg(cells), threshold=0.5, exclude_areas=done)
    assert len(sites) == 2 and sorted(sites["max_proba"]) == [0.7, 0.9] and list(sites["rank"]) == [1, 2]


def test_preselect_reader_loads_only_the_best_segments_per_mine_and_gives_the_same_sites(tmp_path):
    from candidate_sites import load_top_segments

    cells = [(c, 0, 0.1 + 0.005 * c) for c in range(100)]                 # best segments at the right end
    seg = pd.concat([_seg(cells, "m1"), _seg(cells[:50], "m2")], ignore_index=True)
    seg["segment_id"] = range(len(seg))
    path = str(tmp_path / "seg.gpkg")
    seg.to_file(path, driver="GPKG")
    part = load_top_segments(path, area_pct=5.0, fraction_margin=1.0)
    assert len(part) == 5 + 3                                             # 5% of 100 + ceil(5% of 50)
    assert part[part.mine_id == "m1"]["dump_proba"].min() == pytest.approx(seg[seg.mine_id == "m1"]["dump_proba"].nlargest(5).min())
    full = build_candidate_sites(seg, area_pct=5.0)
    fast = build_candidate_sites(part, area_pct=100.0)
    assert sorted(full["max_proba"]) == pytest.approx(sorted(fast["max_proba"]))


def test_sample_every_keeps_original_ranks_and_skips_reviewed_ones():
    import numpy as np
    from shapely.geometry import box
    import geopandas as gpd
    from candidate_sites import build_candidate_sites
    n = 60
    seg = gpd.GeoDataFrame(
        {"mine_id": ["m"] * n, "dump_proba": np.linspace(0.99, 0.5, n)},
        geometry=[box(i * 1000, 0, i * 1000 + 50, 50) for i in range(n)], crs="EPSG:32719")
    full = build_candidate_sites(seg, threshold=0.0, area_pct=100.0, rank_by="max_proba")
    samp = build_candidate_sites(seg, threshold=0.0, area_pct=100.0, rank_by="max_proba",
                                 sample_every=5, sample_skip=10)
    assert list(samp["rank"]) == [15, 20, 25, 30, 35, 40, 45, 50, 55, 60][:len(samp)]
    assert list(samp["site_id"]) == [f"S{r:04d}" for r in samp["rank"]]
    assert set(samp["site_id"]) <= set(full["site_id"])
