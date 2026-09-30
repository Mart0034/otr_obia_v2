"""Tests for plan_sites(): turning a vector file of places (e.g. quarries) into
a tile plan - buffering, latitude filter, size caps and site ids. No network."""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from sites_from_vector import plan_sites


def _utm_box(cx, cy, half):
    return box(cx - half, cy - half, cx + half, cy + half)


def _gdf(geoms, **cols):
    return gpd.GeoDataFrame(pd.DataFrame(cols), geometry=geoms, crs="EPSG:32719")


def test_buffer_grows_the_tile_and_rounds_to_10m():
    g = _gdf([_utm_box(400000, 7400000, 300)])  # 600 m wide
    plan = plan_sites(g, buffer_m=250)
    assert plan.loc[0, "radius_m"] == 550  # (600 + 2*250) / 2
    assert plan.loc[0, "center_x"] == pytest.approx(400000)
    assert plan.loc[0, "status"] == "ok"


def test_tiny_feature_gets_the_minimum_tile():
    g = _gdf([Point(400000, 7400000).buffer(5)])
    plan = plan_sites(g, buffer_m=0, min_radius_m=500)
    assert plan.loc[0, "radius_m"] == 500


def test_latitude_filter_skips_outside_features():
    # y=7400000 in UTM 19S is about -23.9 deg
    g = _gdf([_utm_box(400000, 7400000, 100), _utm_box(400000, 6700000, 100)])
    plan = plan_sites(g, south=-25.0, north=-22.0)
    assert plan["status"].tolist()[0] == "ok"
    assert "südlich" in plan["status"].tolist()[1]


def test_oversized_and_too_small_features_are_skipped_with_a_reason():
    g = _gdf([_utm_box(400000, 7400000, 9000), _utm_box(410000, 7400000, 20)])
    plan = plan_sites(g, max_tile_m=10000, min_area_m2=10000)
    assert "zu groß" in plan.loc[0, "status"]
    assert "Mindestfläche" in plan.loc[1, "status"]


def test_site_ids_use_osm_id_when_present_and_stay_unique():
    g = _gdf([_utm_box(400000, 7400000, 100), _utm_box(401000, 7400000, 100)],
             **{"@id": ["way/123", "way/123"]})
    plan = plan_sites(g)
    assert plan["site_id"].is_unique
    assert all(s.startswith("q_way_123") for s in plan["site_id"])


def test_ids_fall_back_to_a_running_number():
    plan = plan_sites(_gdf([_utm_box(400000, 7400000, 100)]), prefix="quarry")
    assert plan.loc[0, "site_id"] == "quarry_00000"


def test_input_in_lat_lon_is_reprojected():
    g = gpd.GeoDataFrame(geometry=[box(-70.32, -23.81, -70.31, -23.80)], crs="EPSG:4326")
    plan = plan_sites(g)
    assert 300000 < plan.loc[0, "center_x"] < 450000 and 7.3e6 < plan.loc[0, "center_y"] < 7.5e6


def test_missing_crs_is_an_error():
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])
    with pytest.raises(ValueError, match="CRS"):
        plan_sites(g)


def test_merge_overlaps_combines_neighbours_into_one_tile():
    g = _gdf([_utm_box(400000, 7400000, 100), _utm_box(401000, 7400000, 100),
              _utm_box(420000, 7400000, 100)])
    plan = plan_sites(g, buffer_m=500, merge_overlaps=True)
    assert len(plan) == 2  # two neighbours merged, the far one stays alone
    merged = plan[plan["members"] != ""].iloc[0]
    assert merged["site_id"].startswith("q_cluster_")
    assert merged["radius_m"] >= 1000  # spans both neighbours plus buffers


def test_merge_overlaps_leaves_clusters_that_would_be_too_big():
    g = _gdf([_utm_box(400000 + i * 3000, 7400000, 100) for i in range(6)])
    plan = plan_sites(g, buffer_m=2000, merge_overlaps=True, max_tile_m=10000)
    assert len(plan) == 6 and (plan["members"] == "").all()


def test_touching_edges_do_not_count_as_overlap():
    g = _gdf([_utm_box(400000, 7400000, 100), _utm_box(401200, 7400000, 100)])
    plan = plan_sites(g, buffer_m=500, merge_overlaps=True)  # tiles just touch at x=400600/401200
    assert len(plan) == 2
