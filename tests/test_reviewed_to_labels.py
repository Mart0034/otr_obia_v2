"""Tests for reviewed_to_labels: turning QGIS review decisions into labels."""
import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, box

from reviewed_to_labels import precision_by_rank, split_reviews

X0, Y0 = 400000.0, 7400000.0


def _sites(reviews):
    geoms = [box(X0 + 100 * i, Y0, X0 + 100 * i + 20, Y0 + 20) for i in range(len(reviews))]
    return gpd.GeoDataFrame(
        {"rank": range(1, len(reviews) + 1), "site_id": [f"S{i:04d}" for i in range(1, len(reviews) + 1)],
         "mine_id": "tile_a", "review": reviews, "note": ""}, geometry=geoms, crs="EPSG:32719")


def test_split_by_review_value_with_case_and_blanks():
    dump, clean, unsure, summary = split_reviews(_sites(["dump", "Clean ", "unsure", "", None, "DUMP"]))
    assert len(dump) == 2 and len(clean) == 1 and len(unsure) == 1
    assert list(dump["site_id"]) == ["S0001", "S0006"]
    assert len(summary) == 4                      # blanks are not reviewed yet
    assert unsure.geometry.iloc[0].geom_type == "Point"


def test_unknown_review_value_is_an_error():
    with pytest.raises(ValueError, match="maybe"):
        split_reviews(_sites(["dump", "maybe"]))


def test_gaps_between_flagged_segments_are_closed():
    gap = MultiPolygon([box(X0, Y0, X0 + 10, Y0 + 10), box(X0 + 14, Y0, X0 + 24, Y0 + 10)])
    sites = gpd.GeoDataFrame({"rank": [1], "site_id": ["S0001"], "mine_id": ["t"], "review": ["dump"]},
                             geometry=[gap], crs="EPSG:32719")
    dump, *_ = split_reviews(sites, close_gaps_m=5)
    assert dump.geometry.iloc[0].geom_type == "Polygon"


def test_precision_by_rank_ignores_unsure_and_counts_dumps():
    _, _, _, summary = split_reviews(_sites(["dump", "dump", "clean", "unsure", "clean", "dump"]))
    table = precision_by_rank(summary, steps=(2, 5, 6)).set_index("bis_rang")
    assert table.loc[2, "trefferquote"] == pytest.approx(1.0)
    assert table.loc[5, "geprueft"] == 4 and table.loc[5, "trefferquote"] == pytest.approx(0.5)
    assert table.loc[6, "davon_dump"] == 3
