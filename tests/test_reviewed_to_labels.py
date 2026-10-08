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
    assert unsure.geometry.iloc[0].geom_type == "Polygon"   # ignore zone keeps the whole outline
    assert list(dump["site_id"]) == ["S0001", "S0006"]
    assert len(summary) == 4                      # blanks are not reviewed yet


def test_unknown_review_value_is_an_error():
    with pytest.raises(ValueError, match="maybe"):
        split_reviews(_sites(["dump", "maybe"]))


def test_partial_sites_are_ignored_not_labeled_and_count_in_the_lenient_rate():
    dump, clean, unsure, summary = split_reviews(_sites(["dump", "partial", "clean", "partial"]))
    assert len(dump) == 1 and len(clean) == 1 and len(unsure) == 2
    table = precision_by_rank(summary, steps=(4,)).iloc[0]
    assert table["trefferquote"] == pytest.approx(0.25)
    assert table["trefferquote_mit_partial"] == pytest.approx(0.75)


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


def _drawn(*rows):
    """rows: (kind, mine_id, box) in the same grid as _sites()."""
    return gpd.GeoDataFrame({"kind": [r[0] for r in rows], "mine_id": [r[1] for r in rows],
                             "note": ""}, geometry=[r[2] for r in rows], crs="EPSG:32719")


def test_drawn_dump_replaces_the_whole_site_outline_as_label():
    sites = _sites(["", ""])                       # two 20 x 20 m sites, not reviewed
    drawn = _drawn(("dump", "", box(X0 + 2, Y0 + 2, X0 + 8, Y0 + 8)))      # small piece of site 1
    dump, clean, unsure, summary = split_reviews(sites, drawn=drawn)
    assert len(dump) == 1 and dump.geometry.iloc[0].area == pytest.approx(36)    # the drawn shape, not 400
    assert dump.iloc[0]["mine_id"] == "tile_a"                                    # filled from the site
    row = summary[summary["site_id"] == "S0001"].iloc[0]
    assert row["review"] == "partial"                                             # 36 of 400 m2: not most of the site
    assert "S0002" not in set(summary["site_id"])                                 # untouched site stays unreviewed


def test_drawn_dump_that_fills_the_site_counts_as_dump_and_clean_drawn_as_clean():
    sites = _sites(["", ""])
    drawn = _drawn(("dump", "tile_a", box(X0, Y0, X0 + 20, Y0 + 15)),
                   ("clean", "tile_a", box(X0 + 100, Y0, X0 + 120, Y0 + 20)))
    dump, clean, _, summary = split_reviews(sites, drawn=drawn)
    reviews = dict(zip(summary["site_id"], summary["review"]))
    assert reviews == {"S0001": "dump", "S0002": "clean"}
    assert len(dump) == 1 and len(clean) == 1


def test_own_review_is_kept_when_drawn_polygons_exist_but_outline_is_not_used():
    sites = _sites(["clean"])
    drawn = _drawn(("dump", "tile_a", box(X0 + 1, Y0 + 1, X0 + 5, Y0 + 5)))
    dump, clean, _, summary = split_reviews(sites, drawn=drawn)
    assert summary.iloc[0]["review"] == "clean" and len(clean) == 0 and len(dump) == 1


def test_drawn_polygon_far_from_any_site_needs_a_mine_id():
    far = _drawn(("dump", "", box(X0 + 5000, Y0, X0 + 5010, Y0 + 10)))
    with pytest.raises(ValueError, match="mine_id"):
        split_reviews(_sites(["dump"]), drawn=far)
    ok = _drawn(("dump", "mine_007", box(X0 + 5000, Y0, X0 + 5010, Y0 + 10)))
    dump, *_ = split_reviews(_sites(["dump"]), drawn=ok)
    assert set(dump["mine_id"]) == {"tile_a", "mine_007"}


def test_unknown_drawn_kind_is_an_error():
    with pytest.raises(ValueError, match="kind"):
        split_reviews(_sites(["dump"]), drawn=_drawn(("dumb", "tile_a", box(X0, Y0, X0 + 5, Y0 + 5))))
