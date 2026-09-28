"""Tests for parsing Overpass "out geom" building elements into Shapely
polygons (fetch_osm_buildings.py). No network calls - see README for the
validation result against the real 138-mine dataset (0/36 real dumps
overlap an OSM building, but only ~1.6% of false positives do either -
OSM building coverage in this remote area is too sparse to fix the
buildings-look-like-dumps problem on its own)."""
from fetch_osm_buildings import _element_to_polygon


def _way(coords):
    return {"type": "way", "geometry": [{"lat": y, "lon": x} for x, y in coords]}


def test_way_becomes_a_closed_polygon():
    square = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    poly = _element_to_polygon(_way(square))
    assert poly is not None
    assert poly.is_valid
    assert poly.area == 1.0


def test_relation_uses_the_outer_member():
    rel = {
        "type": "relation",
        "members": [
            {"role": "inner", "geometry": [{"lat": 0.2, "lon": 0.2}]},
            {"role": "outer", "geometry": [
                {"lat": y, "lon": x} for x, y in [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
            ]},
        ],
    }
    poly = _element_to_polygon(rel)
    assert poly is not None
    assert poly.area == 4.0


def test_relation_without_outer_member_is_skipped():
    rel = {"type": "relation", "members": [{"role": "inner", "geometry": []}]}
    assert _element_to_polygon(rel) is None


def test_unclosed_way_is_skipped():
    open_way = _way([(0, 0), (1, 0), (1, 1)])  # doesn't return to start
    assert _element_to_polygon(open_way) is None


def test_too_few_points_is_skipped():
    tiny = _way([(0, 0), (1, 1)])
    assert _element_to_polygon(tiny) is None


def test_unknown_element_type_is_skipped():
    assert _element_to_polygon({"type": "node", "lat": 0, "lon": 0}) is None
