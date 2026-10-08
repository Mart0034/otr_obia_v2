"""Tests for the order logic behind the QGIS review panel (no QGIS needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qgis"))
from review_queue import ReviewQueue  # noqa: E402


def _items():
    return [(10, 3, ""), (11, 1, ""), (12, 2, "clean"), (13, 4, "")]   # (fid, rank, review)


def test_unreviewed_sites_come_in_rank_order_and_reviewed_ones_are_skipped():
    q = ReviewQueue(_items())
    assert q.current == 11 and q.done == 1 and q.total == 4         # rank 1 first; fid 12 already done
    assert q.decide("dump") == 11
    assert q.current == 10                                            # rank 3 (rank 2 was already reviewed)
    q.decide("clean")
    assert q.current == 13
    q.decide("unsure")
    assert q.current is None and q.done == 4
    assert q.counts() == {"dump": 1, "clean": 2, "unsure": 1}


def test_skip_moves_to_the_end_and_comes_back():
    q = ReviewQueue(_items())
    q.skip()
    assert q.current == 10
    q.decide("dump")
    q.decide("dump")
    assert q.current == 11                                             # the skipped one returns last
    q.decide("clean")
    assert q.current is None


def test_back_undoes_the_last_decision_and_restores_the_old_value():
    q = ReviewQueue(_items())
    q.decide("dump")
    fid, old = q.back()
    assert (fid, old) == (11, "") and q.current == 11 and q.reviews[11] == ""
    assert q.back() is None                                            # nothing left to undo
    q.decide("clean")
    q.decide("partial")
    assert q.back() == (10, "")
    assert q.current == 10 and q.counts() == {"clean": 2}


def test_draw_flow_marks_without_advancing_until_next():
    q = ReviewQueue(_items())
    assert q.mark("partial") == 11 and q.current == 11                # still on the site while drawing
    q.advance()
    assert q.current == 10 and q.reviews[11] == "partial"


def test_empty_queue_is_harmless():
    q = ReviewQueue([(1, 1, "dump")])
    assert q.current is None and q.decide("dump") is None and q.skip() is None and q.back() is None
