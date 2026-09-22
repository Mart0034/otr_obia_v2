"""Tests for summarize_mine_detection(): with only a handful of positive
mines, whether each known dump was found at all matters more than one
aggregate score across every segment."""
import pandas as pd

from otr_obia_pipeline import summarize_mine_detection


def _result(rows):
    return pd.DataFrame(rows, columns=["mine_id", "segment_id", "label", "dump_pred"])


def test_detected_and_missed_mines_are_reported_correctly():
    result = _result([
        # mine_1: has a true dump, and we predicted at least one segment positive
        ("mine_1", 1, 1, 1),
        ("mine_1", 2, 0, 0),
        # mine_2: has a true dump, but we predicted nothing positive -> missed
        ("mine_2", 1, 1, 0),
        ("mine_2", 2, 0, 0),
        # mine_3: no true dump at all -> not part of the detection summary
        ("mine_3", 1, 0, 1),
    ])

    summary = summarize_mine_detection(result)

    assert set(summary["mine_id"]) == {"mine_1", "mine_2"}
    by_id = summary.set_index("mine_id")
    assert by_id.loc["mine_1", "detected"] == True  # noqa: E712
    assert by_id.loc["mine_2", "detected"] == False  # noqa: E712


def test_no_dump_mines_returns_empty_summary():
    result = _result([
        ("mine_1", 1, 0, 0),
        ("mine_2", 1, 0, 1),
    ])

    summary = summarize_mine_detection(result)

    assert summary.empty
