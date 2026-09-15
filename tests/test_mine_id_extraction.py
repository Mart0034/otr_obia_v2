"""Tests for extract_mine_id(): matching a mine's ID from its image filename
to the digitized-labels file is where the text-vs-number bug lived."""
from otr_obia_pipeline import extract_mine_id


def test_extracts_mine_prefixed_id():
    assert extract_mine_id("data/imagery/mine_12.tif") == "mine_12"


def test_extracts_plain_number_id():
    assert extract_mine_id("data/imagery/7.tif") == "7"


def test_falls_back_to_full_filename_and_logs_a_warning(caplog):
    with caplog.at_level("WARNING"):
        mine_id = extract_mine_id("data/imagery/site-without-digits.tif")
    assert mine_id == "site-without-digits"
    assert "Konnte keine mine_id" in caplog.text
