"""Tests for the reference-picture feature (examples stored on disk, sent ahead of the new picture)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qgis_plugin"))
from otr_review import core  # noqa: E402
from otr_review.examples import DEFAULT_CAPTIONS, ExampleStore  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 200


def test_store_add_list_toggle_rename_delete(tmp_path):
    st = ExampleStore(str(tmp_path / "ex"))
    a = st.add(PNG, "dump", "tire rows", "S0001")
    b = st.add(PNG, "not_dump", "", "S0002")
    assert a != b
    ex = st.list()
    assert [e["kind"] for e in ex] == ["dump", "not_dump"] and ex[0]["caption"] == "tire rows"
    assert ex[1]["caption"] == DEFAULT_CAPTIONS["not_dump"]                       # empty caption -> default
    assert st.set_enabled(a, False) and not st.list()[0]["enabled"]
    assert st.set_caption(b, "dirt heaps on slope") and st.list()[1]["caption"] == "dirt heaps on slope"
    st.delete(a)
    assert [e["id"] for e in st.list()] == [b] and not os.path.exists(str(tmp_path / "ex" / f"{a}.png"))
    with pytest.raises(ValueError):
        st.add(PNG, "maybe")


def test_store_ignores_broken_entries_and_survives_reopen(tmp_path):
    st = ExampleStore(str(tmp_path))
    ok = st.add(PNG, "dump")
    (tmp_path / "junk.json").write_text("{not json")
    (tmp_path / "orphan.json").write_text('{"id": "orphan", "kind": "dump"}')      # json without a png
    assert [e["id"] for e in ExampleStore(str(tmp_path)).list()] == [ok]


def test_active_respects_enabled_max_and_keeps_a_stable_order(tmp_path):
    st = ExampleStore(str(tmp_path))
    ids = [st.add(PNG, "dump" if i % 2 else "not_dump", f"c{i}") for i in range(8)]
    st.set_enabled(ids[0], False)
    act = st.active(4)
    assert len(act) == 4 and [e["id"] for e in act] == sorted(e["id"] for e in act)
    assert {e["kind"] for e in act} == {"dump", "not_dump"} and all(e["b64"] for e in act)
    assert ids[0] not in [e["id"] for e in act]
    assert st.active(0) == [] and [e["id"] for e in st.active(4)] == [e["id"] for e in act]   # stable (cache!)


def test_pick_examples_balances_the_two_kinds_and_fills_up():
    ents = [{"id": f"{i:03d}", "kind": "dump"} for i in range(2)] + [{"id": f"{i:03d}", "kind": "not_dump"} for i in range(2, 10)]
    got = core.pick_examples(ents, 6)
    kinds = [e["kind"] for e in got]
    assert len(got) == 6 and kinds.count("dump") == 2 and kinds.count("not_dump") == 4     # only 2 dumps exist: fill with the other kind
    assert core.pick_examples(ents, 99) == sorted(ents, key=lambda e: e["id"])
    assert core.pick_examples([], 4) == [] and core.pick_examples(ents, 0) == []


def test_payload_without_examples_is_unchanged_and_with_examples_puts_them_first():
    plain = core.build_payload("TARGET")
    assert [b["type"] for b in plain["messages"][0]["content"]] == ["image", "text"]
    ex = [{"id": "1", "kind": "dump", "caption": "tire rows", "b64": "AAA"},
          {"id": "2", "kind": "not_dump", "caption": "", "b64": "BBB"}]
    p = core.build_payload("TARGET", context="Zone radius 100 m", examples=ex, cache=True)
    content = p["messages"][0]["content"]
    assert [b["type"] for b in content] == ["text", "text", "image", "text", "image", "text", "image", "text"]
    assert content[1]["text"] == "Reference example 1 - CONFIRMED TIRE DUMP: tire rows"
    assert content[3]["text"] == "Reference example 2 - CONFIRMED NOT A DUMP"
    assert content[2]["source"]["data"] == "AAA" and content[-2]["source"]["data"] == "TARGET"
    assert content[-1]["text"].endswith("Context: Zone radius 100 m")
    assert content[5]["text"].startswith("End of reference examples") and content[5]["cache_control"] == {"type": "ephemeral"}
    assert sum(1 for b in content if "cache_control" in b) == 1
    assert "cache_control" not in core.build_payload("T", examples=ex, cache=False)["messages"][0]["content"][5]


def test_example_settings_defaults_and_limits():
    d = core.DEFAULT_SETTINGS
    assert d["use_examples"] is True and d["max_examples"] == 6 and d["example_px"] == 768
    s = core.clean_settings({"max_examples": 99, "example_px": 10, "use_examples": False})
    assert s["max_examples"] == 12 and s["example_px"] == 256 and s["use_examples"] is False
