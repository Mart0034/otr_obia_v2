"""Tests for the plugin core (settings, API request/response, auto-clean rules, queue) - no QGIS, no network."""
import io
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qgis_plugin"))
from otr_review import core as ai  # noqa: E402


def _resp(text):
    return {"content": [{"type": "text", "text": text}]}


def test_payload_has_image_context_and_model():
    p = ai.build_payload("QUJD", "image/png", " Sentinel-2 10 m ", "m1")
    assert p["model"] == "m1" and p["system"] == ai.SYSTEM_PROMPT
    img, txt = p["messages"][0]["content"]
    assert img["source"] == {"type": "base64", "media_type": "image/png", "data": "QUJD"}
    assert "Sentinel-2 10 m" in txt["text"]
    assert "Context" not in ai.build_payload("x")["messages"][0]["content"][1]["text"]


def test_parse_plain_and_fenced_json():
    r = ai.parse_reply(_resp('```json\n{"label": "Dump", "confidence": 0.8, '
                             '"dump_location": "inside", "reason": "dark piles"}\n```'))
    r.pop("usage")
    assert r == {"ok": True, "stop_reason": "", "tokens_in": 0, "tokens_out": 0, "label": "dump", "confidence": 0.8,
                 "dump_probability": 0.8, "dump_location": "inside", "reason": "dark piles"}


def test_parse_garbage_and_bad_values_fall_back_to_unsure():
    assert ai.parse_reply(_resp("no idea"))["label"] == "unsure"
    r = ai.parse_reply(_resp('{"label": "spaceship", "confidence": 7}'))
    assert r["label"] == "unsure" and r["confidence"] == 1.0
    assert ai.parse_reply(_resp('{"label": "clean_road", "confidence": "x"}'))["confidence"] == 0.0


def test_call_api_sends_headers_and_returns_json():
    seen = {}

    def opener(req, timeout):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps(_resp("ok")).encode())

    out = ai.call_api({"a": 1}, "sk-test", opener=opener)
    assert out == _resp("ok") and seen["body"] == {"a": 1}
    assert seen["headers"]["x-api-key"] == "sk-test" and seen["headers"]["anthropic-version"] == ai.API_VERSION


def test_call_api_turns_http_errors_into_readable_messages():
    def opener(req, timeout):
        raise urllib.error.HTTPError(ai.API_URL, 401, "Unauthorized", {},
                                     io.BytesIO(b'{"error": {"message": "invalid x-api-key"}}'))
    with pytest.raises(RuntimeError, match="401.*invalid x-api-key"):
        ai.call_api({}, "bad", opener=opener)


def test_log_row_writes_header_once(tmp_path):
    p = str(tmp_path / "log.csv")
    ai.log_row(p, {"a": 1, "b": 2})
    ai.log_row(p, {"a": 3, "b": 4})
    assert open(p).read().splitlines() == ["a,b", "1,2", "3,4"]


def test_padded_box_adds_radius_and_enforces_minimum_side():
    assert ai.padded_box(0, 0, 100, 40, radius=50, min_side=150) == (-50, -55, 150, 95)
    x0, y0, x1, y1 = ai.padded_box(0, 0, 10, 10, radius=5, min_side=150)
    assert (x1 - x0, y1 - y0) == (150, 150) and (x0 + x1) / 2 == 5


def test_image_size_keeps_aspect_ratio():
    assert ai.image_size(200, 100, 1000) == (1000, 500)
    assert ai.image_size(100, 200, 1000) == (500, 1000)


def test_clean_settings_defaults_clamps_and_garbage():
    assert ai.clean_settings(None) == ai.DEFAULT_SETTINGS
    s = ai.clean_settings({"max_tokens": 10**6, "radius_m": "abc", "max_image_px": 10, "temperature": 5,
                           "system_prompt": "  ", "model": " m ", "draw_ring": "yes"})
    assert s["max_tokens"] == 32000 and s["radius_m"] == 50.0 and s["max_image_px"] == 256
    assert s["temperature"] == 1.0 and s["system_prompt"] == ai.SYSTEM_PROMPT
    assert s["model"] == "m" and s["draw_ring"] is True


def test_load_settings_roundtrip_and_bad_json():
    custom = ai.clean_settings({"max_tokens": 900, "temperature": 0.2, "system_prompt": "Custom"})
    assert ai.load_settings(json.dumps(custom)) == custom
    assert ai.load_settings("{not json") == ai.DEFAULT_SETTINGS
    assert ai.load_settings(None) == ai.DEFAULT_SETTINGS


def test_payload_uses_custom_system_tokens_and_optional_temperature():
    p = ai.build_payload("x", system="MY PROMPT", max_tokens=900, temperature=0.3)
    assert p["system"] == "MY PROMPT" and p["max_tokens"] == 900 and p["temperature"] == 0.3
    assert "temperature" not in ai.build_payload("x")
    assert ai.prompt_id("a") != ai.prompt_id("b") and len(ai.prompt_id("a")) == 8


def test_dump_probability_explicit_and_fallback():
    r = ai.parse_reply(_resp('{"label": "clean_road", "confidence": 0.96, "dump_probability": 0.04}'))
    assert r["dump_probability"] == 0.04
    assert ai.parse_reply(_resp('{"label": "clean_road", "confidence": 0.9}'))["dump_probability"] == pytest.approx(0.1)
    assert ai.parse_reply(_resp('{"label": "unsure", "confidence": 0.9}'))["dump_probability"] == 0.5


def test_banner_text_and_color():
    assert ai.banner({"dump_probability": 0.93}) == ("Dump found (93%)", "#c62828")
    assert ai.banner({"dump_probability": 0.04}) == ("No dump (04%)", "#2e7d32")
    assert ai.banner({"dump_probability": 0.5})[0] == "Dump found (50%)"
    assert ai.banner({"dump_probability": 1.0})[0] == "Dump found (100%)"


def test_verdict_to_review_mapping():
    f = ai.verdict_to_review
    assert f({"dump_probability": 0.9, "dump_location": "inside"}) == "dump"
    assert f({"dump_probability": 0.9, "dump_location": "partial"}) == "partial"
    assert f({"dump_probability": 0.9, "dump_location": "nearby"}) == "unsure"
    assert f({"dump_probability": 0.9, "dump_location": ""}) == "dump"
    assert f({"dump_probability": 0.1, "dump_location": "none"}) == "clean"


def test_ai_note_has_label_probability_and_reason():
    r = {"label": "clean_road", "dump_probability": 0.04, "reason": "Haul road\nwith  tracks."}
    assert ai.ai_note(r) == "AI: clean_road, dump 0.04 - Haul road with tracks."
    assert ai.ai_note({"label": "dump", "dump_probability": 0.9, "reason": ""}) == "AI: dump, dump 0.90"


def test_auto_next_setting_defaults_off_and_roundtrips():
    assert ai.DEFAULT_SETTINGS["auto_next"] is False
    s = ai.clean_settings({"auto_next": True})
    assert s["auto_next"] is True and ai.load_settings(json.dumps(s))["auto_next"] is True
    assert ai.clean_settings({"auto_next": "yes"})["auto_next"] is False


def test_thinking_param_per_model():
    f = ai.thinking_param
    assert f("claude-haiku-5-5", "default") is None
    assert f("claude-haiku-5-5", "off") == {"type": "disabled"}
    assert f("claude-sonnet-5-5", "off") == {"type": "between_tools"}
    assert f("claude-opus-5-5", "off") is None                      # lässt sich nicht abschalten
    assert f("claude-haiku-5-5", "off", "xhigh") is None            # API lehnt das ab


def test_payload_carries_effort_and_thinking_only_when_set():
    p = ai.build_payload("x", model="claude-haiku-5-5", effort="low", thinking="off")
    assert p["output_config"] == {"effort": "low"} and p["thinking"] == {"type": "disabled"}
    q = ai.build_payload("x")
    assert "output_config" not in q and "thinking" not in q


def test_parse_flags_unreadable_replies_and_banner_says_so():
    r = ai.parse_reply({"content": [{"type": "thinking", "thinking": "hmm"}], "stop_reason": "max_tokens"})
    assert r["ok"] is False and r["stop_reason"] == "max_tokens"
    assert ai.banner(r)[0] == "No answer (unreadable)"
    assert ai.parse_reply(_resp('{"label": "dump", "confidence": 1}'))["ok"] is True


def test_run_check_retries_once_with_bigger_limit_when_thinking_ate_the_tokens():
    calls = []
    empty = {"content": [{"type": "thinking", "thinking": "..."}], "stop_reason": "max_tokens"}
    good = dict(_resp('{"label": "dump", "confidence": 0.9, "dump_probability": 0.9}'), stop_reason="end_turn")

    def opener(req, timeout):
        body = json.loads(req.data)
        calls.append(body["max_tokens"])
        return io.BytesIO(json.dumps(empty if len(calls) == 1 else good).encode())

    out = ai.run_check({"max_tokens": 400}, "k", opener=opener)
    assert calls == [400, 1600] and out["_retried_with"] == 1600 and ai.parse_reply(out)["ok"]


def test_settings_effort_and_thinking_validation():
    s = ai.clean_settings({"effort": "xhigh", "thinking": "off", "max_tokens": 2048})
    assert (s["effort"], s["thinking"]) == ("xhigh", "off")
    s = ai.clean_settings({"effort": "turbo", "thinking": "maybe"})
    assert (s["effort"], s["thinking"]) == ("", "default") and ai.DEFAULT_SETTINGS["max_tokens"] == 2048


def _err(code):
    return urllib.error.HTTPError(ai.API_URL, code, "x", {}, io.BytesIO(b'{"error": {"message": "slow down"}}'))


def test_rate_limit_is_retried_with_backoff_then_succeeds():
    seq, sleeps = [_err(429), _err(529)], []
    good = dict(_resp('{"label": "dump", "confidence": 1, "dump_probability": 1}'), stop_reason="end_turn")

    def opener(req, timeout):
        if seq:
            raise seq.pop(0)
        return io.BytesIO(json.dumps(good).encode())

    out = ai.run_check({"max_tokens": 100}, "k", opener=opener, sleep=sleeps.append)
    assert ai.parse_reply(out)["ok"] and sleeps == [2, 4]


def test_other_errors_and_exhausted_retries_are_not_retried_forever():
    sleeps = []

    def always_429(req, timeout):
        raise _err(429)
    with pytest.raises(RuntimeError, match="429"):
        ai.run_check({"max_tokens": 100}, "k", opener=always_429, sleep=sleeps.append)
    assert sleeps == [2, 4, 8]
    sleeps.clear()

    def bad_key(req, timeout):
        raise _err(401)
    with pytest.raises(RuntimeError, match="401"):
        ai.run_check({"max_tokens": 100}, "k", opener=bad_key, sleep=sleeps.append)
    assert sleeps == []


def test_prefetch_setting_defaults_to_four_and_is_clamped():
    assert ai.DEFAULT_SETTINGS["prefetch"] == 4
    assert ai.clean_settings({"prefetch": 99})["prefetch"] == 16
    assert ai.clean_settings({"prefetch": 0})["prefetch"] == 0


def test_tidy_reason_strips_filler_openers():
    t = ai.tidy_reason
    assert t("The zone shows open-pit terraces, dark waste-rock.") == "Open-pit terraces, dark waste-rock."
    assert t("This area appears to contain bare ground") == "Bare ground"
    assert t("The image shows: roads") == "Roads"
    assert t("Open-pit terraces") == "Open-pit terraces"            # nothing to strip
    assert t("  multi\n line  text ") == "multi line text"
    r = ai.parse_reply(_resp('{"label": "clean_mine", "confidence": 0.8, "reason": "The zone shows terraces."}'))
    assert r["reason"] == "Terraces."


def test_old_default_prompts_are_replaced_by_the_new_default():
    assert "telegraphic" in ai.SYSTEM_PROMPT
    for legacy in ("old default", ):
        pass
    old = ai.SYSTEM_PROMPT.replace("telegraphic", "wordy")
    assert ai.clean_settings({"system_prompt": old})["system_prompt"] == old        # custom prompt stays
    assert ai.prompt_id(ai.SYSTEM_PROMPT) not in ai.LEGACY_PROMPT_IDS


def test_price_table_and_cost_estimate():
    assert ai.price_per_mtok("claude-haiku-5-5") == (0.10, 0.50)
    assert ai.price_per_mtok("claude-sonnet-5-5") == (2.0, 10.0)
    assert ai.price_per_mtok("claude-opus-5-5") == (4.0, 20.0)
    assert ai.price_per_mtok("mystery-model") is None
    assert ai.price_per_mtok("mystery-model", 1.0, 2.0) == (1.0, 2.0)          # override
    assert ai.estimate_cost("claude-haiku-5-5", 2000, 150) == pytest.approx(2000 * 0.10e-6 + 150 * 0.50e-6)
    assert ai.estimate_cost("mystery-model", 1, 1) is None
    assert ai.fmt_cost(0.000275) == "≈ $0.0003" and ai.fmt_cost(0.0234) == "≈ $0.023" and "unbekannt" in ai.fmt_cost(None)


def test_usage_is_read_and_summed_over_the_retry():
    r = dict(_resp('{"label": "dump", "confidence": 1}'), usage={"input_tokens": 2000, "output_tokens": 150})
    assert ai.usage_of(r) == (2000, 150) and ai.parse_reply(r)["tokens_out"] == 150
    assert ai.usage_of({}) == (0, 0)
    first = {"content": [], "stop_reason": "max_tokens", "usage": {"input_tokens": 2000, "output_tokens": 400}}
    second = dict(_resp('{"label": "dump", "confidence": 1}'), stop_reason="end_turn",
                  usage={"input_tokens": 2000, "output_tokens": 900})
    seq = [first, second]
    out = ai.run_check({"max_tokens": 400}, "k", opener=lambda req, timeout: io.BytesIO(json.dumps(seq.pop(0)).encode()))
    assert ai.usage_of(out) == (4000, 1300)


def test_price_override_settings_are_clamped():
    assert ai.clean_settings({"price_in": -5, "price_out": "x"})["price_in"] == 0.0
    assert ai.clean_settings({"price_in": 3, "price_out": 15})["price_out"] == 15.0


def test_banner_has_a_possible_dump_band_for_recall():
    assert ai.banner({"dump_probability": 0.30}) == ("Possible dump (30%)", "#d9822b")
    assert ai.banner({"dump_probability": 0.19})[0] == "No dump (19%)"
    assert ai.banner({"dump_probability": 0.50})[0] == "Dump found (50%)"
    assert ai.banner({"dump_probability": 0.30}, thr_dump=0.5, thr_possible=0.35)[0] == "No dump (30%)"


def test_clean_is_only_entered_when_the_ai_is_sure_and_saw_nothing():
    f = ai.verdict_to_review
    assert f({"dump_probability": 0.05, "dump_location": "none"}) == "clean"
    assert f({"dump_probability": 0.05, "dump_location": ""}) == "clean"
    assert f({"dump_probability": 0.30, "dump_location": "nearby"}) == "unsure"     # your yard case
    assert f({"dump_probability": 0.05, "dump_location": "nearby"}) == "unsure"
    assert f({"dump_probability": 0.30, "dump_location": "none"}) == "unsure"
    assert f({"dump_probability": 0.9, "dump_location": "inside"}) == "dump"


def test_thresholds_are_validated_and_prompt_counts_yards_as_dumps():
    s = ai.clean_settings({"thr_dump": 0.4, "thr_possible": 0.7})
    assert s["thr_possible"] == 0.4                       # never above thr_dump
    assert "storage yards" in ai.SYSTEM_PROMPT and "at least 0.4" in ai.SYSTEM_PROMPT
    assert ai.prompt_id(ai.SYSTEM_PROMPT) not in ai.LEGACY_PROMPT_IDS


def test_radius_follows_context_when_auto_and_falls_back_otherwise():
    cfg = ai.clean_settings({"radius_m": 50, "radius_inside": 200, "radius_near": 100, "radius_desert": 40})
    assert ai.radius_for("inside_op", cfg) == 200 and ai.radius_for("near_op", cfg) == 100
    assert ai.radius_for("open_desert", cfg) == 40
    assert ai.radius_for("", cfg) == 50 and ai.radius_for("mystery", cfg) == 50
    cfg["auto_radius"] = False
    assert ai.radius_for("inside_op", cfg) == 50


def test_context_radius_defaults_and_validation():
    d = ai.DEFAULT_SETTINGS
    assert (d["radius_inside"], d["radius_near"], d["radius_desert"], d["auto_radius"]) == (200.0, 150.0, 100.0, True)
    assert ai.clean_settings({"radius_inside": 9999})["radius_inside"] == 500.0


def test_zone_note_describes_radius_and_surroundings():
    assert "200 m" in ai.zone_note("inside_op", 200) and "mining operation" in ai.zone_note("inside_op", 200)
    assert "open desert" in ai.zone_note("open_desert", 50)
    assert ai.zone_note("", 50) == "Zone radius 50 m"
    assert "radius is given in the message" in ai.SYSTEM_PROMPT


# ---------------------------------------------------------------- new in the plugin: auto-clean, batch, cache
def _res(p, loc="none", label="clean_desert", ok=True):
    return {"ok": ok, "dump_probability": p, "dump_location": loc, "label": label}


def test_auto_clean_needs_desert_low_probability_nothing_seen_and_a_readable_answer():
    cfg = ai.clean_settings({})
    ok = ai.auto_clean_ok
    assert ok(_res(0.04), "open_desert", cfg)
    assert ok(_res(0.10), "open_desert", cfg)                           # boundary counts
    assert not ok(_res(0.11), "open_desert", cfg)
    assert not ok(_res(0.04), "inside_op", cfg) and not ok(_res(0.04), "near_op", cfg) and not ok(_res(0.04), "", cfg)
    assert not ok(_res(0.04, loc="nearby"), "open_desert", cfg)          # tires seen in the zone
    assert not ok(_res(0.04, label="dump"), "open_desert", cfg)
    assert not ok(_res(0.04, ok=False), "open_desert", cfg)
    assert not ok(_res(0.04), "open_desert", {**cfg, "auto_clean": False})


def test_auto_clean_contexts_and_threshold_are_configurable_and_validated():
    cfg = ai.clean_settings({"auto_clean_contexts": "open_desert, near_op, bogus", "auto_clean_p": 0.2})
    assert cfg["auto_clean_contexts"] == "open_desert,near_op"
    assert ai.auto_clean_ok(_res(0.15), "near_op", cfg)
    assert ai.clean_settings({"auto_clean_p": 0.9})["auto_clean_p"] == 0.5     # never above 50 %
    assert ai.DEFAULT_SETTINGS["auto_clean_p"] == 0.10 and ai.DEFAULT_SETTINGS["spot_check_every"] == 10


def test_spot_check_is_about_one_in_n_and_stable():
    picks = [ai.spot_check_pick(f"S{i:04d}", 10) for i in range(2000)]
    assert 150 <= sum(picks) <= 250
    assert picks == [ai.spot_check_pick(f"S{i:04d}", 10) for i in range(2000)]       # same answer every time
    assert all(ai.spot_check_pick(f"S{i}", 1) for i in range(20))                     # every = 1: check them all


def test_batch_outcome_routes_results():
    cfg = ai.clean_settings({"spot_check_every": 10})
    sites = [f"S{i:04d}" for i in range(400)]
    outcomes = [ai.batch_outcome(_res(0.03), "open_desert", s, cfg) for s in sites]
    assert set(outcomes) == {"auto_clean", "spot_check"} and 20 <= outcomes.count("spot_check") <= 60
    assert ai.batch_outcome(_res(0.3, "nearby"), "open_desert", "S1", cfg) == "pending"
    assert ai.batch_outcome(_res(0.03), "inside_op", "S1", cfg) == "pending"


def test_human_followup_measures_agreement_and_spot_check_errors():
    f = ai.human_followup
    assert f("spot_check", "clean", "clean") == "spot_ok"
    assert f("spot_check", "dump", "clean") == "spot_miss" and f("spot_check", "partial", "clean") == "spot_miss"
    assert f("pending", "clean", "clean") == "human_agree" and f("pending", "dump", "unsure") == "human_differs"
    assert f("accepted", "clean", "clean") is None and f("auto_clean", "clean", "clean") is None and f(None, "x", "y") is None


def test_audit_summary_reports_error_rate_of_spot_checks():
    text = ai.audit_summary(["auto_clean"] * 8 + ["spot_check"] * 2 + ["spot_ok"] * 3 + ["spot_miss"])
    assert "Auto-Clean 8" in text and "Stichprobe offen 2" in text and "4: 1 falsch (25%)" in text
    assert "falsch" not in ai.audit_summary(["auto_clean"])


def test_result_from_fields_roundtrip_and_bad_values():
    r = ai.result_from_fields("clean_desert", 0.04, "none", "Bare sand.")
    assert r["ok"] and r["dump_probability"] == 0.04 and r["cached"] and r["usage"]["output"] == 0
    assert ai.banner(r)[0] == "No dump (04%)"
    assert ai.result_from_fields(None, None, None, None) is None
    assert ai.result_from_fields("x", "NaN-ish", None, None) is None


def test_queue_set_external_removes_without_moving_the_current_site():
    q = ai.ReviewQueue([(1, 1, ""), (2, 2, ""), (3, 3, ""), (4, 4, "")])
    assert q.current == 1
    assert q.set_external(3, "clean", "AI auto") and q.reviews[3] == "clean" and q.notes[3] == "AI auto"
    assert list(q.todo) == [1, 2, 4] and q.current == 1 and q.history == []
    assert not q.set_external(99, "clean")
    q.decide("dump")
    assert q.current == 2 and q.counts() == {"dump": 1, "clean": 1}


def test_cache_in_payload_usage_and_cost():
    p = ai.build_payload("x", cache=True)
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"} and p["system"][0]["text"] == ai.SYSTEM_PROMPT
    assert isinstance(ai.build_payload("x")["system"], str)
    resp = dict(_resp('{"label": "dump", "confidence": 1}'),
                usage={"input_tokens": 1000, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 400,
                       "output_tokens": 100})
    assert ai.usage_detail(resp) == {"input": 1000, "cache_write": 500, "cache_read": 400, "output": 100}
    assert ai.usage_of(resp) == (1900, 100)
    cost = ai.cost_of("claude-haiku-5-5", ai.parse_reply(resp))
    assert cost == pytest.approx((1000 * 0.10 + 500 * 0.125 + 400 * 0.01 + 100 * 0.50) / 1e6)
    assert ai.estimate_cost("claude-haiku-5-5", 1900, 100) > cost                  # the cache is cheaper


def test_defaults_for_the_plugin():
    assert ai.DEFAULT_MODEL == "claude-haiku-5-5" and ai.DEFAULT_SETTINGS["prompt_cache"] is True
    assert ai.PREFS_KEY != ai.OLD_PREFS_KEY and ai.KEY_SETTING == "otr_obia/anthropic_api_key"
    assert ai.clean_settings({"batch_budget_usd": -3})["batch_budget_usd"] == 0
    assert ai.clean_settings({"batch_parallel": 99})["batch_parallel"] == 16


def test_log_row_tolerates_new_columns_in_an_old_log(tmp_path):
    path = str(tmp_path / "log.csv")
    ai.log_row(path, {"a": 1, "b": 2})
    ai.log_row(path, {"a": 3, "b": 4, "c": 5})                     # an older header must not break the new row
    assert open(path).read().splitlines() == ["a,b", "1,2", "3,4"]


def test_overview_counts_dumps_possibles_negatives_and_skips():
    entries = [(0.9, "pending"), (0.6, "pending"), (0.3, "pending"), (0.25, "spot_check"), (0.03, "auto_clean"),
               (0.04, "auto_clean"), (0.05, "spot_check"), (0.15, "pending")]
    o = ai.ai_overview(entries)
    assert o == {"n": 8, "dump": 2, "possible": 2, "none": 4, "auto_clean": 2, "spot_check": 2, "to_review": 6}
    text = ai.fmt_overview(o)
    assert "8 geprüft" in text and "Dump 2" in text and "Möglich 2" in text and "Kein Dump 4" in text
    assert "automatisch übersprungen 2" in text and "Stichprobe 2" in text and "zu prüfen 6" in text
    assert ai.ai_overview([], 0.5, 0.2)["n"] == 0
    assert ai.ai_overview([(0.3, "")], thr_dump=0.25)["dump"] == 1                 # thresholds are respected
