"""Tests for the request/response logic of the QGIS AI classifier (no QGIS, no network)."""
import io
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qgis"))
import ai_classifier as ai  # noqa: E402


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
    assert r == {"ok": True, "stop_reason": "", "label": "dump", "confidence": 0.8, "dump_probability": 0.8,
                 "dump_location": "inside", "reason": "dark piles"}


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
