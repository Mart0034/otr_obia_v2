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
    assert r == {"label": "dump", "confidence": 0.8, "dump_location": "inside", "reason": "dark piles"}


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
