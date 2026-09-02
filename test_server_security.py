"""
Security-behavior tests for server.py.

All Gemini call paths are monkeypatched — no network access is needed.
"""
from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server


JPEG_MAGIC = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP_MAGIC = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 64
GIF_MAGIC = b"GIF89a" + b"\x00" * 64


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient with isolated env, log paths, and session store."""
    monkeypatch.delenv("DEBUG_ENDPOINTS", raising=False)
    monkeypatch.setattr(server, "LOG_PATH", tmp_path / "pipeline_log.jsonl")
    monkeypatch.setattr(server, "RESPONSE_LOG_PATH", tmp_path / "gemini_responses.jsonl")
    monkeypatch.setattr(server, "_sessions", server._sessions.__class__())
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# 1. No tracebacks in error responses
# ---------------------------------------------------------------------------
def test_unexpected_error_returns_generic_500_without_traceback(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("sensitive internal detail xyzzy")

    monkeypatch.setattr(server, "generate_parts", boom)
    resp = client.post("/api/generate", json={"description": "a cat", "api_key": "k"})

    assert resp.status_code == 500
    body = resp.json()
    assert "Traceback" not in resp.text
    assert "xyzzy" not in resp.text            # exception message not leaked
    assert "server.py" not in resp.text        # file paths not leaked
    assert len(body["error_id"]) == 8
    assert body["error_id"] in body["detail"]


def test_model_output_error_maps_to_502_without_traceback(client, monkeypatch):
    def bad_model_output(*args, **kwargs):
        raise ValueError("Unexpected response format: {garbage}")

    monkeypatch.setattr(server, "generate_parts", bad_model_output)
    resp = client.post("/api/generate", json={"description": "a cat", "api_key": "k"})

    assert resp.status_code == 502
    assert "Traceback" not in resp.text
    assert "garbage" not in resp.text
    assert "error_id" in resp.json()


def test_feedback_error_path_has_no_traceback(client, monkeypatch):
    sid = server._create_session("a cat", [], np.zeros((2, 2, 2)))

    def boom(*args, **kwargs):
        raise RuntimeError("feedback pipeline blew up")

    monkeypatch.setattr(server, "run_part_world", boom)
    resp = client.post(
        "/api/feedback", json={"session_id": sid, "feedback": "bigger", "api_key": "k"}
    )

    assert resp.status_code == 500
    assert "Traceback" not in resp.text
    assert "blew up" not in resp.text


# ---------------------------------------------------------------------------
# 2. Debug surfaces gated behind DEBUG_ENDPOINTS=1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/api/telemetry", "/api/responses", "/telemetry"])
def test_debug_endpoints_404_without_flag(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("path", ["/api/telemetry", "/api/responses", "/telemetry"])
def test_debug_endpoints_work_with_flag(client, monkeypatch, path):
    monkeypatch.setenv("DEBUG_ENDPOINTS", "1")
    assert client.get(path).status_code == 200


def test_api_responses_last_is_capped_at_50(client, monkeypatch):
    monkeypatch.setenv("DEBUG_ENDPOINTS", "1")
    server.RESPONSE_LOG_PATH.write_text(
        "\n".join('{"n": %d}' % i for i in range(80)) + "\n"
    )
    resp = client.get("/api/responses", params={"last": 100000})
    assert resp.status_code == 200
    assert len(resp.json()["entries"]) == 50


# ---------------------------------------------------------------------------
# 3. Deprecated endpoints removed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/api/run", "/api/validate"])
def test_deprecated_endpoints_are_gone(client, path):
    # No route exists any more. GET falls through to the static mount (404);
    # POST hits the static catch-all which only allows GET/HEAD (405).
    assert client.get(path).status_code == 404
    assert client.post(path, json={}).status_code in (404, 405)
    assert path not in [getattr(r, "path", None) for r in server.app.routes]


# ---------------------------------------------------------------------------
# 4. Session store bound + eviction
# ---------------------------------------------------------------------------
def test_session_store_evicts_oldest_beyond_cap(client, monkeypatch):
    monkeypatch.setattr(server, "_MAX_SESSIONS", 3)
    grid = np.zeros((2, 2, 2))
    sids = [server._create_session(f"desc {i}", [], grid) for i in range(5)]

    assert len(server._sessions) == 3
    assert server._get_session(sids[0]) is None
    assert server._get_session(sids[1]) is None
    assert server._get_session(sids[4]) is not None


def test_lru_touch_protects_recently_used_session(client, monkeypatch):
    monkeypatch.setattr(server, "_MAX_SESSIONS", 2)
    grid = np.zeros((2, 2, 2))
    a = server._create_session("a", [], grid)
    b = server._create_session("b", [], grid)
    server._get_session(a)                      # touch a → b is now oldest
    server._create_session("c", [], grid)       # evicts b
    assert server._get_session(a) is not None
    assert server._get_session(b) is None


def test_feedback_on_evicted_session_returns_404_expired(client, monkeypatch):
    monkeypatch.setattr(server, "_MAX_SESSIONS", 1)
    grid = np.zeros((2, 2, 2))
    old = server._create_session("old", [], grid)
    server._create_session("new", [], grid)     # evicts old

    resp = client.post("/api/feedback", json={"session_id": old, "feedback": "x"})
    assert resp.status_code == 404
    assert "expired" in resp.json()["error"].lower()


def test_export_on_evicted_session_returns_404_expired(client, monkeypatch):
    monkeypatch.setattr(server, "_MAX_SESSIONS", 1)
    grid = np.zeros((2, 2, 2))
    old = server._create_session("old", [], grid)
    server._create_session("new", [], grid)

    resp = client.post("/api/export", json={"session_id": old})
    assert resp.status_code == 404
    assert "expired" in resp.json()["error"].lower()


# ---------------------------------------------------------------------------
# 5. Input limits
# ---------------------------------------------------------------------------
def test_description_over_limit_rejected_422(client):
    resp = client.post(
        "/api/generate",
        json={"description": "x" * (server.MAX_DESCRIPTION_CHARS + 1), "api_key": "k"},
    )
    assert resp.status_code == 422
    assert "too long" in resp.json()["error"].lower()


def test_description_at_limit_passes_length_check(client, monkeypatch):
    sentinel = RuntimeError("reached pipeline")

    def raise_sentinel(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(server, "generate_parts", raise_sentinel)
    resp = client.post(
        "/api/generate",
        json={"description": "x" * server.MAX_DESCRIPTION_CHARS, "api_key": "k"},
    )
    assert resp.status_code == 500  # sentinel fired → length check passed


def test_image_over_12mb_rejected_422(client):
    raw = JPEG_MAGIC + b"\x00" * (server.MAX_IMAGE_BYTES)  # just over cap
    resp = client.post(
        "/api/generate", json={"description": "a cat", "image": b64(raw), "api_key": "k"}
    )
    assert resp.status_code == 422
    assert "too large" in resp.json()["error"].lower()


@pytest.mark.parametrize("raw", [GIF_MAGIC, b"BM" + b"\x00" * 64, b"<svg></svg>"])
def test_non_jpeg_png_webp_image_rejected_422(client, raw):
    resp = client.post(
        "/api/generate", json={"description": "a cat", "image": b64(raw), "api_key": "k"}
    )
    assert resp.status_code == 422
    assert "unsupported" in resp.json()["error"].lower()


@pytest.mark.parametrize(
    "raw,mime",
    [(JPEG_MAGIC, "image/jpeg"), (PNG_MAGIC, "image/png"), (WEBP_MAGIC, "image/webp")],
)
def test_valid_image_magic_bytes_accepted(client, monkeypatch, raw, mime):
    seen = {}

    def fake_describe(image_bytes, mime_type, key):
        seen["mime"] = mime_type
        return "a red cube", {}

    def raise_sentinel(*args, **kwargs):
        raise RuntimeError("reached generation")

    monkeypatch.setattr(server, "describe_image_bytes", fake_describe)
    monkeypatch.setattr(server, "generate_parts", raise_sentinel)

    # Send with a *lying* data-URL mime to prove the sniffed type wins.
    payload = f"data:image/gif;base64,{b64(raw)}"
    resp = client.post(
        "/api/generate", json={"description": "a cat", "image": payload, "api_key": "k"}
    )
    assert resp.status_code == 500  # sentinel fired → image validation passed
    assert seen["mime"] == mime


def test_malformed_data_url_still_400(client):
    resp = client.post(
        "/api/generate",
        json={"description": "a cat", "image": "data:image/png;notbase64", "api_key": "k"},
    )
    assert resp.status_code == 400
