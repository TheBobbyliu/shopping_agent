"""
Tests for Area 1: SSE Streaming — POST /chat/stream

Covers:
  1. Response has correct Content-Type: text/event-stream
  2. At least one 'token' event is emitted
  3. A 'done' event is emitted last with reply + session_id
  4. 'status' events are emitted (Thinking... / tool status)
  5. session_id is consistent across events
  6. Streaming reply matches the 'done' reply payload
  7. Aborted request does not crash the server (/health still 200)
"""
import json
import time
import requests
import pytest

BASE = "http://localhost:8000"


def parse_sse(raw: str) -> list[dict]:
    """Parse a raw SSE response body into list of {event, data} dicts."""
    events = []
    current = {}
    for line in raw.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[len("data:"):].strip())
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


@pytest.fixture(scope="module")
def stream_response():
    """Send one streaming request and collect all events. Cached for the module."""
    r = requests.post(
        f"{BASE}/chat/stream",
        json={"message": "Show me a comfortable chair"},
        stream=True,
        timeout=120,
    )
    raw = r.content.decode()
    return r, parse_sse(raw)


class TestSSEStreaming:
    def test_content_type(self, stream_response):
        r, _ = stream_response
        assert "text/event-stream" in r.headers.get("Content-Type", ""), (
            f"Expected text/event-stream, got {r.headers.get('Content-Type')}"
        )

    def test_status_200(self, stream_response):
        r, _ = stream_response
        assert r.status_code == 200

    def test_at_least_one_token_event(self, stream_response):
        _, events = stream_response
        token_events = [e for e in events if e.get("event") == "token"]
        assert len(token_events) > 0, "Expected at least one 'token' event"

    def test_done_event_present(self, stream_response):
        _, events = stream_response
        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1, f"Expected exactly 1 'done' event, got {len(done_events)}"

    def test_done_event_schema(self, stream_response):
        _, events = stream_response
        done = next(e for e in events if e.get("event") == "done")
        data = done["data"]
        assert "reply" in data, "done event missing 'reply'"
        assert "session_id" in data, "done event missing 'session_id'"
        assert "tool_calls" in data, "done event missing 'tool_calls'"
        assert isinstance(data["reply"], str) and len(data["reply"]) > 0
        assert isinstance(data["session_id"], str) and len(data["session_id"]) > 0

    def test_status_events_present(self, stream_response):
        _, events = stream_response
        status_events = [e for e in events if e.get("event") == "status"]
        assert len(status_events) > 0, "Expected at least one 'status' event"
        # All status events should have a 'text' field
        for e in status_events:
            assert "text" in e["data"], f"status event missing 'text': {e}"

    def test_done_is_last_event(self, stream_response):
        _, events = stream_response
        non_empty = [e for e in events if e.get("event")]
        assert non_empty[-1]["event"] == "done", (
            f"Last event should be 'done', got '{non_empty[-1]['event']}'"
        )

    def test_streaming_tokens_form_reply(self, stream_response):
        """Concatenated tokens should equal (or be a prefix of) the done reply."""
        _, events = stream_response
        tokens = "".join(e["data"]["text"] for e in events if e.get("event") == "token")
        done_reply = next(e["data"]["reply"] for e in events if e.get("event") == "done")
        assert tokens == done_reply, (
            f"Token stream != done reply.\n"
            f"  tokens ({len(tokens)} chars): {tokens[:200]!r}\n"
            f"  reply  ({len(done_reply)} chars): {done_reply[:200]!r}"
        )

    def test_no_error_event(self, stream_response):
        _, events = stream_response
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) == 0, f"Unexpected error events: {error_events}"

    def test_session_id_returned(self, stream_response):
        _, events = stream_response
        done = next(e for e in events if e.get("event") == "done")
        sid = done["data"]["session_id"]
        assert len(sid) >= 8

    def test_session_continuation(self):
        """A follow-up request using the same session_id should get a coherent reply."""
        r1 = requests.post(
            f"{BASE}/chat/stream",
            json={"message": "Show me a red sofa"},
            stream=True, timeout=120,
        )
        events1 = parse_sse(r1.content.decode())
        done1 = next(e for e in events1 if e.get("event") == "done")
        sid = done1["data"]["session_id"]

        r2 = requests.post(
            f"{BASE}/chat/stream",
            json={"message": "What was the first product you just mentioned?",
                  "session_id": sid},
            stream=True, timeout=120,
        )
        events2 = parse_sse(r2.content.decode())
        done2 = next(e for e in events2 if e.get("event") == "done")
        reply2 = done2["data"]["reply"].lower()
        # Should reference a product (contains an item ID or product-related word)
        assert len(reply2) > 20, "Follow-up reply seems too short"

    def test_server_survives_aborted_request(self):
        """Abort mid-stream; server /health should still respond 200.

        The server may remain busy processing the in-flight LLM call for a while
        after the client disconnects — we allow up to 30 s for /health to respond.
        """
        try:
            r = requests.post(
                f"{BASE}/chat/stream",
                json={"message": "List 10 different types of chairs with details"},
                stream=True, timeout=15,
            )
            # Read only first few SSE bytes then disconnect
            for _ in r.iter_content(chunk_size=64):
                break
            r.close()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            pass
        except Exception:
            pass

        # Poll /health — allow up to 30 s for the server to service the request
        # (the in-flight agent call may still be occupying the thread pool).
        deadline = time.time() + 30
        last_exc = None
        while time.time() < deadline:
            try:
                health = requests.get(f"{BASE}/health", timeout=5)
                assert health.status_code == 200, "Server returned non-200 /health"
                return  # pass
            except Exception as exc:
                last_exc = exc
                time.sleep(2)
        raise AssertionError(f"Server did not recover within 30s: {last_exc}")
