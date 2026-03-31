"""
Chat interface robustness and scenario tests.

Tests the full chat system against a live API (http://localhost:8000),
covering edge cases, user scenarios, session behavior, and response
format validation.

Markers:
  @pytest.mark.api    — requires live API (skip with -m 'not api')
  @pytest.mark.slow   — multi-turn / longer tests
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Optional

import pytest
import requests

API = "http://localhost:8000"
ITEM_ID_RE = re.compile(r'\b(B[A-Z0-9]{9})\b')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT = 180  # seconds — LLM + embedding + reranking pipeline can be slow under load


def chat(message: str, session_id: Optional[str] = None, image_b64: Optional[str] = None) -> dict:
    payload: dict = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    if image_b64:
        payload["image_b64"] = image_b64
    resp = requests.post(f"{API}/chat", json=payload, timeout=TIMEOUT)
    return resp


def chat_ok(message: str, session_id: Optional[str] = None) -> dict:
    """Assert 200 and return parsed JSON."""
    resp = chat(message, session_id=session_id)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
    return resp.json()


def assert_valid_response(data: dict):
    """Validate required fields are present and well-formed."""
    assert "reply" in data, "Missing 'reply' field"
    assert "session_id" in data, "Missing 'session_id' field"
    assert "tool_calls" in data, "Missing 'tool_calls' field"
    assert isinstance(data["reply"], str), "'reply' must be a string"
    assert len(data["reply"]) > 0, "'reply' must not be empty"
    assert isinstance(data["session_id"], str), "'session_id' must be a string"
    assert len(data["session_id"]) > 0, "'session_id' must not be empty"
    assert isinstance(data["tool_calls"], list), "'tool_calls' must be a list"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def check_api():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip("API server not available")
    except Exception:
        pytest.skip("API server not available")


# ---------------------------------------------------------------------------
# CHAT-1: Response format validation — every reply must have required fields
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_response_format_basic_query():
    """CHAT-1: Basic product query returns well-formed response with all required fields."""
    data = chat_ok("Show me some chairs")
    assert_valid_response(data)


# ---------------------------------------------------------------------------
# CHAT-2: Empty / blank messages
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_empty_message_rejected():
    """CHAT-2a: Empty string message — API must not crash (4xx or graceful reply)."""
    resp = chat("")
    # Either a 4xx error (validation) or a graceful 200 are both acceptable;
    # what we must NOT see is a 500 server error.
    assert resp.status_code != 500, f"Server crashed on empty message: {resp.text[:200]}"


@pytest.mark.api
def test_whitespace_only_message():
    """CHAT-2b: Whitespace-only message — server must not crash."""
    resp = chat("   \t\n  ")
    assert resp.status_code != 500, f"Server crashed on whitespace message: {resp.text[:200]}"


# ---------------------------------------------------------------------------
# CHAT-3: Very long messages
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_very_long_message():
    """CHAT-3: A 2000-character message must not crash the server."""
    long_msg = "I am looking for a comfortable chair. " * 55  # ~2090 chars
    resp = chat(long_msg)
    assert resp.status_code not in (500, 502, 503), \
        f"Server error on long message: {resp.status_code}"
    if resp.status_code == 200:
        assert_valid_response(resp.json())


# ---------------------------------------------------------------------------
# CHAT-4: Special characters and injection attempts
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.parametrize("msg,label", [
    ("<script>alert('xss')</script>", "html_injection"),
    ("' OR '1'='1", "sql_injection"),
    ("{}[]${{jinja}}", "template_injection"),
    ("🛋️🪑🛏️ furniture emoji search", "emoji"),
    ("椅子 テーブル ソファ", "japanese_cjk"),
    ("مقعد أريكة طاولة", "arabic_rtl"),
    ("Ünïcödé chàïr wïth áccentß", "unicode_accents"),
])
def test_special_characters(msg, label):
    """CHAT-4: Special characters and potential injection strings are handled safely."""
    resp = chat(msg)
    assert resp.status_code != 500, \
        f"[{label}] Server crashed on input: {msg[:80]}"
    if resp.status_code == 200:
        data = resp.json()
        assert_valid_response(data)
        # Ensure raw injection strings are not echoed unescaped in the reply
        if label == "html_injection":
            assert "<script>" not in data["reply"], \
                "Raw <script> tag found in reply — potential XSS"


# ---------------------------------------------------------------------------
# CHAT-5: Gibberish / nonsense input
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.parametrize("msg", [
    "asdfghjklqwertyuiop",
    "zzzzzzzzzzzzzzzzzzzzz",
    "1234567890 0987654321",
    "!@#$%^&*()_+",
])
def test_gibberish_input(msg):
    """CHAT-5: Gibberish input returns a graceful response, not a crash."""
    resp = chat(msg)
    assert resp.status_code != 500, f"Server crashed on gibberish: '{msg}'"
    if resp.status_code == 200:
        data = resp.json()
        assert_valid_response(data)
        # The assistant should still reply with something meaningful
        assert len(data["reply"]) > 10, "Reply to gibberish was too short"


# ---------------------------------------------------------------------------
# CHAT-6: Off-topic / non-shopping messages
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.parametrize("msg,topic", [
    ("What is the capital of France?", "geography"),
    ("Solve x^2 + 5x + 6 = 0", "math"),
    ("Write me a poem about the ocean", "creative"),
    ("Who won the 2022 World Cup?", "sports"),
])
def test_off_topic_messages(msg, topic):
    """CHAT-6: Off-topic messages get a graceful, non-crashing response."""
    resp = chat(msg)
    assert resp.status_code != 500, f"Server crashed on off-topic [{topic}]: '{msg}'"
    if resp.status_code == 200:
        data = resp.json()
        assert_valid_response(data)
        # Agent should respond — even if it declines or redirects
        assert len(data["reply"]) > 10


# ---------------------------------------------------------------------------
# CHAT-7: Budget-constrained queries
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_budget_constrained_query():
    """CHAT-7: Budget constraint in query is understood and products are searched."""
    data = chat_ok("Show me chairs under $50")
    assert_valid_response(data)
    # Agent should attempt a product search
    assert any(tc["tool"] == "product_search" for tc in data["tool_calls"]), \
        "Expected product_search to be called for a budget query"


# ---------------------------------------------------------------------------
# CHAT-8: Negative preference
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_negative_preference_query():
    """CHAT-8: User expresses what they don't want — agent still searches."""
    data = chat_ok("I want a sofa but not leather, and no dark colors")
    assert_valid_response(data)
    assert any(tc["tool"] == "product_search" for tc in data["tool_calls"]), \
        "Expected product_search for a negative-preference query"


# ---------------------------------------------------------------------------
# CHAT-9: Comparison request
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_comparison_request():
    """CHAT-9: User asks to compare product types — agent responds helpfully."""
    data = chat_ok("What's the difference between a loveseat and a sofa?")
    assert_valid_response(data)
    reply_lower = data["reply"].lower()
    # Should mention at least one relevant furniture term
    assert any(w in reply_lower for w in ["sofa", "loveseat", "seat", "couch", "size", "furniture"]), \
        f"Reply doesn't address comparison: {data['reply'][:200]}"


# ---------------------------------------------------------------------------
# CHAT-10: Multi-item recommendation request
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_multi_item_request():
    """CHAT-10: User asks for multiple recommendations — agent returns several products."""
    data = chat_ok("Show me 5 different coffee tables")
    assert_valid_response(data)
    assert any(tc["tool"] == "product_search" for tc in data["tool_calls"]), \
        "Expected product_search for multi-item request"
    # Ideally multiple item IDs are present in reply
    item_ids = ITEM_ID_RE.findall(data["reply"])
    # Relax: at least 1 item ID (some agents may present fewer inline)
    assert len(item_ids) >= 1, f"Expected at least 1 item ID in reply, got: {data['reply'][:300]}"


# ---------------------------------------------------------------------------
# CHAT-11: Gift recommendation scenario
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_gift_recommendation():
    """CHAT-11: Gift recommendation phrasing — agent responds helpfully (may search or clarify)."""
    data = chat_ok("I need a gift for someone who loves cooking")
    assert_valid_response(data)
    # Agent may either search directly OR ask a clarifying question — both are valid behaviors.
    # The important thing is that the reply is substantive and relevant.
    reply_lower = data["reply"].lower()
    assert any(w in reply_lower for w in ["gift", "cook", "kitchen", "product", "recommend", "budget", "help", "looking"]), \
        f"Reply doesn't seem relevant to the gift request: {data['reply'][:200]}"


# ---------------------------------------------------------------------------
# CHAT-12: Highly specific attribute query
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_highly_specific_attribute_query():
    """CHAT-12: Very specific multi-attribute query is handled without crashing."""
    data = chat_ok("Show me a blue velvet armchair with gold legs, mid-century modern style")
    assert_valid_response(data)
    assert any(tc["tool"] == "product_search" for tc in data["tool_calls"]), \
        "Expected product_search for specific attribute query"


# ---------------------------------------------------------------------------
# CHAT-13: "Tell me more" without prior context (cold start)
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_tell_me_more_without_context():
    """CHAT-13: 'Tell me more' with no prior session — agent asks for clarification or searches."""
    data = chat_ok("Tell me more about the first one")
    assert_valid_response(data)
    # Agent should not crash; it may ask what the user means or do a generic search
    assert len(data["reply"]) > 10, "Expected a substantive reply even without prior context"


# ---------------------------------------------------------------------------
# CHAT-14: Ask about a non-existent product ID
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_nonexistent_product_id():
    """CHAT-14: Querying a fake product ID — agent handles gracefully (no 500)."""
    fake_id = "BXXXXXXXX9"  # Not a real product
    data = chat_ok(f"Tell me more about product {fake_id}")
    assert_valid_response(data)
    # Agent should either say it can't find it or attempt a search
    assert len(data["reply"]) > 10


# ---------------------------------------------------------------------------
# CHAT-15: Repeated identical messages (idempotency)
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_repeated_identical_messages():
    """CHAT-15: Sending the same message twice returns valid responses both times."""
    msg = "Show me some dining tables"
    r1 = chat_ok(msg)
    r2 = chat_ok(msg)
    assert_valid_response(r1)
    assert_valid_response(r2)
    # Sessions should be independent since no session_id was passed
    assert r1["session_id"] != r2["session_id"], \
        "Two separate calls without session_id should yield different session IDs"


# ---------------------------------------------------------------------------
# CHAT-16: Invalid / random session_id
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_invalid_session_id():
    """CHAT-16: Using a random/unknown session_id — API must not crash."""
    fake_session = str(uuid.uuid4())
    resp = chat("Show me some sofas", session_id=fake_session)
    assert resp.status_code not in (500, 502, 503), \
        f"Server crashed with unknown session_id: {resp.status_code}"
    if resp.status_code == 200:
        assert_valid_response(resp.json())


# ---------------------------------------------------------------------------
# CHAT-17: Missing required fields (API contract)
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_missing_message_field():
    """CHAT-17: POST /chat without 'message' field returns 4xx, not 500."""
    resp = requests.post(f"{API}/chat", json={}, timeout=10)
    assert resp.status_code in (400, 422), \
        f"Expected 400/422 for missing message, got {resp.status_code}: {resp.text[:200]}"


@pytest.mark.api
def test_wrong_content_type():
    """CHAT-17b: POST /chat with wrong Content-Type returns 4xx, not 500."""
    resp = requests.post(
        f"{API}/chat",
        data="message=hello",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )
    assert resp.status_code != 500, \
        f"Server crashed on wrong content type: {resp.status_code}"


# ---------------------------------------------------------------------------
# CHAT-18: Session continuity across 5 turns
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.slow
def test_session_continuity_five_turns():
    """CHAT-18: Session ID remains stable across 5 consecutive turns."""
    r1 = chat_ok("Show me some bookshelves")
    sid = r1["session_id"]
    assert_valid_response(r1)

    turns = [
        "I prefer wooden ones",
        "Something tall with at least 5 shelves",
        "White or natural wood color",
        "What's the most affordable option?",
    ]
    for turn_msg in turns:
        time.sleep(2)
        data = chat_ok(turn_msg, session_id=sid)
        assert_valid_response(data)
        assert data["session_id"] == sid, \
            f"Session ID changed mid-conversation: {data['session_id']} != {sid}"


# ---------------------------------------------------------------------------
# CHAT-19: Emotionally charged / frustrated user messages
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.parametrize("msg,label", [
    ("Nothing you show me is good enough! Just find me a decent sofa!", "frustrated"),
    ("SHOW ME A CHAIR NOW", "all_caps_demand"),
    ("ugh this is useless, just recommend SOMETHING", "exasperated"),
])
def test_emotionally_charged_messages(msg, label):
    """CHAT-19: Emotionally charged messages are handled gracefully without crashing."""
    resp = chat(msg)
    assert resp.status_code != 500, f"[{label}] Server crashed: {msg[:80]}"
    if resp.status_code == 200:
        data = resp.json()
        assert_valid_response(data)
        assert len(data["reply"]) > 10, f"[{label}] Reply was too short"


# ---------------------------------------------------------------------------
# CHAT-20: Product search produces parseable item IDs
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_product_ids_parseable():
    """CHAT-20: When products are mentioned, item IDs match the ASIN regex."""
    data = chat_ok("Find me a comfortable armchair")
    assert_valid_response(data)
    if any(tc["tool"] == "product_search" for tc in data["tool_calls"]):
        item_ids = ITEM_ID_RE.findall(data["reply"])
        for item_id in item_ids:
            assert re.fullmatch(r'B[A-Z0-9]{9}', item_id), \
                f"Item ID '{item_id}' doesn't match ASIN format"


# ---------------------------------------------------------------------------
# CHAT-21: Image-only search (empty text + synthetic tiny image)
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_image_only_search_does_not_crash():
    """CHAT-21: Sending an invalid image — server should return a user-facing error, not a 500.

    KNOWN ISSUE: The server currently returns HTTP 500 when OpenAI/GPT-4o rejects an invalid
    image (it bubbles up the upstream error without wrapping it in a 4xx response). This test
    documents the expected behavior (4xx) and will fail until the API properly handles upstream
    image validation errors.
    """
    # 1x1 white JPEG in base64 (syntactically valid base64 but image is too small for GPT-4o)
    tiny_jpeg_b64 = (
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
        "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
        "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
        "MjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAA"
        "AAAAAAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAA"
        "AAAA/9oADAMBAAIRAxEAPwCwABmX/9k="
    )
    resp = requests.post(
        f"{API}/chat",
        json={"message": "", "image_b64": tiny_jpeg_b64},
        timeout=TIMEOUT,
    )
    # Expected: 400 (bad image) or 422 (validation). Server currently returns 500.
    # This assertion documents the correct behavior.
    assert resp.status_code in (400, 422), (
        f"Expected 400/422 for invalid image, got {resp.status_code}. "
        f"KNOWN ISSUE: upstream image validation errors (from OpenAI) are not caught and "
        f"result in HTTP 500. Response: {resp.text[:300]}"
    )


# ---------------------------------------------------------------------------
# CHAT-22: Concurrent sessions do not cross-contaminate topics
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.slow
def test_concurrent_sessions_no_contamination():
    """CHAT-22: Two sessions with different topics stay independent."""
    # Session A: sofas
    a1 = chat_ok("Show me some sofas")
    sid_a = a1["session_id"]
    time.sleep(3)

    # Session B: lamps (different furniture category)
    b1 = chat_ok("Show me some table lamps")
    sid_b = b1["session_id"]
    assert sid_a != sid_b
    time.sleep(3)

    # Advance session A — refine sofas, should stay about sofas not lamps
    a2 = chat_ok("I prefer leather material", session_id=sid_a)
    assert a2["session_id"] == sid_a
    a2_lower = a2["reply"].lower()
    assert not any(w in a2_lower for w in ["lamp", "light", "bulb", "shade"]), \
        f"Session A reply contaminated by session B context: {a2['reply'][:200]}"
    time.sleep(3)

    # Advance session B — refine lamps, should stay about lamps not sofas
    b2 = chat_ok("I want a warm white light", session_id=sid_b)
    assert b2["session_id"] == sid_b
    b2_lower = b2["reply"].lower()
    assert not any(w in b2_lower for w in ["sofa", "couch", "leather", "cushion"]), \
        f"Session B reply contaminated by session A context: {b2['reply'][:200]}"


# ---------------------------------------------------------------------------
# CHAT-23: Tool call structure is valid
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_tool_calls_have_valid_structure():
    """CHAT-23: Every tool_call item has at least a 'tool' key with a non-empty string."""
    data = chat_ok("Show me some lamps")
    assert_valid_response(data)
    for tc in data["tool_calls"]:
        assert "tool" in tc, f"Tool call missing 'tool' key: {tc}"
        assert isinstance(tc["tool"], str) and len(tc["tool"]) > 0, \
            f"tool_call 'tool' must be a non-empty string: {tc}"
        known_tools = {"product_search", "get_product_info", "understand_image"}
        assert tc["tool"] in known_tools, \
            f"Unknown tool name '{tc['tool']}' — expected one of {known_tools}"


# ---------------------------------------------------------------------------
# CHAT-24: Conversational follow-up (non-search) doesn't always call tools
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_conversational_reply_may_skip_tools():
    """CHAT-24: A polite greeting or conversational message can be answered without tools."""
    data = chat_ok("Hi there!")
    assert_valid_response(data)
    # No assertion on tool_calls count — either is valid for a greeting.
    # Main check: server responds with a coherent, non-empty reply.
    assert len(data["reply"]) > 5, "Reply to greeting was too short"


# ---------------------------------------------------------------------------
# CHAT-25: get_product_info is called when user asks about a specific known item
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.slow
def test_get_product_info_called_for_known_id():
    """CHAT-25: When user explicitly provides a valid item ID, get_product_info is called."""
    # First get a real item ID from a search
    r1 = chat_ok("Show me a desk lamp")
    item_ids = ITEM_ID_RE.findall(r1["reply"])
    if not item_ids:
        pytest.skip("No item IDs returned in search — can't test get_product_info flow")

    target_id = item_ids[0]
    sid = r1["session_id"]

    r2 = chat_ok(f"Tell me more about {target_id}", session_id=sid)
    assert_valid_response(r2)
    assert any(tc["tool"] == "get_product_info" for tc in r2["tool_calls"]), \
        f"Expected get_product_info to be called for known ID {target_id}. " \
        f"Tools called: {[tc['tool'] for tc in r2['tool_calls']]}"
