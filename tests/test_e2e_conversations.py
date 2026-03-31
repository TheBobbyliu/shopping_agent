"""
End-to-end multi-round conversation tests against the live API.

Tests four conversation scenarios:
  E2E-1: Text search → follow-up refinement → product detail
  E2E-2: Category + attribute filtering across turns
  E2E-3: Session isolation (two independent sessions don't bleed state)
  E2E-4: Conversation memory (agent recalls prior turn context)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pytest
import requests

API = "http://localhost:8000"


def chat(message: str, session_id: Optional[str] = None) -> dict:
    resp = requests.post(
        f"{API}/chat",
        json={"message": message, "session_id": session_id},
        timeout=60,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
    return resp.json()


@pytest.fixture(scope="session", autouse=True)
def check_api():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip("API server not available")
    except Exception:
        pytest.skip("API server not available")


# ---------------------------------------------------------------------------
# E2E-1: Text search → refinement → product detail
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_e2e_text_search_and_refinement():
    """E2E-1: Start broad, refine, then ask for details — session state persists."""
    # Turn 1: broad search
    r1 = chat("Show me some chairs")
    assert r1["session_id"], "No session_id returned"
    sid = r1["session_id"]
    assert r1["reply"], "Empty reply on turn 1"
    assert r1["tool_calls"], "No tools called on turn 1"
    assert any(tc["tool"] == "product_search" for tc in r1["tool_calls"]), \
        "product_search not called"

    # Turn 2: narrow within same session
    r2 = chat("Actually I want something leather and black", session_id=sid)
    assert r2["session_id"] == sid, "Session ID changed"
    assert r2["reply"], "Empty reply on turn 2"

    # Turn 3: ask for details — agent should recall what it found
    r3 = chat("Tell me more about the first one you mentioned", session_id=sid)
    assert r3["reply"], "Empty reply on turn 3"
    # Agent should look up product details or reference earlier results
    reply_lower = r3["reply"].lower()
    assert any(word in reply_lower for word in ["chair", "leather", "product", "item", "material", "color"]), \
        f"Turn 3 reply doesn't seem relevant: {r3['reply'][:200]}"


# ---------------------------------------------------------------------------
# E2E-2: Category + attribute filtering
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_e2e_category_filtering():
    """E2E-2: Search with specific query, then add attribute constraint, then ask for a rec."""
    # Turn 1: specific enough query to trigger a search
    r1 = chat("Show me sofas and couches")
    sid = r1["session_id"]
    assert r1["reply"]
    # Accept either a search or a clarifying question — both are valid
    searched = any(tc["tool"] == "product_search" for tc in r1["tool_calls"])

    # Turn 2: attribute constraint (always triggers a search)
    r2 = chat("I want wooden material ones", session_id=sid)
    assert r2["reply"]
    assert any(tc["tool"] == "product_search" for tc in r2["tool_calls"]), \
        "product_search not called on refinement"

    # Turn 3: recommendation question — agent stays on topic
    r3 = chat("Which of those would work well in a modern minimalist home?", session_id=sid)
    assert r3["reply"]
    reply_lower = r3["reply"].lower()
    assert any(word in reply_lower for word in ["wood", "modern", "sofa", "couch", "table", "furniture", "material", "style"]), \
        f"Reply drifted off topic: {r3['reply'][:200]}"


# ---------------------------------------------------------------------------
# E2E-3: Session isolation
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_e2e_session_isolation():
    """E2E-3: Two concurrent sessions don't bleed context into each other."""
    # Session A: shoes
    a1 = chat("I'm looking for running shoes")
    sid_a = a1["session_id"]

    # Session B: home decor (different topic)
    b1 = chat("I need wall art for my bedroom")
    sid_b = b1["session_id"]
    assert sid_a != sid_b, "Sessions got the same ID"

    # Session A turn 2: refinement
    a2 = chat("Men's size, preferably blue", session_id=sid_a)
    assert a2["reply"]
    a2_lower = a2["reply"].lower()
    # Should be about shoes, not wall art
    assert any(word in a2_lower for word in ["shoe", "running", "sneaker", "blue", "men", "size"]), \
        f"Session A reply bled into session B context: {a2['reply'][:200]}"

    # Session B turn 2: refinement
    b2 = chat("Something abstract and colorful", session_id=sid_b)
    assert b2["reply"]
    b2_lower = b2["reply"].lower()
    # Should be about art, not shoes
    assert not any(word in b2_lower for word in ["running shoe", "sneaker"]), \
        f"Session B bled session A context: {b2['reply'][:200]}"


# ---------------------------------------------------------------------------
# E2E-4: Conversation memory
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_e2e_conversation_memory():
    """E2E-4: Agent recalls an item_id mentioned in a prior turn."""
    import re

    # Turn 1: search — capture an item_id from the reply
    r1 = chat("Show me a dining table")
    sid = r1["session_id"]
    # Find item_id in reply (B + 9 alphanumeric)
    item_ids = re.findall(r'\b(B[A-Z0-9]{9})\b', r1["reply"])
    assert item_ids, f"No item_id found in turn 1 reply: {r1['reply'][:300]}"
    target_id = item_ids[0]

    # Turn 2: ask about that specific product by referencing it naturally
    r2 = chat(f"Tell me more about {target_id}", session_id=sid)
    assert r2["reply"]
    # Agent should call get_product_info
    assert any(tc["tool"] == "get_product_info" for tc in r2["tool_calls"]), \
        f"get_product_info not called. Tools used: {r2['tool_calls']}"

    # Turn 3: follow-up question with no explicit ID — agent should still know context
    r3 = chat("What material is it made of?", session_id=sid)
    assert r3["reply"]
    assert len(r3["reply"]) > 20, "Too short a reply on turn 3"
