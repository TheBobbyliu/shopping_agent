"""
PostgreSQL-backed conversation context tests.

CM-1: Agent with checkpointer remembers prior turn in same thread
CM-2: Different thread IDs are isolated (no cross-session bleed)
CM-3: API /chat endpoint accepts and uses session_id
CM-4: Session history persists across separate agent invocations
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

PG_URL = os.environ.get("DATABASE_URL", "postgresql://shopping:shopping@localhost:5432/shopping")


@pytest.fixture(scope="session")
def checkpointer():
    """Create a PostgreSQL checkpointer, skip if DB not available."""
    try:
        from context import make_checkpointer
        cp = make_checkpointer(PG_URL)
        return cp
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


@pytest.mark.api
def test_agent_remembers_prior_turn(checkpointer):
    """CM-1: second message in same thread references first."""
    from agent import create_agent
    from langchain_core.messages import HumanMessage

    agent  = create_agent(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    agent.invoke({"messages": [HumanMessage(content="My name is TestUser")]}, thread)
    result = agent.invoke({"messages": [HumanMessage(content="What is my name?")]}, thread)

    ai_msgs = [m for m in result["messages"] if m.__class__.__name__ == "AIMessage"]
    last_reply = ai_msgs[-1].content.lower()
    assert "testuser" in last_reply, f"Agent forgot the name. Reply: {last_reply[:200]}"


@pytest.mark.api
def test_threads_are_isolated(checkpointer):
    """CM-2: two threads don't share state."""
    from agent import create_agent
    from langchain_core.messages import HumanMessage

    agent   = create_agent(checkpointer=checkpointer)
    thread1 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    thread2 = {"configurable": {"thread_id": str(uuid.uuid4())}}

    agent.invoke({"messages": [HumanMessage(content="I love red shoes")]}, thread1)
    result = agent.invoke({"messages": [HumanMessage(content="What did I just say I love?")]}, thread2)

    ai_msgs = [m for m in result["messages"] if m.__class__.__name__ == "AIMessage"]
    last_reply = ai_msgs[-1].content.lower()
    assert "red shoes" not in last_reply, \
        f"Thread isolation broken — thread2 knows about thread1's red shoes: {last_reply[:200]}"


@pytest.mark.api
def test_api_session_id(tmp_path):
    """CM-3: /chat endpoint accepts session_id and returns it."""
    import requests
    base = "http://localhost:8000"
    try:
        r = requests.get(f"{base}/health", timeout=2)
        if r.status_code != 200:
            pytest.skip("API server not running")
    except Exception:
        pytest.skip("API server not running on port 8000")

    sid = str(uuid.uuid4())
    resp = requests.post(f"{base}/chat", json={
        "message": "Hello, remember me",
        "session_id": sid,
        "history": [],
    }, timeout=30)
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert data.get("session_id") == sid


@pytest.mark.api
def test_session_persists_across_requests(tmp_path):
    """CM-4: two sequential /chat calls with same session_id share memory."""
    import requests
    base = "http://localhost:8000"
    try:
        r = requests.get(f"{base}/health", timeout=2)
        if r.status_code != 200:
            pytest.skip("API server not running")
    except Exception:
        pytest.skip("API server not running on port 8000")

    sid = str(uuid.uuid4())
    requests.post(f"{base}/chat", json={
        "message": "My favourite color is purple",
        "session_id": sid,
        "history": [],
    }, timeout=30)

    resp2 = requests.post(f"{base}/chat", json={
        "message": "What is my favourite color?",
        "session_id": sid,
        "history": [],
    }, timeout=30)

    assert resp2.status_code == 200
    reply = resp2.json()["reply"].lower()
    assert "purple" in reply, f"Session memory not working. Reply: {reply[:200]}"
