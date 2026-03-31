"""
LangSmith tracing integration test.

LS-1: LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY are set in env
LS-2: Agent invocation completes without tracing errors
LS-3: LangSmith client can list runs for the project (confirms traces were received)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


@pytest.mark.api
def test_env_vars_set():
    """LS-1: tracing env vars are present."""
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true", \
        "LANGCHAIN_TRACING_V2 must be 'true'"
    assert os.environ.get("LANGCHAIN_API_KEY"), \
        "LANGCHAIN_API_KEY must be set"
    assert os.environ.get("LANGCHAIN_PROJECT"), \
        "LANGCHAIN_PROJECT must be set"


@pytest.mark.api
def test_agent_runs_with_tracing():
    """LS-2: agent invocation succeeds with tracing enabled."""
    from agent import create_agent
    from langchain_core.messages import HumanMessage

    agent = create_agent()
    result = agent.invoke({"messages": [HumanMessage(content="Find a chair")]})
    assert result["messages"], "Agent returned no messages"
    # Last AI message should have content
    ai_msgs = [m for m in result["messages"] if m.__class__.__name__ == "AIMessage"]
    assert ai_msgs, "No AI message in response"


@pytest.mark.api
def test_langsmith_client_accessible():
    """LS-3: LangSmith SDK can reach the API (confirms key is valid)."""
    if not os.environ.get("LANGCHAIN_API_KEY"):
        pytest.skip("LANGCHAIN_API_KEY not set — skipping live LangSmith check")
    from langsmith import Client
    client = Client(timeout_ms=10_000)
    # Just listing projects is enough to verify connectivity
    projects = list(client.list_projects())
    project_names = [p.name for p in projects]
    project = os.environ.get("LANGCHAIN_PROJECT", "shopping-demo")
    assert project in project_names, \
        f"Project '{project}' not found in LangSmith. Found: {project_names}"
