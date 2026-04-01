"""
Tests for Area 2: Guardrails

Covers:
  1. /ready endpoint — 503 before startup, 200 after
  2. Tool retry wrapper — timeout path, error path, structured error return
"""
import json
import sys
import time
import unittest.mock as mock
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

BASE = "http://localhost:8000"


# ---------------------------------------------------------------------------
# /ready endpoint
# ---------------------------------------------------------------------------

class TestReadyEndpoint:
    def test_ready_returns_200_after_startup(self):
        """After the server has fully started, /ready must return 200."""
        r = requests.get(f"{BASE}/ready", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        assert r.json().get("status") == "ready"

    def test_ready_body_schema(self):
        """Response body must contain status='ready'."""
        r = requests.get(f"{BASE}/ready", timeout=5)
        body = r.json()
        assert "status" in body
        assert body["status"] == "ready"

    def test_health_still_works(self):
        """/health must still return 200 (unchanged endpoint)."""
        r = requests.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

    def test_ready_flag_logic(self):
        """Unit-test the flag directly: simulate startup_complete=False."""
        import importlib
        import api.main as main_mod

        # Save original value
        original = main_mod._startup_complete

        try:
            # Simulate not-yet-started state
            main_mod._startup_complete = False
            from fastapi.testclient import TestClient
            client = TestClient(main_mod.app, raise_server_exceptions=False)
            resp = client.get("/ready")
            assert resp.status_code == 503, f"Expected 503 when not ready, got {resp.status_code}"

            # Simulate started state
            main_mod._startup_complete = True
            resp = client.get("/ready")
            assert resp.status_code == 200
        finally:
            main_mod._startup_complete = original


# ---------------------------------------------------------------------------
# Tool retry wrapper
# ---------------------------------------------------------------------------

class TestToolRetryWrapper:
    def setup_method(self):
        """Import the retry wrapper internals."""
        import tools as tools_mod
        self.tools_mod = tools_mod

    def test_success_path_passes_through(self):
        """A function that succeeds on first try should return its value unchanged."""
        from tools import _tool_with_retry

        @_tool_with_retry
        def always_ok(*args, **kwargs):
            return "good result"

        assert always_ok() == "good result"

    def test_exception_returns_structured_error(self):
        """A function that raises should return JSON with error + tool keys."""
        from tools import _tool_with_retry

        @_tool_with_retry
        def always_fails(*args, **kwargs):
            raise ValueError("boom")

        result = always_fails()
        data = json.loads(result)
        assert "error" in data
        assert "tool" in data
        assert data["tool"] == "always_fails"
        assert "boom" in data["error"]

    def test_timeout_returns_structured_error(self):
        """A function that hangs beyond the timeout should return a timeout error JSON."""
        import concurrent.futures
        from tools import _tool_with_retry, _TOOL_TIMEOUT_S

        call_count = {"n": 0}

        @_tool_with_retry
        def always_hangs(*args, **kwargs):
            call_count["n"] += 1
            time.sleep(999)  # will be killed by timeout

        # Patch the timeout to 1s so the test is fast
        with mock.patch("tools._TOOL_TIMEOUT_S", 1):
            result = always_hangs()

        data = json.loads(result)
        assert "error" in data
        assert "timed out" in data["error"].lower() or "timeout" in data["error"].lower()
        assert data["tool"] == "always_hangs"

    def test_timeout_retries_once(self):
        """On timeout, the wrapper should retry exactly once before giving up."""
        import tools as t

        call_count = {"n": 0}

        @t._tool_with_retry
        def slow_fn(*args, **kwargs):
            call_count["n"] += 1
            time.sleep(999)

        with mock.patch("tools._TOOL_TIMEOUT_S", 1):
            result = slow_fn()

        # Should have been called _MAX_RETRIES + 1 times (1 initial + 1 retry)
        assert call_count["n"] == t._MAX_RETRIES + 1, (
            f"Expected {t._MAX_RETRIES + 1} calls, got {call_count['n']}"
        )
        data = json.loads(result)
        assert "error" in data

    def test_product_search_tool_metadata_intact(self):
        """Decorating with @tool + @_tool_with_retry must not corrupt tool metadata."""
        from tools import product_search, get_product_info, understand_image
        for tool in (product_search, get_product_info, understand_image):
            assert tool.name, f"{tool} has no .name"
            assert tool.description, f"{tool} has no .description"
            assert hasattr(tool, "args"), f"{tool} has no .args"
