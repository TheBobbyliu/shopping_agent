from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

import project_logging
from tools import _tool_with_retry


def test_record_error_writes_daily_log(monkeypatch, tmp_path):
    monkeypatch.setattr(project_logging, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(project_logging, "_LOG_DIR", tmp_path / "logs" / "errors")

    err = ValueError("boom")
    log_file = project_logging.record_error(
        "api.chat",
        err,
        context={"session_id": "sess-1"},
    )

    payload = project_logging.build_error_response_content(log_file)
    log_path = tmp_path / log_file

    assert payload["log_file"] == log_file
    assert payload["detail"].endswith(f"See {log_file}.")
    assert log_path.is_file()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[-1])
    assert entry["event"] == "api.chat"
    assert entry["error_type"] == "ValueError"
    assert entry["error"] == "boom"
    assert entry["context"]["session_id"] == "sess-1"
    assert "ValueError: boom" in entry["traceback"]


def test_tool_with_retry_records_daily_error_log(monkeypatch, tmp_path):
    monkeypatch.setattr(project_logging, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(project_logging, "_LOG_DIR", tmp_path / "logs" / "errors")

    @_tool_with_retry
    def explode() -> str:
        raise RuntimeError("kaboom")

    result = json.loads(explode())
    log_paths = list((tmp_path / "logs" / "errors").glob("*.log"))

    assert result["tool"] == "explode"
    assert result["error"] == "kaboom"
    assert len(log_paths) == 1

    entry = json.loads(log_paths[0].read_text(encoding="utf-8").splitlines()[-1])
    assert entry["event"] == "tool.explode"
    assert entry["error_type"] == "RuntimeError"
    assert entry["error"] == "kaboom"
