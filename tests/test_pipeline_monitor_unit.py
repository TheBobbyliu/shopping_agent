from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from pipeline_monitor import PipelineCallback, PipelineMonitor, set_monitor
from tools import _tool_with_retry


def test_tool_with_retry_preserves_monitor_context():
    monitor = PipelineMonitor(session_id="unit-session", user_message="unit test")

    @_tool_with_retry
    def record_stage() -> str:
        from pipeline_monitor import get_monitor

        inner_monitor = get_monitor()
        assert inner_monitor is not None
        with inner_monitor.stage("inner_tool_stage") as out:
            out["ok"] = True
        return json.dumps({"ok": True})

    set_monitor(monitor)
    try:
        result = record_stage()
    finally:
        set_monitor(None)

    assert json.loads(result) == {"ok": True}
    stage = next((s for s in monitor.stages if s["stage"] == "inner_tool_stage"), None)
    assert stage is not None
    assert stage["output"] == {"ok": True}
    assert stage["status"] == "ok"


def test_pipeline_callback_parses_tool_message_content():
    monitor = PipelineMonitor(session_id="unit-session", user_message="unit test")
    callback = PipelineCallback(monitor)
    run_id = "run-1"

    callback.on_tool_start({"name": "product_search"}, "{'query': 'chair'}", run_id=run_id)

    class DummyToolMessage:
        def __init__(self, content: str):
            self.content = content

    callback.on_tool_end(
        DummyToolMessage(json.dumps([{"item_id": "A1"}, {"item_id": "B2"}])),
        run_id=run_id,
    )

    stage = next(s for s in monitor.stages if s["stage"] == "tool_product_search_1")
    assert stage["status"] == "ok"
    assert stage["output"]["result_count"] == 2
    assert stage["output"]["item_ids"] == ["A1", "B2"]
