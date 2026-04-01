"""
Pipeline monitoring — records timing, inputs, and outputs for every stage
in a request, then saves a JSON file under logs/pipeline/.

Usage
-----
In api/main.py:
    monitor = PipelineMonitor(session_id, user_message)
    set_monitor(monitor)
    try:
        result = agent.invoke(...)
    finally:
        monitor.save()
        set_monitor(None)

In search.py (or any downstream code):
    with pipeline_stage("embed_text", {"query": query[:80]}) as out:
        vec = client.embed_text(query)
        out["vector_dim"] = len(vec)

Thread-safety
-------------
Each request runs in its own thread (FastAPI sync endpoint → thread pool).
`threading.local()` gives each thread an independent monitor slot.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOG_DIR = Path(__file__).parent.parent / "logs" / "pipeline"

# ContextVar propagates from the request thread into LangGraph's run_in_executor
# threads, unlike threading.local() which is per-thread only.
_monitor_var: ContextVar[Optional["PipelineMonitor"]] = ContextVar("pipeline_monitor", default=None)


# ---------------------------------------------------------------------------
# Core monitor class
# ---------------------------------------------------------------------------

class PipelineMonitor:
    def __init__(self, session_id: str, user_message: str):
        self.session_id   = session_id
        self.user_message = user_message
        self._t0          = time.perf_counter()
        self.timestamp    = datetime.now(timezone.utc).isoformat()
        self.stages: list[dict] = []

    # --- stage context manager -------------------------------------------

    @contextmanager
    def stage(self, name: str, input_summary: dict | None = None):
        """
        Record one pipeline stage.

        Yields a dict `out` that callers can populate with output metadata:
            with monitor.stage("embed_text", {"query": q}) as out:
                vec = embed(q)
                out["vector_dim"] = len(vec)
        """
        entry: dict[str, Any] = {
            "stage":          name,
            "start_offset_s": round(time.perf_counter() - self._t0, 4),
            "input":          input_summary or {},
            "output":         {},
        }
        self.stages.append(entry)
        t_stage = time.perf_counter()
        try:
            yield entry["output"]
            entry["elapsed_s"] = round(time.perf_counter() - t_stage, 4)
            entry["status"]    = "ok"
        except Exception as exc:
            entry["elapsed_s"] = round(time.perf_counter() - t_stage, 4)
            entry["status"]    = "error"
            entry["error"]     = str(exc)
            raise

    # --- save ----------------------------------------------------------------

    def save(self) -> str:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        fname = _LOG_DIR / f"{ts}_{self.session_id[:8]}.json"

        # Sort stages by start_offset_s so the file reads chronologically
        # (callback-inserted entries may land out of insertion order)
        sorted_stages = sorted(self.stages, key=lambda s: s.get("start_offset_s", 0))

        doc = {
            "session_id":      self.session_id,
            "timestamp":       self.timestamp,
            "user_message":    self.user_message,
            "total_elapsed_s": round(time.perf_counter() - self._t0, 4),
            "stages":          sorted_stages,
        }
        with open(fname, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
        return str(fname)


# ---------------------------------------------------------------------------
# Thread-local accessor
# ---------------------------------------------------------------------------

def get_monitor() -> Optional[PipelineMonitor]:
    """Return the monitor for the current context, or None if not set."""
    return _monitor_var.get()


def set_monitor(monitor: Optional[PipelineMonitor]) -> None:
    _monitor_var.set(monitor)


# ---------------------------------------------------------------------------
# Convenience context manager for use in search.py and tools.py
# ---------------------------------------------------------------------------

@contextmanager
def pipeline_stage(name: str, input_summary: dict | None = None):
    """
    No-op when no monitor is active; otherwise delegates to monitor.stage().
    Safe to call from anywhere without importing PipelineMonitor directly.
    """
    mon = get_monitor()
    if mon is not None:
        with mon.stage(name, input_summary) as out:
            yield out
    else:
        yield {}


# ---------------------------------------------------------------------------
# LangChain callback — captures LLM turns and tool boundaries
# ---------------------------------------------------------------------------

try:
    from langchain_core.callbacks import BaseCallbackHandler

    class PipelineCallback(BaseCallbackHandler):
        """
        Attaches to agent.invoke() via config["callbacks"].
        Records one stage entry per LLM call and per tool call.
        Sub-stages from search.py are inserted by pipeline_stage() and will be
        interleaved in the list; save() sorts everything by start_offset_s.
        """

        def __init__(self, monitor: PipelineMonitor) -> None:
            self._mon         = monitor
            self._llm_starts: dict[str, tuple[float, dict]] = {}
            self._tool_starts: dict[str, tuple[float, dict]] = {}
            self._llm_count   = 0
            self._tool_count  = 0

        # LLM events
        def on_llm_start(self, serialized, prompts, run_id=None, **kwargs):
            self._llm_count += 1
            t0 = time.perf_counter()
            entry: dict[str, Any] = {
                "stage":          f"llm_{self._llm_count}",
                "start_offset_s": round(t0 - self._mon._t0, 4),
                "input": {
                    "model":       (serialized or {}).get("name", "?"),
                    "num_prompts": len(prompts) if prompts else 0,
                },
                "output": {},
            }
            self._mon.stages.append(entry)
            self._llm_starts[str(run_id)] = (t0, entry)

        def on_llm_end(self, response, run_id=None, **kwargs):
            key = str(run_id)
            if key not in self._llm_starts:
                return
            t0, entry = self._llm_starts.pop(key)
            entry["elapsed_s"] = round(time.perf_counter() - t0, 4)
            entry["status"]    = "ok"

            # Capture the LLM output text
            try:
                texts = []
                for gen_list in response.generations:
                    for g in gen_list:
                        text = getattr(g, "text", "") or ""
                        if not text and hasattr(g, "message"):
                            content = getattr(g.message, "content", "")
                            if isinstance(content, str):
                                text = content
                        if text:
                            texts.append(text)
                if texts:
                    full_text = "\n".join(texts)
                    entry["output"]["text_preview"] = full_text[:400]
                    entry["output"]["text_chars"]   = len(full_text)
            except Exception:
                pass

            # Extract token usage — OpenAI stores it in response.llm_output,
            # Anthropic and others may store it in generation_info.
            try:
                usage: dict = {}
                llm_output = getattr(response, "llm_output", None) or {}
                tok = llm_output.get("token_usage", {})
                if tok:
                    usage = {
                        "input_tokens":  tok.get("prompt_tokens"),
                        "output_tokens": tok.get("completion_tokens"),
                        "total_tokens":  tok.get("total_tokens"),
                    }
                if not any(v for v in usage.values() if v is not None):
                    for gen_list in response.generations:
                        for g in gen_list:
                            info = getattr(g, "generation_info", None) or {}
                            for k in ("input_tokens", "output_tokens", "total_tokens"):
                                if k in info:
                                    usage[k] = info[k]
                filtered = {k: v for k, v in usage.items() if v is not None}
                if filtered:
                    entry["output"]["token_usage"] = filtered
            except Exception:
                pass

        def on_llm_error(self, error, run_id=None, **kwargs):
            key = str(run_id)
            if key not in self._llm_starts:
                return
            t0, entry = self._llm_starts.pop(key)
            entry["elapsed_s"] = round(time.perf_counter() - t0, 4)
            entry["status"]    = "error"
            entry["error"]     = str(error)

        # Tool events
        def on_tool_start(self, serialized, input_str, run_id=None, **kwargs):
            self._tool_count += 1
            tool_name = (serialized or {}).get("name", "unknown")
            t0 = time.perf_counter()
            entry: dict[str, Any] = {
                "stage":          f"tool_{tool_name}_{self._tool_count}",
                "start_offset_s": round(t0 - self._mon._t0, 4),
                "input":          {"args_preview": str(input_str)[:300]},
                "output":         {},
            }
            self._mon.stages.append(entry)
            self._tool_starts[str(run_id)] = (t0, entry)

        def on_tool_end(self, output, run_id=None, **kwargs):
            key = str(run_id)
            if key not in self._tool_starts:
                return
            t0, entry = self._tool_starts.pop(key)
            entry["elapsed_s"] = round(time.perf_counter() - t0, 4)
            entry["status"]    = "ok"
            try:
                raw_output = getattr(output, "content", output)
                if isinstance(raw_output, list):
                    text_parts = []
                    for block in raw_output:
                        if isinstance(block, str):
                            text_parts.append(block)
                        elif isinstance(block, dict) and isinstance(block.get("text"), str):
                            text_parts.append(block["text"])
                    out_str = "".join(text_parts) if text_parts else str(raw_output)
                elif isinstance(raw_output, str):
                    out_str = raw_output
                else:
                    out_str = str(raw_output)

                # For product_search, show how many results came back
                import json as _json
                parsed = _json.loads(out_str)
                if isinstance(parsed, list):
                    entry["output"]["result_count"] = len(parsed)
                    entry["output"]["item_ids"] = [p.get("item_id") for p in parsed[:5]]
                elif isinstance(parsed, dict):
                    entry["output"]["keys"] = list(parsed.keys())[:10]
                    if "error" in parsed:
                        entry["output"]["error"] = str(parsed["error"])[:200]
                    else:
                        entry["output"]["preview"] = out_str[:200]
                else:
                    entry["output"]["preview"] = out_str[:200]
            except Exception:
                entry["output"]["preview"] = str(getattr(output, "content", output))[:200]

        def on_tool_error(self, error, run_id=None, **kwargs):
            key = str(run_id)
            if key not in self._tool_starts:
                return
            t0, entry = self._tool_starts.pop(key)
            entry["elapsed_s"] = round(time.perf_counter() - t0, 4)
            entry["status"]    = "error"
            entry["error"]     = str(error)

except ImportError:
    # langchain_core not available — provide a no-op stub
    class PipelineCallback:  # type: ignore
        def __init__(self, monitor):
            pass
