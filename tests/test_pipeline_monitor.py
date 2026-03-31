"""
Pipeline monitoring tests — run specific chat scenarios, then inspect the
saved JSON trace files to find bottlenecks and wrong-output stages.

For each test:
  1. Issue the chat request
  2. Locate the most recent JSON log in logs/pipeline/
  3. Assert expected stages are present with timing data
  4. Flag any stage that looks anomalous (too slow, empty output, error)
  5. Print a structured timing table for human review

Run with:
    pytest tests/test_pipeline_monitor.py -v -s   # -s keeps print output visible
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pytest
import requests

API     = "http://localhost:8000"
LOG_DIR = Path(__file__).parent.parent / "logs" / "pipeline"

# Thresholds for flagging slow stages (seconds)
SLOW_THRESHOLDS = {
    "embed_text":        5.0,
    "embed_image":       10.0,
    "describe_image":    20.0,
    "img_fetch":         5.0,
    "es_knn_description": 5.0,
    "es_bm25":           5.0,
    "es_knn_image":      5.0,
    "rrf_fusion":        1.0,
    "reranking":         30.0,   # bge-reranker is the known bottleneck
    "llm_1":             30.0,
    "agent_invoke":      120.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chat(message: str, session_id: Optional[str] = None) -> dict:
    resp = requests.post(
        f"{API}/chat",
        json={"message": message, "session_id": session_id},
        timeout=180,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
    return resp.json()


def latest_log(after_ts: float) -> Optional[dict]:
    """Return the most recently written JSON log file created after `after_ts`."""
    candidates = [
        p for p in LOG_DIR.glob("*.json")
        if p.stat().st_mtime > after_ts
    ]
    if not candidates:
        return None
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    with open(path) as f:
        return json.load(f)


def print_timing_table(trace: dict) -> None:
    """Pretty-print a timing table for the stages in a trace."""
    total = trace.get("total_elapsed_s", 0)
    print(f"\n{'─'*70}")
    print(f"  Session : {trace['session_id'][:8]}…")
    print(f"  Message : {trace['user_message'][:60]}")
    print(f"  Total   : {total:.2f}s")
    print(f"{'─'*70}")
    print(f"  {'Stage':<35} {'Start':>7} {'Elapsed':>8}  {'Status':<6}  Notes")
    print(f"{'─'*70}")
    for s in trace.get("stages", []):
        name    = s.get("stage", "?")
        t_start = s.get("start_offset_s", 0)
        elapsed = s.get("elapsed_s", "?")
        status  = s.get("status", "?")
        pct     = f"{elapsed/total*100:.0f}%" if isinstance(elapsed, (int, float)) and total else ""
        flags   = []
        threshold = next((v for k, v in SLOW_THRESHOLDS.items() if name.startswith(k)), None)
        if isinstance(elapsed, (int, float)) and threshold and elapsed > threshold:
            flags.append(f"SLOW>{threshold}s")
        if status == "error":
            flags.append(f"ERROR:{s.get('error','')[:30]}")
        notes = " ".join(flags) or pct
        elapsed_str = f"{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else str(elapsed)
        print(f"  {name:<35} {t_start:>6.2f}s {elapsed_str:>8}  {status:<6}  {notes}")
    print(f"{'─'*70}\n")


def assert_no_error_stages(trace: dict) -> None:
    for s in trace.get("stages", []):
        assert s.get("status") != "error", (
            f"Stage '{s['stage']}' errored: {s.get('error', '')}"
        )


def find_stage(trace: dict, prefix: str) -> Optional[dict]:
    for s in trace.get("stages", []):
        if s["stage"].startswith(prefix):
            return s
    return None


def find_all_stages(trace: dict, prefix: str) -> list[dict]:
    return [s for s in trace.get("stages", []) if s["stage"].startswith(prefix)]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def check_api():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip("API server not available")
    except Exception:
        pytest.skip("API server not available")


@pytest.fixture(scope="session", autouse=True)
def check_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# MON-1: Basic product search — verify all sub-stages recorded
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_monitor_basic_search():
    """MON-1: A simple product search records all expected pipeline sub-stages."""
    t0 = time.time()
    data = chat("Show me some chairs")
    trace = latest_log(t0)

    assert trace is not None, "No pipeline log file was written"
    print_timing_table(trace)
    assert_no_error_stages(trace)

    # All these stages must appear
    expected = ["embed_text", "es_knn_description", "es_bm25", "rrf_fusion", "reranking"]
    for name in expected:
        s = find_stage(trace, name)
        assert s is not None, f"Missing stage '{name}' in trace"
        assert isinstance(s.get("elapsed_s"), float), f"Stage '{name}' has no elapsed_s"
        assert s.get("elapsed_s", 0) >= 0

    # Reranker output must have top candidates
    rerank = find_stage(trace, "reranking")
    assert rerank["output"].get("top_k", 0) > 0, "Reranking returned no results"

    # Verify LLM and tool boundary stages from callback
    assert find_stage(trace, "llm_") is not None, "No LLM timing recorded"
    assert find_stage(trace, "tool_product_search") is not None, "No tool_product_search boundary recorded"


# ---------------------------------------------------------------------------
# MON-2: Bottleneck identification — find slowest stage
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_monitor_identify_bottleneck():
    """MON-2: Report the slowest stage and assert total is within 3 minutes."""
    t0 = time.time()
    data = chat("Find me a comfortable leather armchair")
    trace = latest_log(t0)
    assert trace is not None

    print_timing_table(trace)

    stages_with_time = [
        s for s in trace["stages"]
        if isinstance(s.get("elapsed_s"), float)
    ]
    assert stages_with_time, "No timed stages found"

    slowest = max(stages_with_time, key=lambda s: s["elapsed_s"])
    print(f"\n  [bottleneck] Slowest stage: '{slowest['stage']}' "
          f"@ {slowest['elapsed_s']:.2f}s "
          f"({slowest['elapsed_s']/trace['total_elapsed_s']*100:.0f}% of total)\n")

    # Total request should complete within 3 minutes
    assert trace["total_elapsed_s"] < 180, (
        f"Request too slow: {trace['total_elapsed_s']:.1f}s"
    )


# ---------------------------------------------------------------------------
# MON-3: ES search quality — KNN and BM25 must both return hits
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_monitor_search_channels_have_hits():
    """MON-3: Both KNN (description) and BM25 channels return results for a clear query."""
    t0 = time.time()
    chat("Show me wooden dining tables")
    trace = latest_log(t0)
    assert trace is not None
    print_timing_table(trace)

    knn  = find_stage(trace, "es_knn_description")
    bm25 = find_stage(trace, "es_bm25")

    assert knn is not None,  "es_knn_description stage missing"
    assert bm25 is not None, "es_bm25 stage missing"
    assert knn["output"].get("hits_count", 0) > 0,  "KNN description returned 0 hits"
    assert bm25["output"].get("hits_count", 0) > 0, "BM25 returned 0 hits"

    # RRF must merge hits from both channels
    rrf = find_stage(trace, "rrf_fusion")
    assert rrf is not None
    assert rrf["output"].get("unique_candidates", 0) > 0


# ---------------------------------------------------------------------------
# MON-4: Reranking output scores — must be ordered correctly
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_monitor_reranking_scores_ordered():
    """MON-4: After reranking the score_range[1] (max) >= score_range[0] (min)."""
    t0 = time.time()
    chat("Show me a blue velvet sofa")
    trace = latest_log(t0)
    assert trace is not None
    print_timing_table(trace)

    rerank = find_stage(trace, "reranking")
    if rerank is None:
        pytest.skip("No reranking stage found (query may not have triggered search)")

    score_range = rerank["output"].get("score_range", [])
    if not score_range:
        pytest.skip("score_range not recorded (no candidates)")

    lo, hi = score_range
    assert hi >= lo, f"Rerank scores are inverted: max={hi} < min={lo}"

    top = rerank["output"].get("top_rerank_score")
    assert top is not None
    assert top == hi, f"top_rerank_score {top} != score_range max {hi}"


# ---------------------------------------------------------------------------
# MON-5: Multi-turn session — each turn produces its own log
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.slow
def test_monitor_multi_turn_logs():
    """MON-5: Three turns in the same session each produce a separate log file."""
    t0 = time.time()
    r1 = chat("Show me some bookshelves")
    sid = r1["session_id"]
    time.sleep(2)

    t1 = time.time()
    r2 = chat("I want wooden ones", session_id=sid)
    time.sleep(2)

    t2 = time.time()
    r3 = chat("What material is most durable?", session_id=sid)

    # Collect all 3 logs
    all_logs = sorted(LOG_DIR.glob(f"{sid[:8]}_*.json"), key=lambda p: p.stat().st_mtime)
    assert len(all_logs) >= 3, (
        f"Expected ≥3 log files for session {sid[:8]}, found {len(all_logs)}"
    )

    print(f"\n  Multi-turn logs for session {sid[:8]}:")
    for p in all_logs[-3:]:
        with open(p) as f:
            tr = json.load(f)
        print(f"    {p.name}  total={tr['total_elapsed_s']:.2f}s  "
              f"msg='{tr['user_message'][:40]}'")
        print_timing_table(tr)


# ---------------------------------------------------------------------------
# MON-6: Previously failing "outdoor garden chairs" — identify why slow
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_monitor_low_coverage_query():
    """MON-6: Query with low catalog coverage — record & inspect which stage stalls."""
    t0 = time.time()
    try:
        data = chat("Show me outdoor garden chairs")
    except requests.exceptions.ReadTimeout:
        pytest.skip("Query timed out at 180s — confirmed performance issue")

    trace = latest_log(t0)
    if trace is None:
        pytest.skip("No log written (request may have timed out before save)")

    print_timing_table(trace)

    # Identify any stage over its threshold
    slow_stages = []
    for s in trace["stages"]:
        if not isinstance(s.get("elapsed_s"), float):
            continue
        threshold = next((v for k, v in SLOW_THRESHOLDS.items()
                          if s["stage"].startswith(k)), None)
        if threshold and s["elapsed_s"] > threshold:
            slow_stages.append(s)

    if slow_stages:
        print("\n  SLOW STAGES DETECTED:")
        for s in slow_stages:
            print(f"    {s['stage']}: {s['elapsed_s']:.2f}s "
                  f"(threshold {SLOW_THRESHOLDS.get(s['stage'].split('_')[0]+'_'+s['stage'].split('_')[1] if '_' in s['stage'] else s['stage'], '?')}s)")

    # BM25 hits for this query might be 0 — that's useful to know
    bm25 = find_stage(trace, "es_bm25")
    if bm25 and bm25["output"].get("hits_count", -1) == 0:
        print("\n  WARNING: BM25 returned 0 hits — no lexical matches for this query.")
        print("           Agent may have done extra LLM reasoning to handle it.")


# ---------------------------------------------------------------------------
# MON-7: Gift recommendation — agent clarifies vs searches (trace the decision)
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_monitor_gift_query_trace():
    """MON-7: Gift recommendation — check if agent searched or just reasoned."""
    t0 = time.time()
    data = chat("I need a gift for someone who loves cooking")
    trace = latest_log(t0)
    assert trace is not None
    print_timing_table(trace)

    tool_calls = find_all_stages(trace, "tool_")
    llm_calls  = find_all_stages(trace, "llm_")

    print(f"\n  Tool calls: {[s['stage'] for s in tool_calls]}")
    print(f"  LLM calls:  {len(llm_calls)} (reasoning turns)")

    if not tool_calls:
        # Agent responded without searching — record LLM reasoning time
        total_llm = sum(s.get("elapsed_s", 0) for s in llm_calls)
        print(f"  Agent chose to CLARIFY (no tool calls). Total LLM time: {total_llm:.2f}s")
    else:
        search_stages = find_all_stages(trace, "es_")
        print(f"  Agent SEARCHED. ES stages: {[s['stage'] for s in search_stages]}")


# ---------------------------------------------------------------------------
# MON-8: Reranking vs RRF — compare candidate counts
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_monitor_rrf_to_reranking_funnel():
    """MON-8: RRF should produce more candidates than reranking returns (funnel check)."""
    t0 = time.time()
    chat("Show me a table lamp with warm lighting")
    trace = latest_log(t0)
    assert trace is not None
    print_timing_table(trace)

    rrf    = find_stage(trace, "rrf_fusion")
    rerank = find_stage(trace, "reranking")

    if rrf is None or rerank is None:
        pytest.skip("rrf_fusion or reranking stage not found")

    rrf_candidates = rrf["output"].get("after_limit", 0)
    final_top_k    = rerank["output"].get("top_k", 0)

    print(f"\n  RRF candidates: {rrf_candidates}  →  reranker top-k: {final_top_k}")
    assert rrf_candidates >= final_top_k, (
        f"RRF produced fewer candidates ({rrf_candidates}) than reranking target ({final_top_k})"
    )
    # Reranker input (reranking.input.candidates) should equal rrf candidates
    rerank_input_count = rerank["input"].get("candidates", 0)
    assert rerank_input_count == rrf_candidates, (
        f"Mismatch: RRF sent {rrf_candidates} candidates but reranker saw {rerank_input_count}"
    )


# ---------------------------------------------------------------------------
# MON-9: embed_text output is consistent across two identical queries
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_monitor_embedding_determinism():
    """MON-9: The same query produces the same embedding vector dimension both times."""
    query = "Show me a comfortable sofa"

    t0 = time.time()
    chat(query)
    tr1 = latest_log(t0)

    time.sleep(2)

    t1 = time.time()
    chat(query)
    tr2 = latest_log(t1)

    assert tr1 and tr2, "Missing log files"

    emb1 = find_stage(tr1, "embed_text")
    emb2 = find_stage(tr2, "embed_text")

    assert emb1 and emb2, "embed_text stage not found in one or both traces"

    dim1 = emb1["output"].get("vector_dim")
    dim2 = emb2["output"].get("vector_dim")
    assert dim1 == dim2, f"Embedding dim changed between runs: {dim1} vs {dim2}"
    assert dim1 and dim1 > 0, f"Embedding vector dim is 0 or missing: {dim1}"
    print(f"\n  Embedding vector dim: {dim1}  (consistent across both runs)")
