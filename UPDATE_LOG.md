# Update Log

---

## 2026-04-02 — Warehouse CLI

New `preprocessing/warehouse.py` CLI for operational staff to manage the Elasticsearch product database. Also adds `POST /embed` to the API server so the CLI reuses the already-warm embedding model.

### Commands

| Command | Description |
|---|---|
| `python warehouse.py add items.json` | Embed and index new products; skips already-indexed items |
| `python warehouse.py delete items.json` | Delete products by `item_id`; warns on missing |
| `python warehouse.py check --item-id ID` | Fetch a single product (vectors stripped) |
| `python warehouse.py check --group CATEGORY [--count N]` | Fetch N products by category |
| `python warehouse.py check --count N` | Fetch N products across all categories |

### New / modified files

| File | Change |
|---|---|
| `api/main.py` | Added `POST /embed` — exposes warm `_get_embedding_client()` singleton; returns 400 for missing image, 503 if not ready |
| `preprocessing/warehouse.py` | New CLI entry point; all command logic |
| `tests/test_warehouse.py` | 2 `/embed` integration tests + 13 unit tests + 6 `@pytest.mark.api` integration tests |

### Design highlights

- Embedding routed through running API server (`POST /embed`) — no cold model load in CLI
- Embed API unreachable → exits immediately with error message (spec requirement)
- `cmd_add` refreshes ES index before exiting (even on embed failure) so partial batches are visible
- All skip/warn messages go to stderr; JSON output and summaries go to stdout
- Input validation: required fields, duplicate `item_id` detection, non-dict array elements

### Test results (unit)

```
pytest tests/test_warehouse.py -k "not api"  →  13/13 ✓
```

---

## 2026-04-01 — Frontend Image Flow + History Restore Fixes

### Bugs found and fixed during live browser testing

**Bug 1 — Next.js frontend had no real `/api/chat/stream` route**
- Root cause: `frontend-next/app/api/chat/route.ts` contained logic for both `/api/chat` and `/api/chat/stream`, but only the `/api/chat` route actually existed. Browser image requests to `/api/chat/stream` fell through to the dev rewrite path instead of the intended proxy handler.
- Symptom: image-only requests in the frontend stayed on `Connecting...` / `Searching for products...` and never rendered the final answer, even though direct backend calls to `POST /chat/stream` worked.
- Fix: added `frontend-next/app/api/chat/stream/route.ts` as a dedicated SSE proxy route that forwards the request body to `http://localhost:8000/chat/stream` and returns `text/event-stream` directly.

**Bug 2 — saved image chats intentionally dropped the uploaded preview**
- Root cause: `frontend-next/components/ChatWindow.tsx` stripped `imagePreview` from every message before writing `messages_<session_id>` to `localStorage`.
- Symptom: reopening a previous image chat showed the assistant reply and product cards, but the original uploaded image was gone.
- Fix: changed chat-history persistence to store `messages` as-is so user messages keep their `imagePreview` data URL.
- Note: sessions saved before this fix cannot recover their missing preview because the image data was never stored.

### New regression coverage

- `tests/test_frontend_history.py` — browser regression test for image-chat history restore
- `tests/scripts/frontend_image_history_regression.js` — Playwright script that:
  - mocks `/api/ready` and `/api/chat/stream`
  - uploads a fixture image
  - saves the chat
  - switches away and back
  - asserts the uploaded preview still appears
- `tests/fixtures/upload-test.svg` — lightweight deterministic upload fixture for the browser test

### Verification

| Check | Result |
|-------|--------|
| Live browser image upload on `http://127.0.0.1:3000` | reply now renders ✓ |
| Reopen saved image chat from sidebar | uploaded preview now persists ✓ |
| `pytest tests/test_frontend_history.py -v` | 1/1 ✓ |

---

## 2026-03-31 — Test Suite + Bug Fixes

### New test files
- `tests/test_guardrails.py` — 9 tests for `/ready` endpoint and `_tool_with_retry`
- `tests/test_parallel_embed.py` — 6 tests for sequential/parallel embedding paths
- `tests/test_sse_streaming.py` — 12 tests for SSE streaming: content-type, token events, done schema, status events, session continuation, abort survival

### Bugs found and fixed during testing

**Bug 1 — `tools.py`: `ThreadPoolExecutor` context manager hung on timeout**
- Root cause: `with ThreadPoolExecutor() as ex:` calls `shutdown(wait=True)` on `__exit__`, which blocks indefinitely waiting for the timed-out thread (sleeping for 999s in tests, 90s in production).
- Fix: replaced `with` context manager with explicit `ex.shutdown(wait=False)` after `future.result()` raises `TimeoutError`.
- Impact: without this fix, a timed-out tool call would block the request thread for the full sleep duration instead of retrying.

**Bug 2 — `search.py`: `asyncio.gather()` incompatible with Python 3.11 when called outside async context**
- Root cause: in Python 3.11, `asyncio.gather()` returns a `Future` (not a coroutine) if called when an event loop exists but isn't running; `asyncio.run()` rejects non-coroutines.
- Fix: wrapped the gather call in an explicit `async def _gather_embeds()` function so `asyncio.run()` always receives a true coroutine.

**Bug 3 — `embed.py`: HuggingFace Rust tokenizer not thread-safe (`RuntimeError: Already borrowed`)**
- Root cause: `embed_texts` and `embed_images` share the same `Visualized_BGE` model instance (and its tokenizer). Concurrent calls from `asyncio.to_thread` caused the Rust borrow checker to raise.
- Fix: added `threading.Lock()` on `_LocalBackend` wrapping the tokenizer+inference sections of both `embed_texts` and `embed_images`.
- Note: the lock serializes the model calls, so `PARALLEL_EMBED=1` parallelism primarily benefits I/O-bound work (image download, image decode) rather than the model itself.

**Bug 4 — `api/main.py`: `agent.stream()` blocking sync iterator starved the async event loop**
- Root cause: `agent.stream()` is a blocking synchronous iterator called directly inside an `async def _generate()` generator. Each iteration blocked the entire uvicorn event loop during LLM/tool calls, preventing `/health` and other endpoints from being served.
- Fix: moved `agent.stream()` into `asyncio.to_thread(_run_agent)`. The thread feeds results into an `asyncio.Queue`; the async generator `await queue.get()` between items, keeping the event loop free.

### Final test results
| Suite | Tests | Result |
|-------|-------|--------|
| `test_guardrails.py` | 9 | 9/9 ✓ |
| `test_parallel_embed.py` | 6 | 6/6 ✓ |
| `test_sse_streaming.py` | 12 | 12/12 ✓ |
| **Total** | **27** | **27/27 ✓** |

---

## 2026-03-31 — Full Implementation (4 Areas)

### 1. Backend Guardrails (`agent/tools.py`, `api/main.py`)

- **Tool retry wrapper**: Added `_tool_with_retry` decorator — 90s timeout, 1 retry on timeout, structured `{"error": ..., "tool": ...}` return on failure so the agent can surface errors to the user instead of crashing.
- **Error logging**: Added `_log_tool_error` helper that records failures through the existing `PipelineMonitor` machinery.
- **`/ready` endpoint**: Added `_startup_complete` flag in `api/main.py`; set to `True` at end of `_startup()`. `GET /ready` returns 503 until warm-up completes, then 200 — used by frontend to gate rendering.

### 2. Performance (`agent/search.py`)

- **Parallel embedding**: Text + image embeddings now run concurrently under `PARALLEL_EMBED=1` env flag using `asyncio.run(asyncio.gather(asyncio.to_thread(...), asyncio.to_thread(...)))`. Safe because `chat()` is a sync FastAPI handler (no running event loop).
- **Reranker analysis**: Added comment above reranker call documenting why multithreading won't help (batches all 50 candidates in one call; GIL + CUDA serialization at C++ level makes parallelism moot).

### 3. Frontend UI — Streaming + Redesign

**New infrastructure:**
- `lib/api.ts` — added `checkReady()` and `streamChat()` (fetch + ReadableStream SSE parser); kept `sendChat()` as fallback.
- `app/api/chat/route.ts` — SSE proxy route passes `text/event-stream` through with no timeout cap.
- `api/main.py` — `POST /chat/stream` SSE endpoint: emits `status`, `token`, `done`, `error` events using `agent.stream(stream_mode="messages")`.

**New/rewritten components:**
- `ChatWindow.tsx` — orchestrates streaming state, optimistic assistant message, session creation + localStorage save.
- `ChatMessage.tsx` — `react-markdown` + `remark-gfm` for assistant messages; `memo()` on `ProductCard` + `ChatMessage`; module-level `MD_COMPONENTS` map.
- `ChatSidebar.tsx` — localStorage-backed session history; `sessions-updated` custom event for cross-component sync.
- `ChatInput.tsx` — image upload + textarea + send button, extracted from ChatWindow.
- `StatusIndicator.tsx` — animated status text.
- `page.tsx` — two-column layout (sidebar + chat); polls `/ready` before rendering.

**"Verdure" design system** (dark forest, premium AI aesthetic):
- `tailwind.config.ts` — extended with `vd-*` color palette, `font-display` (Fraunces), `font-sans` (Outfit), `font-mono` (JetBrains Mono), custom keyframes (`fade-up`, `shimmer`, `cursor-blink`), box shadows (`accent-glow`, `card`, `card-hover`).
- `globals.css` — Google Fonts import, grain overlay via SVG filter, custom scrollbar, `::selection` accent, `prose-verdure` markdown styles with `–` accent bullets.
- All components restyled to `vd-*` tokens: dark backgrounds, mint-green `#6ee7a0` accent, Fraunces italic brand mark.

**React best practices applied:**
- `useCallback` on stable callbacks passed to child components.
- Removed dead `useEffect` for session sync (redundant with `key` prop remount).
- Hoisted `MD_COMPONENTS` to module level (prevents ReactMarkdown re-parsing on every streaming token).

### 4. Tests (`tests/`)

- `tests/fixtures/agent_test_samples.csv` — 302 rows: 100 text queries, 100 image queries, 102 follow-up turns (34 × 3-turn conversations). Schema: `test_type, test_id, turn, test_query, image_path, expected_results`.
- `tests/test_runner.py` — CSV-driven runner; groups turns by `test_id`; computes precision@k; outputs `tests/results/run_<timestamp>.csv`; exits 1 if crash rate > 5%.
- `tests/TEST_SAMPLES.md` — documents schema, how to add samples, run command, metrics tracked.

---
