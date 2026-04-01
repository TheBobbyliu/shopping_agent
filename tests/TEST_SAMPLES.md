# Agent Test Samples

Automated test suite for the shopping AI agent, driven by a CSV sample list.

## Files

| File | Description |
|------|-------------|
| `tests/fixtures/agent_test_samples.csv` | 302-row test fixture (100 text, 100 image, 102 followup) |
| `tests/test_runner.py` | CSV-driven test runner |
| `tests/results/` | Output CSVs from each run (auto-created) |

---

## CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `test_type` | `text` \| `image` \| `followup` | Scenario category |
| `test_id` | string | Groups rows into conversations. Rows with the same `test_id` form one multi-turn session |
| `turn` | int | 1-indexed turn number within the conversation |
| `test_query` | string | User message text. Empty for image-only rows |
| `image_path` | string | Relative path to image file from repo root (empty for text queries) |
| `expected_results` | JSON array | Item IDs considered relevant; `[]` means no product assertion (e.g. follow-up turns 2–3) |

### Scenario types

**text** — Single-turn natural language product search.
- `test_query` is the user's search request
- `expected_results` contains item IDs that should appear in the agent's reply

**image** — Single-turn image-based product search.
- `image_path` points to a product image in the ABO dataset
- `test_query` is empty (the runner substitutes a generic message)
- `expected_results` contains at minimum the item that owns the image

**followup** — Three-turn conversation testing session memory.
- Turn 1: broad search query
- Turn 2: refinement ("I prefer black", "show me more...")
- Turn 3: detail request ("tell me more about the first one", "what's the material?")
- `expected_results` is `[]` for turns 2 and 3 — correctness is structural (agent recalls prior context), not result-set specific

---

## How to Run

### Prerequisites

```bash
# Backend must be running
conda run -n shopping uvicorn api.main:app --host 0.0.0.0 --port 8000

# Install requests if not already available
pip install requests
```

### Run all tests

```bash
python tests/test_runner.py
```

### Run only text queries (fastest)

```bash
python tests/test_runner.py --type text
```

### Run only image queries

```bash
python tests/test_runner.py --type image
```

### Run only multi-turn follow-up conversations

```bash
python tests/test_runner.py --type followup
```

### Run a quick smoke test (first 5 conversations)

```bash
python tests/test_runner.py --limit 5
```

### Point at a non-default backend

```bash
python tests/test_runner.py --api http://my-server:8000
```

### Full options

```
--api URL       Backend base URL (default: http://localhost:8000)
--type TYPE     Filter by test_type: text, image, or followup
--limit N       Max number of test conversations (respects multi-turn groupings)
--timeout S     Seconds per request (default: 180)
```

---

## Output

Results are written to `tests/results/run_<timestamp>.csv` with columns:

| Column | Description |
|--------|-------------|
| `test_id` | Test identifier |
| `test_type` | text / image / followup |
| `turn` | Turn number |
| `test_query` | First 80 chars of the query |
| `status_code` | HTTP status (200 = success, -1 = exception) |
| `elapsed_s` | Request duration in seconds |
| `precision` | Fraction of expected item IDs found in reply (1.0 when `expected_results` is empty) |
| `has_reply` | `yes` / `no` — whether the agent returned non-empty text |
| `error` | Error message if request failed |
| `reply_preview` | First 120 chars of the agent's reply |

### Exit code

- `0` — all tests passed (crash_rate ≤ 5%)
- `1` — crash_rate > 5%

---

## Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Precision** | `len(found ∩ expected) / len(expected)` | > 0.5 for text/image |
| **Session continuity** | Turns 2–3 of followup get non-empty replies | 100% |
| **Crash rate** | Requests with error or non-200 status | < 5% |

---

## How to Add New Tests

### Add a text query

Append a row to `agent_test_samples.csv`:

```csv
"text","txt_101","1","ergonomic standing desk","","[]"
```

Or add an entry to `tests/fixtures/queries_text.json` and re-run the CSV generator (if you have labeled `relevant_ids`).

### Add an image query

Pick any product image from `data/abo-images-small/images/small/` and append:

```csv
"image","img_101","1","","data/abo-images-small/images/small/ab/ab12cd34.jpg","[\"B01ITEMID01\"]"
```

The `item_id` that owns the image should always be in `expected_results`.

### Add a multi-turn conversation

Assign a new `test_id` and use sequential turn numbers:

```csv
"followup","followup_035","1","Show me outdoor chairs","","[]"
"followup","followup_035","2","Something weather-resistant","","[]"
"followup","followup_035","3","What material is the first option?","","[]"
```

---

## Notes

- Image tests require the ABO image dataset to be present at `data/abo-images-small/`
- The runner skips images that don't exist on disk (warns to stderr)
- Each followup conversation uses a real backend `session_id`, so the agent's LangGraph checkpointer must be running (PostgreSQL) for memory to work across turns
- If the checkpointer is unavailable, the agent falls back to stateless mode; followup turn 2 and 3 assertions will still pass (they only assert non-empty reply), but context recall won't be tested
