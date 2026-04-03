# Warehouse CLI — Design Spec

**Date:** 2026-04-01  
**Status:** Approved

---

## Overview

A command-line tool for operational staff to manage the product database (Elasticsearch). Supports adding/deleting individual items and checking database contents.

---

## Files

| File | Action |
|---|---|
| `preprocessing/warehouse.py` | New — CLI entry point and all command logic |
| `api/main.py` | Modified — add `POST /embed` endpoint |
| `tests/test_warehouse.py` | New — unit + integration tests |

---

## ES Indices

Only the existing `products` index is used. No new indices required.

---

## Commands

### `warehouse add <json_file>`

**Input JSON** (array of objects):
```json
[
  {
    "item_id": "B123456789",
    "description": "Samsung 65-inch 4K QLED TV with HDR",
    "image_path": "data/images/tv.jpg",
    "name": "Samsung QN65Q80C",
    "brand": "Samsung",
    "color": "Black",
    "voltage": "120V"
  }
]
```

**Required fields per item:** `item_id`, `description`, `image_path`.  
All other fields pass through to ES as-is.

**Flow:**
1. Parse and validate JSON — error on malformed JSON or missing required fields (show which items/fields fail)
2. Detect duplicate `item_id`s within the input file → abort with error listing the duplicates
3. Query ES with `get_indexed_ids()` to find already-indexed IDs → collect as skip set
4. Print skip warnings for each: `"[skip] B123456789 — already indexed"`
5. For each new item (in order):
   - Print: `"[1/5] B000111222 — embedding..."`
   - `POST /embed` with `{"text": "<description>", "image_path": "<image_path>"}` → `{description_vector, image_vector}`
   - Add vectors to the document and index into ES
   - Print: `"[1/5] B000111222 — done"`
6. Final summary: `"Added: 5, Skipped: 2, Errors: 0"`

---

### `warehouse delete <json_file>`

**Input JSON** (array of objects):
```json
[{"item_id": "B123456789"}, {"item_id": "B987654321"}]
```

**Flow:**
1. Parse and validate JSON — error on missing `item_id`
2. Detect duplicate `item_id`s within input → abort with error
3. Query ES to find which IDs exist → warn for each missing: `"[warn] B000000000 — not found, skipping"`
4. Delete found IDs from ES
5. Summary: `"Deleted: 4, Not found: 1"`

---

### `warehouse check`

Flags (mutually exclusive):
- `--item-id <id>` — fetch single item by ID
- `--group <name> [--count N]` — fetch items by category, default count 10
- `--count N` — fetch N items across all groups (no ordering guarantee), default 10

Output: pretty-printed JSON to stdout. Vectors (`description_vector`, `image_vector`) are always stripped.

---

## `/embed` Endpoint (api/main.py)

```
POST /embed
Request:  {"text": "...", "image_path": "..."}
Response: {"description_vector": [...], "image_vector": [...]}
```

- Calls `_get_embedding_client()` singleton — no new model load
- `image_path` must be a valid local path accessible to the API server
- Returns HTTP 400 if `image_path` does not exist
- Returns HTTP 503 if embedding client is not yet ready (startup not complete)

---

## Error Handling Summary

| Situation | Behaviour |
|---|---|
| JSON file not found | Exit with error |
| Malformed JSON | Exit with error, show parse message |
| Missing required field | Exit with error, list affected items |
| Duplicate `item_id` in input | Exit with error, list duplicates |
| Item already indexed (add) | Skip with warning, continue |
| Item not found (delete) | Warn, continue |
| Embed API unreachable | Exit with error, show URL |
| ES unreachable | Exit with error |

---

## Tests

### Unit tests (no ES, no API — mocked)

- `test_validate_add_json` — missing required fields caught correctly
- `test_duplicate_item_ids_in_input` — duplicates cause abort
- `test_skip_already_indexed` — already-indexed items are skipped, not errored
- `test_missing_item_delete_warns` — missing IDs on delete produce warnings, not errors

### Integration tests (`@pytest.mark.api`, require ES + API)

- `test_add_items` — add 2 items → verify appear in ES with vectors
- `test_add_skips_existing` — re-add same items → verify skip warnings
- `test_delete_items` — delete 1 existing + 1 nonexistent → verify deletion + warning
- `test_check_item_id` — returns correct fields, no vectors
- `test_check_group` — filters by category, respects --count
- `test_check_count` — returns N items across all groups
