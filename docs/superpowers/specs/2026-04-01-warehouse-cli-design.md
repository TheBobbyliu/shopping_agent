# Warehouse CLI — Design Spec

**Date:** 2026-04-01  
**Status:** Approved

---

## Overview

A command-line tool for operational staff to manage the product database (Elasticsearch). Supports creating/deleting product groups, adding/deleting individual items, and checking database contents.

---

## Files

| File | Action |
|---|---|
| `preprocessing/warehouse.py` | New — CLI entry point and all command logic |
| `api/main.py` | Modified — add `POST /embed` endpoint |
| `tests/test_warehouse.py` | New — unit + integration tests |

---

## ES Indices

| Index | Purpose |
|---|---|
| `products` (existing) | Product documents |
| `product_groups` (new) | Group registry — one document per group, `_id = group_name` |

`product_groups` document schema:
```json
{ "name": "ELECTRONICS" }
```

---

## Commands

### `warehouse create-group <name>`

1. Check if a document with `_id = name` already exists in `product_groups`  
2. If yes → exit with error: `"Group '<name>' already exists."`  
3. If no → index `{"name": name}` into `product_groups` with `_id = name`  
4. Print: `"Group '<name>' created."`

---

### `warehouse delete-group <name>`

1. Check group exists in `product_groups` → error if not found  
2. Count items in `products` where `category == name`  
3. Print: `"Found <N> items in group '<name>'."`  
4. Prompt: `"Are you sure you want to delete the group and all its items? [y/N]: "`  
5. If not confirmed → abort with `"Aborted."`  
6. Delete all matching items from `products` via `delete_by_query`  
7. Delete group document from `product_groups`  
8. Print: `"Group '<name>' and <N> items deleted."`

---

### `warehouse add <json_file>`

**Input JSON** (array of objects):
```json
[
  {
    "item_id": "B123456789",
    "description": "Samsung 65-inch 4K QLED TV with HDR",
    "image_path": "data/images/tv.jpg",
    "target_group_name": "ELECTRONICS",
    "name": "Samsung QN65Q80C",
    "brand": "Samsung",
    "color": "Black",
    "voltage": "120V"
  }
]
```

**Required fields per item:** `item_id`, `description`, `image_path`, `target_group_name`.  
All other fields pass through to ES as-is.

**Flow:**
1. Parse and validate JSON — error on malformed JSON or missing required fields (show which items/fields fail)
2. Detect duplicate `item_id`s within the input file → abort with error listing the duplicates
3. Verify `target_group_name` exists in `product_groups` for every item → error if any group is unknown
4. Query ES with `get_indexed_ids()` to find already-indexed IDs → collect as skip set
5. Print skip warnings for each: `"[skip] B123456789 — already indexed"`
6. For each new item (in order):
   - Print: `"[1/5] B000111222 — embedding..."`
   - `POST /embed` with `{"text": "<description>", "image_path": "<image_path>"}` → `{description_vector, image_vector}`
   - Add vectors + `category = target_group_name` to the document, remove `target_group_name` key
   - Index into ES
   - Print: `"[1/5] B000111222 — done"`
7. Final summary: `"Added: 5, Skipped: 2, Errors: 0"`

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
- `--group <name> [--count N]` — fetch items in group, default count 10
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
| Unknown `target_group_name` | Exit with error |
| Group already exists (create-group) | Exit with error |
| Group not found (delete-group) | Exit with error |
| Item already indexed (add) | Skip with warning, continue |
| Item not found (delete) | Warn, continue |
| Embed API unreachable | Exit with error, show URL |
| ES unreachable | Exit with error |

---

## Tests

### Unit tests (no ES, no API — mocked)

- `test_validate_add_json` — missing required fields caught correctly
- `test_duplicate_item_ids_in_input` — duplicates cause abort
- `test_unknown_group_rejected` — unknown target_group_name errors
- `test_skip_already_indexed` — already-indexed items are skipped, not errored
- `test_missing_item_delete_warns` — missing IDs on delete produce warnings, not errors

### Integration tests (`@pytest.mark.api`, require ES + API)

- `test_create_group` — create new group → verify in `product_groups`
- `test_create_group_duplicate` — create same group twice → second call errors
- `test_delete_group_with_confirmation` — delete group → verify items removed
- `test_delete_group_abort` — answer N at prompt → verify nothing deleted
- `test_add_items` — add 2 items → verify appear in ES with vectors
- `test_add_skips_existing` — re-add same items → verify skip warnings
- `test_delete_items` — delete 1 existing + 1 nonexistent → verify deletion + warning
- `test_check_item_id` — returns correct fields, no vectors
- `test_check_group` — filters by category, respects --count
- `test_check_count` — returns N items across all groups
