# Repository Guidelines

## Project Structure & Module Organization
The repository root is the parent directory of this file. `api/main.py` exposes the FastAPI server. `agent/` contains the LangGraph agent, search logic, tools, context persistence, and pipeline monitoring. `preprocessing/` holds the ABO data pipeline (`extract.py`, `embed.py`, `index.py`, `pipeline.py`). `frontend-next/` is the active Next.js UI, and `tests/` contains pytest suites plus fixtures. Treat `uploads/`, `logs/`, and large files under `data/` as generated/runtime assets, not source.

## Build, Test, and Development Commands
Run these from the repo root:

- `cd api && uvicorn main:app --reload --port 8000` starts the API on `localhost:8000`.
- `cd frontend-next && npm run dev` starts the Next.js app on `localhost:3000`.
- `cd frontend-next && npm run build` verifies the production frontend build.
- `pytest` runs the default Python test suite under `tests/`.
- `pytest -m api -v` runs integration tests against live services.
- `pytest -m "not slow"` skips long-running checks.
- `python preprocessing/pipeline.py --index-name products_test --recreate --limit 500 --stratify` rebuilds the test search index.

## Coding Style & Naming Conventions
Use 4-space indentation in Python, keep type hints where the code already uses them, and follow existing `snake_case` names for functions, modules, and variables. Reserve `PascalCase` for classes such as Pydantic models. In the frontend, keep React components in `PascalCase` files under `frontend-next/components/` and shared helpers under `frontend-next/lib/`. No formatter or linter config is checked in, so match surrounding style and keep route handlers thin by pushing reusable logic into `agent/` or `preprocessing/`.

## Testing Guidelines
Name test files `test_*.py` and test functions `test_*`. Reuse pytest markers from `pytest.ini`: `api`, `performance`, and `slow`. `tests/conftest.py` forces `ES_INDEX=products_test`; do not point automated tests at the production index. Put reusable sample inputs in `tests/fixtures/` and prefer targeted files such as `tests/test_chat_interface.py` or `tests/test_frontend.py`.

## Commit & Pull Request Guidelines
Recent commits use short, lowercase summaries like `fix bugs` and `version 1`. Keep subjects brief and imperative, but make them more specific when possible, for example `search: tighten reranker thresholds`. Pull requests should describe user-visible changes, note any required `.env` or infrastructure updates, and list exact verification commands. Include screenshots for `frontend-next/` UI changes and sample request/response details for API contract changes.

## Security & Configuration Tips
Do not commit `.env`, catalog data, uploads, logs, or frontend build artifacts. Core environment variables used throughout the repo include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `ELASTICSEARCH_URL`, `DATABASE_URL`, `ES_INDEX`, `ES_TEST_INDEX`, and `API_BASE_URL`.
