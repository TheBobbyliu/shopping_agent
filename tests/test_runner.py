"""
CSV-driven test runner for the shopping agent.

Loads tests/fixtures/agent_test_samples.csv and runs each row against the
POST /chat API. Multi-turn conversations are grouped by test_id and share a
session_id across turns.

Usage:
    python tests/test_runner.py [--api URL] [--type text|image|followup] [--limit N] [--timeout S]
    python tests/test_runner.py --resume tests/results/run_<timestamp>.csv

Output:
    tests/results/run_<timestamp>.csv
    Exit code 1 if crash_rate > 5%
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
FIXTURE  = Path(__file__).parent / "fixtures" / "agent_test_samples.csv"
RESULTS  = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

ITEM_ID_RE = re.compile(r"\b(B[A-Z0-9]{9})\b")

DEFAULT_API     = "http://localhost:8000"
DEFAULT_TIMEOUT = 180  # seconds per request

FIELDNAMES = ["test_id","test_type","turn","test_query","status_code","elapsed_s","precision","has_reply","error","reply_preview"]


def precision_at_k(found: list[str], expected: list[str]) -> float:
    """Fraction of expected items found anywhere in the reply."""
    if not expected:
        return 1.0  # nothing to find → trivially correct
    return len(set(found) & set(expected)) / len(expected)


def load_image_b64(image_path: str) -> str | None:
    """Read an image file and return its base64 encoding."""
    path = BASE_DIR / image_path
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def load_resume(resume_path: str, all_rows: list[dict]) -> tuple[set[str], list[dict]]:
    """
    Load a previous results CSV and return:
      - set of test_ids where ALL turns are already recorded (safe to skip)
      - list of previously recorded result dicts (to carry forward)
    """
    path = Path(resume_path)
    if not path.exists():
        print(f"[WARN] Resume file not found: {resume_path}", file=sys.stderr)
        return set(), []

    prior = list(csv.DictReader(path.open(encoding="utf-8")))
    completed_pairs = {(r["test_id"], r["turn"]) for r in prior}

    # Determine which turns each test_id needs
    turns_needed: dict[str, set[str]] = defaultdict(set)
    for r in all_rows:
        turns_needed[r["test_id"]].add(r["turn"])

    fully_done = {
        tid for tid, turns in turns_needed.items()
        if all((tid, t) in completed_pairs for t in turns)
    }
    return fully_done, prior


def run(api: str, test_type_filter: str | None, limit: int | None, timeout: int, resume: str | None):
    rows = list(csv.DictReader(FIXTURE.open(encoding="utf-8")))

    if test_type_filter:
        rows = [r for r in rows if r["test_type"] == test_type_filter]
    if limit:
        # Respect multi-turn groupings: keep complete conversations
        kept_ids: set[str] = set()
        kept: list[dict] = []
        for r in rows:
            if r["test_id"] not in kept_ids:
                if len(kept_ids) >= limit:
                    break
                kept_ids.add(r["test_id"])
            kept.append(r)
        rows = kept

    # ── Resume: load previously completed results ──────────────────────────
    completed_ids: set[str] = set()
    prior_results: list[dict] = []
    if resume:
        completed_ids, prior_results = load_resume(resume, rows)
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} test_id(s) already complete — skipping.")

    rows_to_run = [r for r in rows if r["test_id"] not in completed_ids]
    skipped = len(rows) - len(rows_to_run)

    # ── Create output file; seed with prior results ─────────────────────────
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path  = RESULTS / f"run_{timestamp}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in prior_results:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})

    print(f"Output: {out_path}")
    if skipped:
        print(f"Skipped {skipped} row(s) from {len(completed_ids)} completed test_id(s).")
    print(f"Running {len(rows_to_run)} row(s) …")

    # ── Main loop ──────────────────────────────────────────────────────────
    sessions: dict[str, str] = {}   # test_id → session_id
    new_results: list[dict] = []

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)

        for i, row in enumerate(rows_to_run, 1):
            test_id   = row["test_id"]
            turn      = int(row["turn"])
            test_type = row["test_type"]
            query     = row["test_query"]
            img_path  = row["image_path"]
            expected  = json.loads(row["expected_results"] or "[]")

            session_id = sessions.get(test_id) if turn > 1 else None

            payload: dict = {"message": query or "Find products similar to this image."}
            if session_id:
                payload["session_id"] = session_id
            if img_path:
                b64 = load_image_b64(img_path)
                if b64:
                    payload["image_b64"] = b64
                else:
                    print(f"  [WARN] Image not found: {img_path}", file=sys.stderr)

            t0 = time.perf_counter()
            status_code = -1
            reply = ""
            error = ""
            precision = 0.0

            try:
                resp = requests.post(f"{api}/chat", json=payload, timeout=timeout)
                status_code = resp.status_code
                elapsed = round(time.perf_counter() - t0, 2)

                if resp.ok:
                    data = resp.json()
                    reply = data.get("reply", "")

                    if turn == 1:
                        sessions[test_id] = data.get("session_id", "")

                    found_ids = ITEM_ID_RE.findall(reply)
                    precision = precision_at_k(found_ids, expected)
                else:
                    error = resp.text[:200]
                    elapsed = round(time.perf_counter() - t0, 2)

            except requests.exceptions.Timeout:
                elapsed = timeout
                error = f"Timeout after {timeout}s"
            except Exception as e:
                elapsed = round(time.perf_counter() - t0, 2)
                error = str(e)

            result = {
                "test_id":       test_id,
                "test_type":     test_type,
                "turn":          turn,
                "test_query":    query[:80],
                "status_code":   status_code,
                "elapsed_s":     elapsed,
                "precision":     round(precision, 3),
                "has_reply":     "yes" if reply else "no",
                "error":         error[:200],
                "reply_preview": reply[:120].replace("\n", " "),
            }
            new_results.append(result)

            # Write and flush immediately so progress survives interruption
            w.writerow(result)
            f.flush()

            status_str = "OK" if not error and reply else ("NO_REPLY" if not error else "ERR")
            print(f"  [{i:3d}/{len(rows_to_run)}] {test_id} t{turn} {status_str} {elapsed}s P={precision:.2f}"
                  + (f" ERR:{error[:60]}" if error else ""))

    # ── Summary stats (new results only) ───────────────────────────────────
    all_results = prior_results + new_results  # type: ignore[operator]
    total   = len(all_results)
    crashes = sum(1 for r in all_results if r.get("error") or str(r.get("status_code")) not in ("-1", "200"))
    no_reply = sum(1 for r in all_results if r.get("has_reply") == "no" and not r.get("error"))
    precisions = [float(r["precision"]) for r in all_results if r.get("precision") not in (None, "")]
    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    crash_rate = crashes / total if total else 0.0

    print(f"\n{'='*50}")
    print(f"Results: {out_path}")
    print(f"Total rows:    {total}  (new: {len(new_results)}, carried: {len(prior_results)})")
    print(f"Crashes:       {crashes} ({crash_rate:.1%})")
    print(f"No reply:      {no_reply}")
    print(f"Avg precision: {avg_p:.3f}")
    print(f"{'='*50}")

    if crash_rate > 0.05:
        print("FAIL: crash_rate > 5%", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shopping agent CSV test runner")
    parser.add_argument("--api",     default=DEFAULT_API,  help="Backend base URL")
    parser.add_argument("--type",    default=None,          help="Filter by test_type (text/image/followup)")
    parser.add_argument("--limit",   type=int, default=None, help="Max number of test conversations to run")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Seconds per request")
    parser.add_argument("--resume",  default=None, metavar="FILE",
                        help="Path to a previous results CSV to resume from (skips completed test_ids)")
    args = parser.parse_args()

    print(f"Running agent tests against {args.api}")
    print(f"CSV: {FIXTURE} ({sum(1 for _ in FIXTURE.open())-1} rows)")
    run(api=args.api, test_type_filter=args.type, limit=args.limit, timeout=args.timeout, resume=args.resume)
