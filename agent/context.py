"""
PostgreSQL-backed conversation context for the shopping agent.

Uses LangGraph's PostgresSaver checkpointer so every agent invocation
within the same session_id resumes from persisted state.

Usage:
    from context import make_checkpointer, thread_config
    cp = make_checkpointer(DATABASE_URL)
    agent = create_agent(checkpointer=cp)
    agent.invoke({"messages": [...]}, config=thread_config(session_id))
"""
from __future__ import annotations

import os


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://shopping:shopping@localhost:5432/shopping",
)

_conn = None  # module-level reference so close_checkpointer() can reach it


def make_checkpointer(url: str | None = None):
    """
    Create and return a PostgresSaver checkpointer backed by a persistent connection.
    Creates the required tables if they don't exist yet.
    Call close_checkpointer() on process shutdown to release the connection.
    """
    global _conn
    import psycopg
    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres import PostgresSaver

    _conn = psycopg.connect(
        url or DATABASE_URL,
        autocommit=True,
        row_factory=dict_row,
    )
    cp = PostgresSaver(_conn)
    cp.setup()   # CREATE TABLE IF NOT EXISTS …
    return cp


def close_checkpointer():
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def thread_config(session_id: str) -> dict:
    """Return the LangGraph config dict for a given session."""
    return {"configurable": {"thread_id": session_id}}
