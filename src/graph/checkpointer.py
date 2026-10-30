# src/graph/checkpointer.py

"""
Checkpointer backend selection.

PostgresSaver when a DSN env var is set, SqliteSaver otherwise. Both
backends implement LangGraph's backend-agnostic checkpoint interface, so
the graph and every agent stay unchanged.

Why a connection pool for Postgres:
  - Streamlit runs sessions on threads; a psycopg3 connection is not
    thread-safe.
  - The pool reconnects transparently if the postgres container restarts
    mid-session.

Why setup() runs on its own autocommit connection:
  - LangGraph's migrations include CREATE INDEX CONCURRENTLY, which
    cannot run inside a transaction block.
  - A session-level advisory lock serializes simultaneous first starts
    of multiple app instances against the same database, avoiding
    CREATE TABLE races.
"""

import os

from langgraph.checkpoint.base import BaseCheckpointSaver

_SETUP_LOCK_KEY = 0x6E6F746573  # "notes", arbitrary but stable


def _log(msg: str) -> None:
    print(f"[Checkpointer] {msg}")


def _postgres_checkpointer(dsn: str) -> BaseCheckpointSaver:
    """Build a PostgresSaver backed by a connection pool."""
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver

    setup_conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
    try:
        setup_conn.execute("SELECT pg_advisory_lock(%s)", (_SETUP_LOCK_KEY,))
        try:
            PostgresSaver(setup_conn).setup()
        finally:
            setup_conn.execute("SELECT pg_advisory_unlock(%s)", (_SETUP_LOCK_KEY,))
    finally:
        setup_conn.close()

    pool = ConnectionPool(
        conninfo=dsn,
        min_size=0,
        max_size=4,
        open=True,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    _log("Postgres checkpointer ready (schema verified)")
    return PostgresSaver(pool)


def _sqlite_checkpointer(db_path: str) -> BaseCheckpointSaver:
    """Legacy SQLite path, unchanged from pre-migration behavior."""
    import sqlite3
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver

    Path("data").mkdir(exist_ok=True)
    if db_path == "data/checkpoints.db":
        db_path = os.getenv("CHECKPOINT_DB", db_path)

    # IMPORTANT: create the connection directly, not via context manager.
    # SqliteSaver.from_conn_string() returns a context manager; using it
    # with `with` would close the connection when the block exits. The
    # graph lives longer than build_graph(), so the connection must stay
    # open for the process lifetime.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    _log(f"SQLite fallback active: {db_path}")
    return SqliteSaver(conn)


def make_checkpointer(dsn_env_var: str, sqlite_default: str) -> BaseCheckpointSaver:
    """
    Choose and build the checkpoint backend.

    dsn_env_var:    env var holding the Postgres DSN (CHECKPOINT_DB_URL
                    for the terminal graph, CHECKPOINT_DB_URL_UI for the
                    Streamlit graph)
    sqlite_default: SQLite path used only when the DSN is unset/empty
    """
    dsn = os.getenv(dsn_env_var, "").strip()
    if not dsn:
        return _sqlite_checkpointer(sqlite_default)

    try:
        return _postgres_checkpointer(dsn)
    except Exception as e:
        raise RuntimeError(
            f"Could not initialise the Postgres checkpointer from "
            f"{dsn_env_var}. Postgres is configured but not reachable: {e}\n"
            f"Start it with:  make postgres\n"
            f"Or clear {dsn_env_var} to fall back to SQLite checkpoints."
        ) from e
