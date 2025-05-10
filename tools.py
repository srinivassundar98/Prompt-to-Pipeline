"""
tools.py

Two LangChain tools for the SQLite NL→SQL demo:

1. run_sql   – Executes a read‑only query on demo.db and returns rows as JSON.
2. get_schema (optional) – Returns a one‑line description of the DB schema.

Both are synchronous functions, so they work with Zero‑Shot‑ReAct agents
that call tool.run() instead of tool.arun().
"""

import json
import sqlite3
from langchain.tools import tool

DB = "demo.db"  # path to the SQLite file ------------------------------------------------


# -----------------------------------------------------------------------
# Helper: pre‑compute a compact schema string once at import time
# -----------------------------------------------------------------------
def _schema_string() -> str:
    with sqlite3.connect(DB) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        parts = []
        for t in tables:
            cur.execute(f"PRAGMA table_info({t})")
            cols = ", ".join(f"{col[1]} {col[2]}" for col in cur.fetchall())
            parts.append(f"{t}({cols})")
        return "; ".join(parts)


SCHEMA_STR = _schema_string()


# -----------------------------------------------------------------------
# Tool 1: run_sql  (synchronous)
# -----------------------------------------------------------------------
@tool
def run_sql(query: str) -> str:
    """Execute a read‑only SQL query on demo.db and return rows as JSON."""
    try:
        with sqlite3.connect(DB) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query).fetchall()
            return json.dumps([dict(r) for r in rows])
    except Exception as e:
        return f"ERROR: {e}"


# -----------------------------------------------------------------------
# Tool 2: get_schema  (optional; agent doesn't strictly need it)
# -----------------------------------------------------------------------
@tool
def get_schema(_: str) -> str:
    """Return a compact description of the SQLite schema."""
    return SCHEMA_STR
