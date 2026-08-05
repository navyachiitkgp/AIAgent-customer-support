"""SQLite persistence for calls, turns, and coaching metrics."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from voiceiq.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id TEXT PRIMARY KEY,
    customer_id TEXT,
    representative_id TEXT,
    intent TEXT,
    summary TEXT,
    summary_redacted TEXT,
    keywords_json TEXT,
    sentiment_customer TEXT,
    sentiment_representative TEXT,
    sentiment_ending TEXT,
    model TEXT,
    audio_path TEXT,
    transcript_path TEXT,
    html_path TEXT,
    resolved INTEGER DEFAULT 1,
    escalated INTEGER DEFAULT 0,
    talk_ratio_customer REAL,
    talk_ratio_agent REAL,
    turn_count INTEGER,
    customer_turn_count INTEGER,
    agent_turn_count INTEGER,
    interruption_proxy INTEGER DEFAULT 0,
    duration_sec REAL,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL,
    turn_index INTEGER,
    speaker TEXT,
    text TEXT,
    text_redacted TEXT,
    start_sec REAL,
    end_sec REAL,
    sentiment TEXT,
    FOREIGN KEY(call_id) REFERENCES calls(call_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT,
    status TEXT,
    call_id TEXT,
    error TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_calls_intent ON calls(intent);
CREATE INDEX IF NOT EXISTS idx_calls_rep ON calls(representative_id);
CREATE INDEX IF NOT EXISTS idx_calls_created ON calls(created_at);
CREATE INDEX IF NOT EXISTS idx_turns_call ON turns(call_id);
"""


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    settings = get_settings()
    settings.ensure_dirs()
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_conn(db_path: Optional[Path] = None):
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_call(record: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    record = dict(record)
    record.setdefault("created_at", now)
    record["updated_at"] = now
    if isinstance(record.get("keywords"), list):
        record["keywords_json"] = json.dumps(record.pop("keywords"))
    elif "keywords_json" not in record:
        record["keywords_json"] = "[]"

    cols = [
        "call_id",
        "customer_id",
        "representative_id",
        "intent",
        "summary",
        "summary_redacted",
        "keywords_json",
        "sentiment_customer",
        "sentiment_representative",
        "sentiment_ending",
        "model",
        "audio_path",
        "transcript_path",
        "html_path",
        "resolved",
        "escalated",
        "talk_ratio_customer",
        "talk_ratio_agent",
        "turn_count",
        "customer_turn_count",
        "agent_turn_count",
        "interruption_proxy",
        "duration_sec",
        "created_at",
        "updated_at",
    ]
    values = [record.get(c) for c in cols]
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "call_id")
    sql = f"""
        INSERT INTO calls ({",".join(cols)}) VALUES ({placeholders})
        ON CONFLICT(call_id) DO UPDATE SET {updates}
    """
    with get_conn(db_path) as conn:
        conn.execute(sql, values)


def replace_turns(call_id: str, turns: Iterable[Dict[str, Any]], db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    rows = []
    for i, t in enumerate(turns):
        rows.append(
            (
                call_id,
                t.get("turn_index", i),
                t.get("speaker"),
                t.get("text"),
                t.get("text_redacted"),
                t.get("start_sec"),
                t.get("end_sec"),
                t.get("sentiment"),
            )
        )
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM turns WHERE call_id = ?", (call_id,))
        conn.executemany(
            """
            INSERT INTO turns
            (call_id, turn_index, speaker, text, text_redacted, start_sec, end_sec, sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def list_calls(
    *,
    intent: Optional[str] = None,
    representative_id: Optional[str] = None,
    unresolved_only: bool = False,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    init_db(db_path)
    clauses = ["1=1"]
    params: List[Any] = []
    if intent:
        clauses.append("intent = ?")
        params.append(intent)
    if representative_id:
        clauses.append("representative_id = ?")
        params.append(representative_id)
    if unresolved_only:
        clauses.append("resolved = 0")
    if date_from:
        clauses.append("date(created_at) >= date(?)")
        params.append(date_from)
    if date_to:
        clauses.append("date(created_at) <= date(?)")
        params.append(date_to)
    sql = f"SELECT * FROM calls WHERE {' AND '.join(clauses)} ORDER BY created_at DESC"
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["keywords"] = json.loads(d.get("keywords_json") or "[]")
        except json.JSONDecodeError:
            d["keywords"] = []
        out.append(d)
    return out


def get_call(call_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["keywords"] = json.loads(d.get("keywords_json") or "[]")
        except json.JSONDecodeError:
            d["keywords"] = []
        turns = conn.execute(
            "SELECT * FROM turns WHERE call_id = ? ORDER BY turn_index",
            (call_id,),
        ).fetchall()
        d["turns"] = [dict(t) for t in turns]
        return d


def count_by_intent(db_path: Optional[Path] = None) -> Dict[str, int]:
    init_db(db_path)
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT intent, COUNT(*) AS n FROM calls GROUP BY intent ORDER BY n DESC"
        ).fetchall()
    return {r["intent"] or "Unknown": r["n"] for r in rows}


def enqueue_job(source_path: str, db_path: Optional[Path] = None) -> int:
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO jobs (source_path, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (source_path, "queued", now, now),
        )
        return int(cur.lastrowid)


def update_job(job_id: int, **fields: Any) -> None:
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    cols = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [job_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", values)


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
