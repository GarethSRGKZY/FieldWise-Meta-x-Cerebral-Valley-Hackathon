"""Offline escalation queue (SQLite, standard library only).

Only redacted text and EXIF-stripped photos are ever stored here, so nothing
personal waits on disk for the network to come back.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL,              -- 'ask' | 'diagnose'
    payload TEXT NOT NULL,           -- JSON: redacted inputs + local answer
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | done | failed
    attempts INTEGER NOT NULL DEFAULT 0,
    result TEXT,
    updated_at REAL
);
"""


class EscalationQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, kind: str, payload: dict) -> str:
        qid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute("INSERT INTO escalations (id, created_at, kind, payload) VALUES (?,?,?,?)",
                      (qid, time.time(), kind, json.dumps(payload)))
        return qid

    def pending(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM escalations WHERE status='pending' ORDER BY created_at LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def all(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM escalations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def mark(self, qid: str, status: str, result: dict | None = None) -> None:
        with self._conn() as c:
            c.execute("UPDATE escalations SET status=?, result=?, attempts=attempts+1, updated_at=? WHERE id=?",
                      (status, json.dumps(result) if result is not None else None, time.time(), qid))

    def bump(self, qid: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE escalations SET attempts=attempts+1, updated_at=? WHERE id=?", (time.time(), qid))

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        d["result"] = json.loads(d["result"]) if d.get("result") else None
        return d
