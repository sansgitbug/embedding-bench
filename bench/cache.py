"""On-disk embedding cache (SQLite), one row per text.

Per-text rows mean an interrupted or quota-limited indexing run resumes where
it stopped. Each row points at the API call that produced it, so token counts
and timings survive reruns: a cached rerun reports the originally measured
numbers instead of re-spending.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL, kind TEXT NOT NULL, n_texts INTEGER NOT NULL,
    tokens INTEGER NOT NULL, tokens_source TEXT NOT NULL,
    elapsed_s REAL NOT NULL, wait_s REAL NOT NULL, retries INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    key TEXT PRIMARY KEY, call_id INTEGER NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL
);
"""

CALL_FIELDS = ("n_texts", "tokens", "tokens_source", "elapsed_s", "wait_s", "retries", "created_at")


class EmbeddingCache:
    def __init__(self, path: Path, namespace: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(SCHEMA)
        self._namespace = namespace

    def _key(self, model: str, kind: str, text: str) -> str:
        raw = "\x00".join((self._namespace, model, kind, text))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model: str, kind: str, text: str) -> tuple[np.ndarray, int] | None:
        """Return (vector, call_id) or None if this text was never embedded."""
        row = self._db.execute(
            "SELECT vector, call_id FROM embeddings WHERE key = ?",
            (self._key(model, kind, text),),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32), row[1]

    def put_call(
        self,
        model: str,
        kind: str,
        texts: list[str],
        vectors: np.ndarray,
        *,
        tokens: int,
        tokens_source: str,
        elapsed_s: float,
        wait_s: float,
        retries: int,
    ) -> int:
        """Store one API call and the vectors it returned. Returns the call_id."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(f"expected {len(texts)} vectors, got shape {vectors.shape}")
        with self._db:
            cur = self._db.execute(
                "INSERT INTO calls (model, kind, n_texts, tokens, tokens_source, elapsed_s, "
                "wait_s, retries, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (model, kind, len(texts), tokens, tokens_source, elapsed_s, wait_s, retries, time.time()),
            )
            call_id = cur.lastrowid
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings (key, call_id, dim, vector) VALUES (?,?,?,?)",
                [
                    (self._key(model, kind, t), call_id, vectors.shape[1], v.tobytes())
                    for t, v in zip(texts, vectors, strict=True)
                ],
            )
        return call_id

    def call(self, call_id: int) -> dict:
        row = self._db.execute(
            f"SELECT {', '.join(CALL_FIELDS)} FROM calls WHERE call_id = ?", (call_id,)
        ).fetchone()
        return dict(zip(CALL_FIELDS, row, strict=True))

    def close(self) -> None:
        self._db.close()
