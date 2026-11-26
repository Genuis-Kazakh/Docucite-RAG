"""SQLite metadata storage: documents + chunks.

chunks.id doubles as the FAISS vector id (Day 2), so deleting a document
returns the exact ids to remove from the index.
"""
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page        INTEGER,
    section     TEXT,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
"""


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    settings = get_settings()
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_document(filename: str, stored_path: str, file_type: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (filename, stored_path, file_type) VALUES (?, ?, ?)",
            (filename, stored_path, file_type),
        )
        return int(cur.lastrowid)


def insert_chunks(document_id: int, chunks: list[dict]) -> list[int]:
    """chunks: dicts with chunk_index, page, section, text. Returns new chunk ids."""
    ids: list[int] = []
    with get_conn() as conn:
        for c in chunks:
            cur = conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, page, section, text) "
                "VALUES (?, ?, ?, ?, ?)",
                (document_id, c["chunk_index"], c.get("page"), c.get("section"), c["text"]),
            )
            ids.append(int(cur.lastrowid))
    return ids


def list_documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT d.id, d.filename, d.file_type, d.uploaded_at, COUNT(c.id) AS chunk_count "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "GROUP BY d.id ORDER BY d.id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT d.id, d.filename, d.stored_path, d.file_type, d.uploaded_at, "
            "COUNT(c.id) AS chunk_count FROM documents d "
            "LEFT JOIN chunks c ON c.document_id = d.id WHERE d.id = ? GROUP BY d.id",
            (doc_id,),
        ).fetchone()
        return dict(row) if row else None


def get_chunk_ids(doc_id: int) -> list[int]:
    with get_conn() as conn:
        rows = conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,)).fetchall()
        return [r["id"] for r in rows]


def get_chunks_by_ids(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT c.id, c.document_id, c.page, c.section, c.text, c.chunk_index, d.filename "
            f"FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE c.id IN ({marks})",
            ids,
        ).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]  # keep caller's (ranked) order


def get_document_chunks(doc_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, chunk_index, page, section, text FROM chunks "
            "WHERE document_id = ? ORDER BY chunk_index", (doc_id,)).fetchall()
        return [dict(r) for r in rows]


def delete_document(doc_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return cur.rowcount > 0


def count_stats() -> dict:
    with get_conn() as conn:
        d = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        c = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {"documents": d, "chunks": c}
