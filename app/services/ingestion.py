"""Parse -> chunk -> store metadata -> embed. Shared by the API and the eval script."""
import logging
from pathlib import Path

from app import database as db
from app.config import get_settings
from app.services.chunker import chunk_sections
from app.services.parser import parse_file
from app.services.vector_store import get_vector_store

log = logging.getLogger("docucite.ingest")


def ingest_file(path: Path, filename: str) -> dict:
    """Raises ParseError for unreadable/empty files. Keeps SQLite and FAISS consistent."""
    s = get_settings()
    chunks = chunk_sections(parse_file(path), s.chunk_size, s.chunk_overlap)
    if not chunks:
        from app.services.parser import ParseError
        raise ParseError("File contains no extractable text.")

    doc_id = db.insert_document(filename, str(path), path.suffix.lower().lstrip("."))
    try:
        ids = db.insert_chunks(doc_id, [c.as_dict() for c in chunks])
        get_vector_store().add(ids, [c.text for c in chunks])
    except Exception:
        log.exception("Ingestion failed; rolling back document %s", doc_id)
        db.delete_document(doc_id)
        raise
    return db.get_document(doc_id)


def remove_document(doc_id: int) -> bool:
    doc = db.get_document(doc_id)
    if not doc:
        return False
    get_vector_store().remove(db.get_chunk_ids(doc_id))
    db.delete_document(doc_id)
    Path(doc["stored_path"]).unlink(missing_ok=True)
    return True


def reindex_all() -> int:
    """Rebuild the FAISS index from SQLite chunks (after changing embedding model)."""
    store = get_vector_store()
    store.reset()
    with db.get_conn() as conn:
        rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    for i in range(0, len(rows), 64):
        batch = rows[i:i + 64]
        store.add([r["id"] for r in batch], [r["text"] for r in batch])
    return len(rows)
