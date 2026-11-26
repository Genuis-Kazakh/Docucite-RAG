import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app import database as db
from app.config import get_settings
from app.schemas import DocumentChunks, DocumentOut, UploadResponse
from app.services.ingestion import ingest_file, reindex_all, remove_document
from app.services.parser import ParseError

log = logging.getLogger("docucite.api")
router = APIRouter(prefix="/api/v1")


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", Path(name).name, flags=re.UNICODE)[:120] or "file"


@router.post("/documents/upload", response_model=UploadResponse, status_code=201, tags=["documents"])
def upload_document(file: UploadFile):
    s = get_settings()
    original = file.filename or "file"
    ext = Path(original).suffix.lower()
    if ext not in s.allowed_extensions:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {', '.join(s.allowed_extensions)}")

    data = file.file.read(s.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {s.max_upload_mb} MB.")
    if not data.strip():
        raise HTTPException(422, "File is empty.")

    s.ensure_dirs()
    stored = s.upload_dir / f"{uuid.uuid4().hex[:8]}_{_safe_name(original)}"
    stored.write_bytes(data)
    try:
        doc = ingest_file(stored, original)
    except ParseError as e:
        stored.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    except Exception:
        stored.unlink(missing_ok=True)
        log.exception("Upload failed")
        raise HTTPException(500, "Failed to process the document.")
    return UploadResponse(document=DocumentOut(**doc))


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents():
    return db.list_documents()


@router.get("/documents/{doc_id}/chunks", response_model=DocumentChunks, tags=["documents"])
def document_chunks(doc_id: int):
    """Indexed passages of one document (used by the source viewer in the UI)."""
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    return DocumentChunks(document=DocumentOut(**doc), chunks=db.get_document_chunks(doc_id))


@router.delete("/documents/{doc_id}", tags=["documents"])
def delete_document(doc_id: int):
    if not remove_document(doc_id):
        raise HTTPException(404, "Document not found.")
    return {"deleted": doc_id}


@router.post("/admin/reindex", tags=["admin"])
def reindex():
    """Rebuild the FAISS index from stored chunks (use after changing the embedding model)."""
    return {"reindexed_chunks": reindex_all()}
