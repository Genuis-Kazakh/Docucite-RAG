"""Pydantic request/response models."""
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    uploaded_at: datetime
    chunk_count: int


class UploadResponse(BaseModel):
    document: DocumentOut
    message: str = "Document uploaded and indexed."


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)


class Citation(BaseModel):
    source_id: int = Field(..., description="Number used in the answer, e.g. [1]")
    label: str = Field(..., description="Human-readable, e.g. 'policy.pdf, p. 4'")
    filename: str
    page: int | None = None
    section: str | None = None
    snippet: str
    score: float | None = None
    document_id: int | None = None
    chunk_id: int | None = None


class AskResponse(BaseModel):
    answer: str
    answerable: bool = Field(..., description="False if documents contain no answer")
    mode: str = Field("llm", description="'llm' or 'extractive' (no API key configured)")
    citations: list[Citation] = []


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    vectors: int | None = None


class ChunkOut(BaseModel):
    id: int
    chunk_index: int
    page: int | None = None
    section: str | None = None
    text: str


class DocumentChunks(BaseModel):
    document: DocumentOut
    chunks: list[ChunkOut]
