"""Overlapping chunker that prefers paragraph/sentence boundaries and keeps page/section tags."""
from dataclasses import dataclass

from app.services.parser import Section

_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass
class Chunk:
    chunk_index: int
    text: str
    page: int | None = None
    section: str | None = None

    def as_dict(self) -> dict:
        return {"chunk_index": self.chunk_index, "text": self.text,
                "page": self.page, "section": self.section}


def _find_cut(text: str, start: int, hard_end: int, min_len: int) -> int:
    """Best cut in (start+min_len, hard_end]; falls back to hard_end."""
    window_start = start + min_len
    for sep in _BREAKS:
        pos = text.rfind(sep, window_start, hard_end)
        if pos != -1:
            return pos + len(sep)
    return hard_end


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    pieces: list[str] = []
    start, n = 0, len(text)
    min_len = int(chunk_size * 0.6)  # don't cut too early

    while start < n:
        hard_end = min(start + chunk_size, n)
        end = n if hard_end == n else _find_cut(text, start, hard_end, min_len)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # guarantee progress
    return pieces


def chunk_sections(sections: list[Section], chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        for piece in split_text(sec.text, chunk_size, overlap):
            chunks.append(Chunk(len(chunks), piece, sec.page, sec.label))
    return chunks
