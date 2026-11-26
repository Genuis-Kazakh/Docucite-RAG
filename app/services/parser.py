"""Extract text from PDF / DOCX / MD / TXT into page- or section-tagged pieces."""
import re
from dataclasses import dataclass
from pathlib import Path


class ParseError(Exception):
    """Raised when a file cannot be read or contains no usable text."""


@dataclass
class Section:
    text: str
    page: int | None = None      # PDF page (1-based); 1 for plain txt
    label: str | None = None     # heading for DOCX/MD sections


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control chars; keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = _CONTROL.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise ParseError("PDF is password-protected.")
        pages = [(i, p.extract_text() or "") for i, p in enumerate(reader.pages, start=1)]
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"Cannot read PDF: {e}") from e
    return [Section(clean_text(t), page=i) for i, t in pages if clean_text(t)]


def _parse_docx(path: Path) -> list[Section]:
    from docx import Document

    try:
        doc = Document(str(path))
    except Exception as e:
        raise ParseError(f"Cannot read DOCX: {e}") from e

    sections: list[Section] = []
    label, buf = None, []

    def flush():
        text = clean_text("\n".join(buf))
        if text:
            sections.append(Section(text, label=label))

    for p in doc.paragraphs:
        if p.style and p.style.name.lower().startswith("heading"):
            flush()
            label, buf = p.text.strip() or None, []
        elif p.text.strip():
            buf.append(p.text)
    flush()
    return sections


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _split_markdown(text: str) -> list[Section]:
    matches = list(_MD_HEADING.finditer(text))
    if not matches:
        return [Section(clean_text(text), page=1)]
    sections: list[Section] = []
    pre = clean_text(text[: matches[0].start()])
    if pre:
        sections.append(Section(pre, page=1))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = clean_text(text[m.end():end])
        if body:
            sections.append(Section(body, label=m.group(2).strip()))
    return sections


def _read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ParseError("Unsupported text encoding.")


def parse_file(path: str | Path) -> list[Section]:
    path = Path(path)
    ext = path.suffix.lower()
    if not path.exists():
        raise ParseError("File not found.")

    if ext == ".pdf":
        sections = _parse_pdf(path)
    elif ext == ".docx":
        sections = _parse_docx(path)
    elif ext == ".md":
        sections = _split_markdown(_read_text(path))
    elif ext == ".txt":
        t = clean_text(_read_text(path))
        sections = [Section(t, page=1)] if t else []
    else:
        raise ParseError(f"Unsupported file type: {ext}")

    if not sections:
        raise ParseError("File contains no extractable text (empty or scanned image).")
    return sections
