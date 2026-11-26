import pytest

from app.services.parser import ParseError, parse_file
from app.services.vector_store import HashEmbedder, VectorStore


def test_parse_md_sections(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# A\nfoo bar\n## B\nbaz qux", encoding="utf-8")
    secs = parse_file(p)
    assert [s.label for s in secs] == ["A", "B"]


def test_parse_docx(tmp_path):
    from docx import Document
    d = Document()
    d.add_heading("Rules", 1)
    d.add_paragraph("No phones allowed.")
    p = tmp_path / "a.docx"
    d.save(p)
    assert parse_file(p)[0].label == "Rules"


def test_parse_empty_and_unknown(tmp_path):
    (tmp_path / "e.txt").write_text("  ")
    with pytest.raises(ParseError):
        parse_file(tmp_path / "e.txt")
    (tmp_path / "x.xyz").write_text("a")
    with pytest.raises(ParseError):
        parse_file(tmp_path / "x.xyz")


def test_vector_store_persistence_and_remove(tmp_path):
    path = tmp_path / "f.index"
    vs = VectorStore(HashEmbedder(), path)
    vs.add([1, 2], ["robot motor overheating error", "vacation days per year"])
    assert vs.search("motor overheating", 1)[0][0] == 1
    vs2 = VectorStore(HashEmbedder(), path)       # reload from disk
    assert vs2.count == 2
    vs2.remove([1])
    assert vs2.count == 1 and vs2.search("motor overheating", 1)[0][0] == 2
