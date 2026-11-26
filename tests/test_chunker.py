import pytest

from app.services.chunker import chunk_sections, split_text
from app.services.parser import Section, clean_text


def test_empty_text():
    assert split_text("   \n ") == []


def test_short_text_single_chunk():
    assert split_text("Hello world.", 500, 80) == ["Hello world."]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # overlap: tail of chunk i appears at the start of chunk i+1
    for a, b in zip(chunks, chunks[1:]):
        assert a[-15:].split()[-1] in b[:80]


def test_no_text_lost():
    text = " ".join(f"w{i}" for i in range(300))
    joined = " ".join(split_text(text, 120, 20))
    assert all(f"w{i}" in joined for i in range(300))


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)


def test_metadata_and_global_index():
    secs = [Section("A " * 300, page=1), Section("B " * 300, page=2, label="Intro")]
    chunks = chunk_sections(secs, 200, 30)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].page == 1 and chunks[-1].page == 2
    assert chunks[-1].section == "Intro"


def test_clean_text():
    assert clean_text("a   b\r\n\r\n\r\n\r\nc\x00") == "a b\n\nc"
