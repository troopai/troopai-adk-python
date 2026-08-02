"""Tests for document loaders and the loader registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from troopai.adk.exceptions import DocumentLoadError, UnsupportedDocumentSourceError
from troopai.adk.rag.loaders import (
    CSVLoader,
    DirectoryLoader,
    DOCXLoader,
    GithubLoader,
    JSONLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    WebsiteLoader,
    YoutubeChannelLoader,
    YoutubeVideoLoader,
    is_url,
    resolve_loader,
)


async def test_text_loader_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")
    docs = await TextLoader().load(str(path))
    assert len(docs) == 1
    assert docs[0].content == "hello world"
    assert docs[0].source == str(path)


async def test_text_loader_blank_file_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")
    assert await TextLoader().load(str(path)) == []


async def test_text_loader_missing_file_raises() -> None:
    with pytest.raises(DocumentLoadError):
        await TextLoader().load("/no/such/file.txt")


async def test_markdown_loader_tags_format(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nbody", encoding="utf-8")
    docs = await MarkdownLoader().load(str(path))
    assert docs[0].metadata["format"] == "markdown"


async def test_csv_loader_one_document_per_row(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,role\nAda,eng\nGrace,admiral\n", encoding="utf-8")
    docs = await CSVLoader().load(str(path))
    assert len(docs) == 2
    assert "name: Ada" in docs[0].content
    assert docs[0].metadata["row"] == "0"


async def test_json_loader_list_is_per_element(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
    docs = await JSONLoader().load(str(path))
    assert len(docs) == 2
    assert docs[1].metadata["index"] == "1"


async def test_json_loader_object_is_single_document(tmp_path: Path) -> None:
    path = tmp_path / "obj.json"
    path.write_text('{"a": 1, "b": 2}', encoding="utf-8")
    docs = await JSONLoader().load(str(path))
    assert len(docs) == 1


async def test_json_loader_invalid_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        await JSONLoader().load(str(path))


async def test_directory_loader_fans_out_and_skips_unsupported(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")
    docs = await DirectoryLoader().load(str(tmp_path))
    sources = sorted(Path(doc.source).name for doc in docs)
    assert sources == ["a.txt", "b.md"]


async def test_directory_loader_non_directory_raises(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        await DirectoryLoader().load(str(path))


def test_is_url() -> None:
    assert is_url("https://example.com/page")
    assert is_url("http://example.com")
    assert not is_url("/local/path.txt")
    assert not is_url("relative.pdf")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("notes.txt", TextLoader),
        ("README.md", MarkdownLoader),
        ("data.csv", CSVLoader),
        ("data.json", JSONLoader),
        ("https://example.com/page", WebsiteLoader),
        ("https://github.com/owner/repo", GithubLoader),
        ("https://www.youtube.com/watch?v=abc123", YoutubeVideoLoader),
        ("https://www.youtube.com/@somechannel", YoutubeChannelLoader),
    ],
)
def test_resolve_loader_dispatch(source: str, expected: type) -> None:
    assert isinstance(resolve_loader(source), expected)


def test_resolve_loader_directory(tmp_path: Path) -> None:
    assert isinstance(resolve_loader(str(tmp_path)), DirectoryLoader)


def test_resolve_loader_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedDocumentSourceError):
        resolve_loader("archive.zip")


async def test_pdf_loader_extracts_pages(tmp_path: Path) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "doc.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Quantum mechanics is fascinating.")
    document.save(str(path))
    document.close()
    docs = await PDFLoader().load(str(path))
    assert len(docs) == 1
    assert "Quantum mechanics" in docs[0].content
    assert docs[0].metadata["page"] == "1"


async def test_docx_loader_extracts_paragraphs(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "doc.docx"
    document = docx.Document()
    document.add_paragraph("The first paragraph.")
    document.add_paragraph("The second paragraph.")
    document.save(str(path))
    docs = await DOCXLoader().load(str(path))
    assert len(docs) == 1
    assert "first paragraph" in docs[0].content
    assert "second paragraph" in docs[0].content


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id_handles_every_routed_url_shape(source: str, expected: str) -> None:
    # The registry routes every non-channel youtube.com/youtu.be URL to the
    # video loader, so /shorts/, /embed/, and /live/ forms must all parse.
    from troopai.adk.rag.loaders.youtube import _extract_video_id

    assert _extract_video_id(source) == expected


def test_extract_video_id_unparseable_url_raises() -> None:
    from troopai.adk.rag.loaders.youtube import _extract_video_id

    with pytest.raises(DocumentLoadError):
        _extract_video_id("https://www.youtube.com/feed/subscriptions")
