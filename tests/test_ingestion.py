"""Tests for app.ingestion module."""

from unittest.mock import MagicMock, patch

import pytest

from app.ingestion import (
    chunk_pages,
    clean_text,
    extract_pages,
    extract_pages_from_pdf,
    extract_pages_from_txt,
    ingest_document,
)

# ── clean_text ───────────────────────────────────────────────────────────────


class TestCleanText:
    def test_normalises_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_collapses_newlines_and_tabs(self):
        assert clean_text("hello\n\tworld") == "hello world"

    def test_strips_leading_trailing(self):
        assert clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_already_clean(self):
        assert clean_text("hello world") == "hello world"

    def test_only_whitespace(self):
        assert clean_text("   \n\t  ") == ""


# ── extract_pages_from_txt ───────────────────────────────────────────────────


class TestExtractPagesFromTxt:
    def test_reads_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world", encoding="utf-8")
        result = extract_pages_from_txt(str(f))
        assert len(result) == 1
        assert result[0]["page"] == 1
        assert result[0]["text"] == "Hello world"

    def test_reads_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = extract_pages_from_txt(str(f))
        assert len(result) == 1
        assert result[0]["text"] == ""


# ── extract_pages_from_pdf ───────────────────────────────────────────────────


class TestExtractPagesFromPdf:
    @patch("app.ingestion.PdfReader")
    def test_extracts_pages(self, mock_reader_cls):
        page1 = MagicMock()
        page1.extract_text.return_value = "Page one content"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page two content"

        mock_reader = MagicMock()
        mock_reader.pages = [page1, page2]
        mock_reader_cls.return_value = mock_reader

        result = extract_pages_from_pdf("fake.pdf")
        assert len(result) == 2
        assert result[0] == {"page": 1, "text": "Page one content"}
        assert result[1] == {"page": 2, "text": "Page two content"}

    @patch("app.ingestion.PdfReader")
    def test_skips_empty_pages(self, mock_reader_cls):
        page1 = MagicMock()
        page1.extract_text.return_value = "Content"
        page2 = MagicMock()
        page2.extract_text.return_value = ""
        page3 = MagicMock()
        page3.extract_text.return_value = "   "  # whitespace only

        mock_reader = MagicMock()
        mock_reader.pages = [page1, page2, page3]
        mock_reader_cls.return_value = mock_reader

        result = extract_pages_from_pdf("fake.pdf")
        assert len(result) == 1
        assert result[0]["page"] == 1

    @patch("app.ingestion.PdfReader")
    def test_handles_none_text(self, mock_reader_cls):
        page1 = MagicMock()
        page1.extract_text.return_value = None

        mock_reader = MagicMock()
        mock_reader.pages = [page1]
        mock_reader_cls.return_value = mock_reader

        result = extract_pages_from_pdf("fake.pdf")
        assert len(result) == 0


# ── extract_pages ────────────────────────────────────────────────────────────


class TestExtractPages:
    @patch("app.ingestion.extract_pages_from_pdf")
    def test_routes_pdf(self, mock_pdf):
        mock_pdf.return_value = [{"page": 1, "text": "pdf"}]
        result = extract_pages("file.pdf")
        mock_pdf.assert_called_once_with("file.pdf")
        assert result == [{"page": 1, "text": "pdf"}]

    @patch("app.ingestion.extract_pages_from_txt")
    def test_routes_txt(self, mock_txt):
        mock_txt.return_value = [{"page": 1, "text": "txt"}]
        extract_pages("file.txt")
        mock_txt.assert_called_once_with("file.txt")

    @patch("app.ingestion.extract_pages_from_txt")
    def test_routes_md(self, mock_txt):
        mock_txt.return_value = [{"page": 1, "text": "md"}]
        extract_pages("file.md")
        mock_txt.assert_called_once_with("file.md")

    def test_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_pages("file.docx")

    @patch("app.ingestion.extract_pages_from_pdf")
    def test_case_insensitive_extension(self, mock_pdf):
        mock_pdf.return_value = []
        extract_pages("file.PDF")
        mock_pdf.assert_called_once()


# ── chunk_pages ──────────────────────────────────────────────────────────────


class TestChunkPages:
    def test_single_small_page(self):
        pages = [{"page": 1, "text": "Hello world"}]
        chunks = chunk_pages(pages, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0]["chunk_text"] == "Hello world"
        assert chunks[0]["pages"] == [1]

    def test_chunking_with_overlap(self):
        text = "A" * 100
        pages = [{"page": 1, "text": text}]
        chunks = chunk_pages(pages, chunk_size=50, overlap=10)
        assert len(chunks) >= 2
        # Each chunk should be at most 50 chars
        for c in chunks:
            assert len(c["chunk_text"]) <= 50

    def test_multi_page_tracking(self):
        pages = [
            {"page": 1, "text": "A" * 50},
            {"page": 2, "text": "B" * 50},
        ]
        # chunk_size big enough to span both pages
        chunks = chunk_pages(pages, chunk_size=200, overlap=0)
        assert len(chunks) == 1
        assert 1 in chunks[0]["pages"]
        assert 2 in chunks[0]["pages"]

    def test_empty_pages(self):
        pages = [{"page": 1, "text": "   "}]
        chunks = chunk_pages(pages, chunk_size=100, overlap=0)
        assert len(chunks) == 0

    def test_no_pages(self):
        chunks = chunk_pages([], chunk_size=100, overlap=0)
        assert chunks == []

    def test_pages_are_sorted_and_unique(self):
        pages = [
            {"page": 3, "text": "AAA"},
            {"page": 1, "text": "BBB"},
        ]
        chunks = chunk_pages(pages, chunk_size=200, overlap=0)
        assert chunks[0]["pages"] == [1, 3]


# ── ingest_document ──────────────────────────────────────────────────────────


class TestIngestDocument:
    @patch("app.ingestion.extract_pages")
    def test_returns_records(self, mock_extract):
        mock_extract.return_value = [
            {"page": 1, "text": "Hello world this is a test document."},
        ]
        records = ingest_document("/path/to/doc.pdf")
        assert len(records) >= 1
        rec = records[0]
        assert rec["id"] == "doc.pdf::chunk-0"
        assert rec["source"] == "doc.pdf"
        assert "chunk_text" in rec
        assert "pages" in rec

    @patch("app.ingestion.extract_pages")
    def test_record_id_format(self, mock_extract):
        mock_extract.return_value = [
            {"page": 1, "text": "A" * 600},
        ]
        records = ingest_document("/path/to/my_file.pdf")
        for i, rec in enumerate(records):
            assert rec["id"] == f"my_file.pdf::chunk-{i}"

    @patch("app.ingestion.extract_pages")
    def test_pages_as_comma_string(self, mock_extract):
        mock_extract.return_value = [
            {"page": 1, "text": "A" * 300},
            {"page": 2, "text": "B" * 300},
        ]
        records = ingest_document("/fake/path.pdf")
        # At least one record should exist
        assert len(records) >= 1
        # Pages should be comma-separated string
        for rec in records:
            assert isinstance(rec["pages"], str)
