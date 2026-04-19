"""Tests for main.py CLI entry point."""

import sys
from unittest.mock import patch

import pytest

from main import _print_hits, ask, ingest, main, serve

# ── _print_hits ──────────────────────────────────────────────────────────────


class TestPrintHits:
    def test_prints_hits(self, capsys):
        hits = [
            {"score": 0.95, "source": "a.pdf", "pages": "3", "chunk_text": "Hello world"},
            {"score": 0.80, "source": "b.pdf", "pages": "", "chunk_text": "Another chunk"},
        ]
        _print_hits("Test Label", hits)
        output = capsys.readouterr().out
        assert "Test Label" in output
        assert "2 results" in output
        assert "0.9500" in output
        assert "a.pdf" in output
        assert "p.3" in output

    def test_prints_empty_hits(self, capsys):
        _print_hits("Empty", [])
        output = capsys.readouterr().out
        assert "0 results" in output


# ── main CLI dispatch ────────────────────────────────────────────────────────


class TestMain:
    def test_no_args_prints_help(self, capsys):
        with patch.object(sys, "argv", ["main.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_unknown_command(self, capsys):
        with patch.object(sys, "argv", ["main.py", "unknown"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("main.serve")
    def test_serve_command(self, mock_serve):
        with patch.object(sys, "argv", ["main.py", "serve"]):
            main()
        mock_serve.assert_called_once()

    @patch("main.ingest")
    def test_ingest_command(self, mock_ingest):
        with patch.object(sys, "argv", ["main.py", "ingest", "doc.pdf"]):
            main()
        mock_ingest.assert_called_once_with("doc.pdf")

    def test_ingest_missing_file_arg(self):
        with patch.object(sys, "argv", ["main.py", "ingest"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("main.ask")
    def test_ask_command(self, mock_ask):
        with patch.object(sys, "argv", ["main.py", "ask", "What", "is", "revenue?"]):
            main()
        mock_ask.assert_called_once_with("What is revenue?", use_reranker=True, debug=False)

    @patch("main.ask")
    def test_ask_with_flags(self, mock_ask):
        with patch.object(sys, "argv", ["main.py", "ask", "Q?", "--no-rerank", "--debug"]):
            main()
        mock_ask.assert_called_once_with("Q?", use_reranker=False, debug=True)

    def test_ask_no_question(self):
        with patch.object(sys, "argv", ["main.py", "ask"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_ask_only_flags_no_question(self):
        with patch.object(sys, "argv", ["main.py", "ask", "--debug"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


# ── serve ────────────────────────────────────────────────────────────────────


class TestServe:
    @patch("main.uvicorn")
    def test_serve_starts_uvicorn(self, mock_uvicorn):
        serve()
        mock_uvicorn.run.assert_called_once_with("app.api:app", host="0.0.0.0", port=8000, reload=True)


# ── ingest ───────────────────────────────────────────────────────────────────


class TestIngest:
    @patch("app.embedding.upsert_chunks")
    @patch("app.ingestion.ingest_document")
    @patch("os.path.exists", return_value=True)
    def test_ingest_success(self, mock_exists, mock_ingest_doc, mock_upsert):
        mock_ingest_doc.return_value = [{"id": "x"}]
        mock_upsert.return_value = 1
        ingest("doc.pdf")
        mock_ingest_doc.assert_called_once()
        mock_upsert.assert_called_once()

    @patch("os.path.exists", return_value=False)
    def test_ingest_file_not_found(self, mock_exists):
        with pytest.raises(SystemExit):
            ingest("missing.pdf")


# ── ask ──────────────────────────────────────────────────────────────────────


class TestAsk:
    @patch("app.agent.agentic_rag")
    def test_ask_with_results(self, mock_rag, capsys):
        mock_rag.return_value = {
            "answer": "42",
            "sources": [
                {"id": "a::0", "score": 0.9, "source": "a.pdf", "pages": "1", "chunk_text": "text"},
            ],
            "sub_queries": ["Q1", "Q2"],
            "pipeline": "decompose → generate",
        }
        ask("What?")
        output = capsys.readouterr().out
        assert "42" in output
        assert "Q1" in output

    @patch("app.agent.agentic_rag")
    def test_ask_no_results(self, mock_rag, capsys):
        mock_rag.return_value = {
            "answer": "",
            "sources": [],
            "sub_queries": ["Q1"],
            "pipeline": "decompose → generate",
        }
        ask("What?")
        output = capsys.readouterr().out
        assert "No relevant results" in output

    @patch("app.agent.agentic_rag")
    def test_ask_single_query_message(self, mock_rag, capsys):
        mock_rag.return_value = {
            "answer": "answer",
            "sources": [
                {"id": "a::0", "score": 0.9, "source": "a.pdf", "pages": "", "chunk_text": "text"},
            ],
            "sub_queries": ["Q1"],
            "pipeline": "p",
        }
        ask("Q1")
        output = capsys.readouterr().out
        assert "single query" in output

    @patch("app.agent.agentic_rag")
    def test_ask_debug_mode(self, mock_rag, capsys):
        mock_rag.return_value = {
            "answer": "answer",
            "sources": [
                {"id": "a::0", "score": 0.9, "source": "a.pdf", "pages": "1", "chunk_text": "text"},
            ],
            "sub_queries": ["Q1"],
            "pipeline": "p",
        }
        ask("Q1", debug=True)
        output = capsys.readouterr().out
        assert "Merged results" in output
        assert "Source coverage" in output
