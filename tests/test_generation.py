"""Tests for app.generation module."""

from unittest.mock import MagicMock, patch

from app.generation import build_context_block, generate_answer

# ── build_context_block ──────────────────────────────────────────────────────


class TestBuildContextBlock:
    def test_formats_single_chunk(self):
        chunks = [
            {"chunk_text": "Revenue was $10B", "source": "report.pdf", "pages": "5"},
        ]
        result = build_context_block(chunks)
        assert "[1]" in result
        assert "source: report.pdf" in result
        assert "p.5" in result
        assert "Revenue was $10B" in result

    def test_formats_multiple_chunks(self):
        chunks = [
            {"chunk_text": "First chunk", "source": "a.pdf", "pages": "1"},
            {"chunk_text": "Second chunk", "source": "b.pdf", "pages": "3"},
        ]
        result = build_context_block(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "source: a.pdf" in result
        assert "source: b.pdf" in result

    def test_handles_missing_pages(self):
        chunks = [{"chunk_text": "text", "source": "doc.pdf"}]
        result = build_context_block(chunks)
        assert "p." not in result

    def test_handles_empty_pages_string(self):
        chunks = [{"chunk_text": "text", "source": "doc.pdf", "pages": ""}]
        result = build_context_block(chunks)
        assert ", p." not in result

    def test_handles_missing_source(self):
        chunks = [{"chunk_text": "text"}]
        result = build_context_block(chunks)
        assert "unknown" in result

    def test_empty_chunks(self):
        result = build_context_block([])
        assert result == ""


# ── generate_answer ──────────────────────────────────────────────────────────


class TestGenerateAnswer:
    @patch("app.generation._llm")
    def test_returns_llm_content(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "The answer is 42."
        mock_llm.invoke.return_value = mock_response

        chunks = [
            {"chunk_text": "context text", "source": "doc.pdf", "pages": "1"},
        ]
        answer = generate_answer("What is the answer?", chunks)
        assert answer == "The answer is 42."

    @patch("app.generation._llm")
    def test_sends_system_and_human_messages(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "answer"
        mock_llm.invoke.return_value = mock_response

        chunks = [{"chunk_text": "ctx", "source": "x.pdf", "pages": "1"}]
        generate_answer("question?", chunks)

        messages = mock_llm.invoke.call_args[0][0]
        assert messages[0][0] == "system"
        assert messages[1][0] == "human"
        assert "question?" in messages[1][1]
        assert "ctx" in messages[1][1]

    @patch("app.generation._llm")
    def test_includes_context_in_prompt(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "answer"
        mock_llm.invoke.return_value = mock_response

        chunks = [
            {"chunk_text": "Apple revenue was $94B", "source": "apple.pdf", "pages": "3"},
        ]
        generate_answer("revenue?", chunks)

        messages = mock_llm.invoke.call_args[0][0]
        human_msg = messages[1][1]
        assert "Apple revenue was $94B" in human_msg
        assert "[1]" in human_msg
