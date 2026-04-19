"""Tests for app.reranker module."""

from unittest.mock import MagicMock, patch

from app.reranker import rerank
from tests.conftest import make_chunk


class TestRerank:
    @patch("app.reranker._pc")
    @patch("app.reranker.vector_search")
    def test_rerank_returns_reranked_results(self, mock_search, mock_pc):
        candidates = [
            make_chunk(0, "a.pdf", score=0.9, text="chunk zero"),
            make_chunk(1, "a.pdf", score=0.8, text="chunk one"),
            make_chunk(2, "b.pdf", score=0.7, text="chunk two"),
        ]
        mock_search.return_value = candidates

        # Simulate reranker response — reorders: index 2 first, then 0, then 1
        rerank_result = MagicMock()
        rerank_result.data = [
            {"index": 2, "score": 0.98},
            {"index": 0, "score": 0.85},
        ]
        mock_pc.inference.rerank.return_value = rerank_result

        result = rerank("test query", top_k=5, top_n=2)

        assert len(result) == 2
        assert result[0]["id"] == "b.pdf::chunk-2"
        assert result[0]["score"] == 0.98
        assert result[1]["id"] == "a.pdf::chunk-0"
        assert result[1]["score"] == 0.85

    @patch("app.reranker._pc")
    @patch("app.reranker.vector_search")
    def test_rerank_empty_candidates(self, mock_search, mock_pc):
        mock_search.return_value = []
        result = rerank("test query")
        assert result == []
        mock_pc.inference.rerank.assert_not_called()

    @patch("app.reranker._pc")
    @patch("app.reranker.vector_search")
    def test_rerank_preserves_metadata(self, mock_search, mock_pc):
        candidates = [
            make_chunk(0, "doc.pdf", pages="3,4", score=0.9, text="hello world"),
        ]
        mock_search.return_value = candidates

        rerank_result = MagicMock()
        rerank_result.data = [{"index": 0, "score": 0.99}]
        mock_pc.inference.rerank.return_value = rerank_result

        result = rerank("query", top_k=5, top_n=1)
        assert result[0]["source"] == "doc.pdf"
        assert result[0]["pages"] == "3,4"
        assert result[0]["chunk_text"] == "hello world"

    @patch("app.reranker._pc")
    @patch("app.reranker.vector_search")
    def test_rerank_calls_with_correct_model(self, mock_search, mock_pc):
        mock_search.return_value = [make_chunk(0)]

        rerank_result = MagicMock()
        rerank_result.data = [{"index": 0, "score": 0.9}]
        mock_pc.inference.rerank.return_value = rerank_result

        rerank("query", top_k=5, top_n=1)

        call_kwargs = mock_pc.inference.rerank.call_args[1]
        assert call_kwargs["model"] == "bge-reranker-v2-m3"
        assert call_kwargs["top_n"] == 1
        assert call_kwargs["rank_fields"] == ["text"]
