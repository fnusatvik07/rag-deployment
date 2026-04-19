"""Tests for app.agent module."""

import json
from unittest.mock import MagicMock, patch

from app.agent import agentic_rag, decompose_query, multi_retrieve
from tests.conftest import make_chunk

# ── decompose_query ──────────────────────────────────────────────────────────


class TestDecomposeQuery:
    @patch("app.agent._llm")
    def test_returns_sub_queries(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = json.dumps({"sub_queries": ["Q1", "Q2"]})
        mock_llm.invoke.return_value = mock_resp

        result = decompose_query("Compare Apple and Nike")
        assert result == ["Q1", "Q2"]

    @patch("app.agent._llm")
    def test_single_query_passthrough(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = json.dumps({"sub_queries": ["What is revenue?"]})
        mock_llm.invoke.return_value = mock_resp

        result = decompose_query("What is revenue?")
        assert result == ["What is revenue?"]

    @patch("app.agent._llm")
    def test_strips_markdown_code_fences(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = '```json\n{"sub_queries": ["Q1"]}\n```'
        mock_llm.invoke.return_value = mock_resp

        result = decompose_query("question")
        assert result == ["Q1"]

    @patch("app.agent._llm")
    def test_strips_code_fence_no_language(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = '```\n{"sub_queries": ["Q1"]}\n```'
        mock_llm.invoke.return_value = mock_resp

        result = decompose_query("question")
        assert result == ["Q1"]

    @patch("app.agent._llm")
    def test_invalid_json_falls_back(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = "not valid json"
        mock_llm.invoke.return_value = mock_resp

        result = decompose_query("What is revenue?")
        assert result == ["What is revenue?"]

    @patch("app.agent._llm")
    def test_empty_sub_queries_falls_back(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = json.dumps({"sub_queries": []})
        mock_llm.invoke.return_value = mock_resp

        result = decompose_query("question")
        assert result == ["question"]

    @patch("app.agent._llm")
    def test_missing_key_falls_back(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = json.dumps({"queries": ["Q1"]})
        mock_llm.invoke.return_value = mock_resp

        result = decompose_query("question")
        assert result == ["question"]


# ── multi_retrieve ───────────────────────────────────────────────────────────


class TestMultiRetrieve:
    @patch("app.agent.rerank")
    def test_single_query_with_reranker(self, mock_rerank):
        mock_rerank.return_value = [make_chunk(0, score=0.9), make_chunk(1, score=0.8)]

        result = multi_retrieve(["Q1"], use_reranker=True)
        assert len(result) == 2
        mock_rerank.assert_called_once()

    @patch("app.agent.search")
    def test_single_query_without_reranker(self, mock_search):
        mock_search.return_value = [make_chunk(0, score=0.9)]

        result = multi_retrieve(["Q1"], use_reranker=False, top_n=5)
        assert len(result) == 1
        mock_search.assert_called_once_with("Q1", top_k=5)

    @patch("app.agent.rerank")
    def test_deduplicates_across_queries(self, mock_rerank):
        # Same chunk ID from both queries, different scores
        mock_rerank.side_effect = [
            [make_chunk(0, score=0.8)],
            [make_chunk(0, score=0.95)],
        ]

        result = multi_retrieve(["Q1", "Q2"], use_reranker=True)
        assert len(result) == 1
        assert result[0]["score"] == 0.95  # keeps higher score

    @patch("app.agent.rerank")
    def test_merges_unique_chunks(self, mock_rerank):
        mock_rerank.side_effect = [
            [make_chunk(0, "a.pdf", score=0.9)],
            [make_chunk(0, "b.pdf", score=0.8)],
        ]

        result = multi_retrieve(["Q1", "Q2"], use_reranker=True)
        assert len(result) == 2

    @patch("app.agent.rerank")
    def test_sorted_by_score_descending(self, mock_rerank):
        mock_rerank.return_value = [
            make_chunk(0, score=0.5),
            make_chunk(1, score=0.9),
            make_chunk(2, score=0.7),
        ]

        result = multi_retrieve(["Q1"])
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    @patch("app.agent.rerank")
    def test_empty_results(self, mock_rerank):
        mock_rerank.return_value = []
        result = multi_retrieve(["Q1"])
        assert result == []


# ── agentic_rag ──────────────────────────────────────────────────────────────


class TestAgenticRag:
    @patch("app.agent.generate_answer")
    @patch("app.agent.multi_retrieve")
    @patch("app.agent.decompose_query")
    def test_full_pipeline(self, mock_decompose, mock_retrieve, mock_generate):
        mock_decompose.return_value = ["Q1", "Q2"]
        mock_retrieve.return_value = [make_chunk(0, score=0.9)]
        mock_generate.return_value = "The answer."

        result = agentic_rag("Compare things")

        assert result["answer"] == "The answer."
        assert result["sub_queries"] == ["Q1", "Q2"]
        assert result["sources"] == [make_chunk(0, score=0.9)]
        assert "pipeline" in result

    @patch("app.agent.generate_answer")
    @patch("app.agent.multi_retrieve")
    @patch("app.agent.decompose_query")
    def test_no_chunks_returns_empty(self, mock_decompose, mock_retrieve, mock_generate):
        mock_decompose.return_value = ["Q1"]
        mock_retrieve.return_value = []

        result = agentic_rag("question")

        assert "couldn't find" in result["answer"].lower()
        assert result["sources"] == []
        mock_generate.assert_not_called()

    @patch("app.agent.generate_answer")
    @patch("app.agent.multi_retrieve")
    @patch("app.agent.decompose_query")
    def test_pipeline_label_with_reranker(self, mock_decompose, mock_retrieve, mock_generate):
        mock_decompose.return_value = ["Q1"]
        mock_retrieve.return_value = [make_chunk(0)]
        mock_generate.return_value = "ans"

        result = agentic_rag("q", use_reranker=True)
        assert "rerank" in result["pipeline"]

    @patch("app.agent.generate_answer")
    @patch("app.agent.multi_retrieve")
    @patch("app.agent.decompose_query")
    def test_pipeline_label_without_reranker(self, mock_decompose, mock_retrieve, mock_generate):
        mock_decompose.return_value = ["Q1"]
        mock_retrieve.return_value = [make_chunk(0)]
        mock_generate.return_value = "ans"

        result = agentic_rag("q", use_reranker=False)
        assert "rerank" not in result["pipeline"]

    @patch("app.agent.generate_answer")
    @patch("app.agent.multi_retrieve")
    @patch("app.agent.decompose_query")
    def test_passes_params_to_multi_retrieve(self, mock_decompose, mock_retrieve, mock_generate):
        mock_decompose.return_value = ["Q1"]
        mock_retrieve.return_value = [make_chunk(0)]
        mock_generate.return_value = "ans"

        agentic_rag("q", use_reranker=False, top_k=15, top_n=3)

        mock_retrieve.assert_called_once_with(["Q1"], use_reranker=False, top_k=15, top_n=3)
