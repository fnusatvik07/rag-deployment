"""Tests for app.retrieval module."""

from unittest.mock import MagicMock, patch

from app.retrieval import _ensure_source_diversity, _parse_hits, search
from tests.conftest import make_chunk, make_pinecone_hit, make_pinecone_response

# ── _parse_hits ──────────────────────────────────────────────────────────────


class TestParseHits:
    def test_parses_standard_response(self):
        raw = make_pinecone_response(
            [
                make_pinecone_hit(0, "a.pdf", "1", 0.95, "text A"),
                make_pinecone_hit(1, "b.pdf", "2", 0.80, "text B"),
            ]
        )
        hits = _parse_hits(raw)
        assert len(hits) == 2
        assert hits[0]["id"] == "a.pdf::chunk-0"
        assert hits[0]["score"] == 0.95
        assert hits[0]["chunk_text"] == "text A"
        assert hits[0]["source"] == "a.pdf"
        assert hits[0]["pages"] == "1"

    def test_empty_response(self):
        raw = {"result": {"hits": []}}
        assert _parse_hits(raw) == []

    def test_missing_fields(self):
        raw = {"result": {"hits": [{"_id": "x", "_score": 0.5, "fields": {}}]}}
        hits = _parse_hits(raw)
        assert hits[0]["chunk_text"] == ""
        assert hits[0]["source"] == ""

    def test_missing_result_key(self):
        raw = {}
        assert _parse_hits(raw) == []


# ── _ensure_source_diversity ─────────────────────────────────────────────────


class TestEnsureSourceDiversity:
    def test_single_source_returns_top_k(self):
        hits = [make_chunk(i, "a.pdf", score=1.0 - i * 0.1) for i in range(10)]
        result = _ensure_source_diversity(hits, top_k=5)
        assert len(result) == 5
        assert all(h["source"] == "a.pdf" for h in result)

    def test_two_sources_reserves_minority_slots(self):
        # 8 from source A, 2 from source B
        hits = [make_chunk(i, "a.pdf", score=0.9 - i * 0.01) for i in range(8)]
        hits += [make_chunk(i, "b.pdf", score=0.5 - i * 0.01) for i in range(2)]
        result = _ensure_source_diversity(hits, top_k=6)
        sources = [h["source"] for h in result]
        assert "b.pdf" in sources

    def test_respects_top_k_limit(self):
        hits = [make_chunk(i, "a.pdf", score=0.9) for i in range(5)]
        hits += [make_chunk(i, "b.pdf", score=0.8) for i in range(5)]
        result = _ensure_source_diversity(hits, top_k=4)
        assert len(result) <= 4

    def test_sorted_by_score_descending(self):
        hits = [make_chunk(0, "a.pdf", score=0.9)]
        hits += [make_chunk(0, "b.pdf", score=0.95)]
        result = _ensure_source_diversity(hits, top_k=5)
        scores = [h["score"] for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_hits(self):
        assert _ensure_source_diversity([], top_k=5) == []

    def test_three_sources(self):
        hits = [make_chunk(i, "a.pdf", score=0.9 - i * 0.01) for i in range(6)]
        hits += [make_chunk(i, "b.pdf", score=0.7 - i * 0.01) for i in range(3)]
        hits += [make_chunk(i, "c.pdf", score=0.5 - i * 0.01) for i in range(3)]
        result = _ensure_source_diversity(hits, top_k=6)
        sources = set(h["source"] for h in result)
        # Should include at least 2 sources
        assert len(sources) >= 2


# ── search ───────────────────────────────────────────────────────────────────


class TestSearch:
    @patch("app.retrieval._pc")
    def test_search_returns_results(self, mock_pc):
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index

        raw = make_pinecone_response(
            [
                make_pinecone_hit(0, "a.pdf", "1", 0.9, "result text"),
            ]
        )
        mock_index.search.return_value = raw

        results = search("test query", top_k=5)
        assert len(results) == 1
        assert results[0]["chunk_text"] == "result text"

    @patch("app.retrieval._pc")
    def test_search_fetches_3x_candidates(self, mock_pc):
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.search.return_value = make_pinecone_response([])

        search("query", top_k=10)

        call_args = mock_index.search.call_args
        query_dict = call_args[1]["query"]
        assert query_dict["top_k"] == 30  # 10 * 3

    @patch("app.retrieval._pc")
    def test_search_empty_results(self, mock_pc):
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.search.return_value = make_pinecone_response([])

        results = search("query")
        assert results == []
