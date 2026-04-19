"""Tests for app.api module — FastAPI endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


# ── /health ──────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── /ingest ──────────────────────────────────────────────────────────────────


class TestIngestEndpoint:
    @patch("app.api.upsert_chunks")
    @patch("app.api.ingest_document")
    def test_ingest_success(self, mock_ingest, mock_upsert):
        mock_ingest.return_value = [{"id": "x", "chunk_text": "t", "source": "f.pdf"}]
        mock_upsert.return_value = 1

        resp = client.post("/ingest", json={"file_path": "/tmp/f.pdf"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["file"] == "/tmp/f.pdf"
        assert data["chunks"] == 1
        assert data["message"] == "Document ingested successfully"

    @patch("app.api.ingest_document")
    def test_ingest_file_not_found(self, mock_ingest):
        mock_ingest.side_effect = FileNotFoundError("not found")
        resp = client.post("/ingest", json={"file_path": "/missing.pdf"})
        assert resp.status_code == 404

    @patch("app.api.ingest_document")
    def test_ingest_unsupported_type(self, mock_ingest):
        mock_ingest.side_effect = ValueError("Unsupported file type")
        resp = client.post("/ingest", json={"file_path": "/tmp/f.docx"})
        assert resp.status_code == 400

    @patch("app.api.ingest_document")
    def test_ingest_server_error(self, mock_ingest):
        mock_ingest.side_effect = RuntimeError("boom")
        resp = client.post("/ingest", json={"file_path": "/tmp/f.pdf"})
        assert resp.status_code == 500


# ── /chat (agentic) ─────────────────────────────────────────────────────────


class TestChatEndpoint:
    @patch("app.api.agentic_rag")
    def test_chat_agentic(self, mock_rag):
        mock_rag.return_value = {
            "answer": "The answer",
            "sources": [
                {"id": "a::0", "score": 0.9, "source": "a.pdf", "pages": "1", "chunk_text": "text " * 50},
            ],
            "sub_queries": ["Q1"],
            "pipeline": "decompose → multi-retrieve → rerank → generate",
        }

        resp = client.post("/chat", json={"question": "What?", "agentic": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "The answer"
        assert len(data["sources"]) == 1
        assert data["sub_queries"] == ["Q1"]
        assert data["pipeline"] is not None

    @patch("app.api.generate_answer")
    @patch("app.api.rerank")
    @patch("app.api.search")
    def test_chat_classic_with_reranker(self, mock_search, mock_rerank, mock_generate):
        chunks = [
            {"id": "a::0", "score": 0.9, "source": "a.pdf", "pages": "1", "chunk_text": "text " * 50},
        ]
        mock_search.return_value = chunks
        mock_rerank.return_value = chunks
        mock_generate.return_value = "Classic answer"

        resp = client.post("/chat", json={"question": "What?", "agentic": False, "use_reranker": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Classic answer"
        assert "rerank" in data["pipeline"]

    @patch("app.api.generate_answer")
    @patch("app.api.search")
    def test_chat_classic_no_reranker(self, mock_search, mock_generate):
        chunks = [
            {"id": "a::0", "score": 0.9, "source": "a.pdf", "pages": "1", "chunk_text": "text " * 50},
        ]
        mock_search.return_value = chunks
        mock_generate.return_value = "Answer without rerank"

        resp = client.post("/chat", json={"question": "What?", "agentic": False, "use_reranker": False})
        assert resp.status_code == 200
        data = resp.json()
        assert "rerank" not in data["pipeline"]

    @patch("app.api.generate_answer")
    @patch("app.api.search")
    def test_chat_classic_no_results(self, mock_search, mock_generate):
        mock_search.return_value = []

        resp = client.post("/chat", json={"question": "What?", "agentic": False, "use_reranker": False})
        assert resp.status_code == 200
        data = resp.json()
        assert "couldn't find" in data["answer"].lower()
        assert data["sources"] == []

    @patch("app.api.generate_answer")
    @patch("app.api.rerank")
    @patch("app.api.search")
    def test_chat_classic_debug_mode(self, mock_search, mock_rerank, mock_generate):
        chunks = [
            {"id": "a::0", "score": 0.9, "source": "a.pdf", "pages": "1", "chunk_text": "text " * 50},
        ]
        mock_search.return_value = chunks
        mock_rerank.return_value = chunks
        mock_generate.return_value = "Debug answer"

        resp = client.post("/chat", json={"question": "Q?", "agentic": False, "use_reranker": True, "debug": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieved"] is not None
        assert data["reranked"] is not None

    @patch("app.api.agentic_rag")
    def test_chat_server_error(self, mock_rag):
        mock_rag.side_effect = RuntimeError("boom")
        resp = client.post("/chat", json={"question": "Q?"})
        assert resp.status_code == 500


# ── /search ──────────────────────────────────────────────────────────────────


class TestSearchEndpoint:
    @patch("app.api.rerank")
    def test_search_with_reranker(self, mock_rerank):
        mock_rerank.return_value = [{"id": "a", "score": 0.9}]
        resp = client.post("/search", json={"query": "test", "use_reranker": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"
        assert len(data["results"]) == 1

    @patch("app.api.search")
    def test_search_without_reranker(self, mock_search):
        mock_search.return_value = [{"id": "a", "score": 0.8}]
        resp = client.post("/search", json={"query": "test", "use_reranker": False})
        assert resp.status_code == 200

    @patch("app.api.search")
    def test_search_server_error(self, mock_search):
        mock_search.side_effect = RuntimeError("boom")
        resp = client.post("/search", json={"query": "test", "use_reranker": False})
        assert resp.status_code == 500


# ── /generate ────────────────────────────────────────────────────────────────


class TestGenerateEndpoint:
    @patch("app.api.generate_answer")
    @patch("app.api.rerank")
    @patch("app.api.search")
    def test_generate_with_reranker(self, mock_search, mock_rerank, mock_generate):
        chunks = [
            {"id": "a::0", "score": 0.9, "source": "a.pdf", "pages": "1", "chunk_text": "text " * 50},
        ]
        mock_search.return_value = chunks
        mock_rerank.return_value = chunks
        mock_generate.return_value = "Generated answer"

        resp = client.post("/generate", json={"question": "Q?", "use_reranker": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Generated answer"
        assert "rerank" in data["pipeline"]

    @patch("app.api.generate_answer")
    @patch("app.api.search")
    def test_generate_without_reranker(self, mock_search, mock_generate):
        chunks = [
            {"id": "a::0", "score": 0.9, "source": "a.pdf", "pages": "1", "chunk_text": "text " * 50},
        ]
        mock_search.return_value = chunks
        mock_generate.return_value = "Answer"

        resp = client.post("/generate", json={"question": "Q?", "use_reranker": False, "top_n": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert "rerank" not in data["pipeline"]

    @patch("app.api.search")
    def test_generate_no_results(self, mock_search):
        mock_search.return_value = []

        resp = client.post("/generate", json={"question": "Q?", "use_reranker": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"] == []

    @patch("app.api.generate_answer")
    @patch("app.api.search")
    def test_generate_server_error(self, mock_search, mock_generate):
        mock_search.side_effect = RuntimeError("fail")
        resp = client.post("/generate", json={"question": "Q?"})
        assert resp.status_code == 500
