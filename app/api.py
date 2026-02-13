"""
API module — FastAPI application exposing the RAG chatbot endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.ingestion import ingest_document
from app.embedding import upsert_chunks
from app.retrieval import search
from app.reranker import rerank
from app.generation import generate_answer

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Classic Chatbot",
    description="A simple Retrieval-Augmented Generation chatbot using Pinecone & LangChain",
    version="0.1.0",
)


# ── Request / Response models ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    file_path: str  # absolute path to the document to ingest


class IngestResponse(BaseModel):
    file: str
    chunks: int
    message: str


class ChatRequest(BaseModel):
    question: str
    use_reranker: bool = True   # toggle reranking on/off
    debug: bool = False         # show both retrieval + reranked results


class SourceChunk(BaseModel):
    id: str
    score: float
    source: str
    pages: str                  # e.g. "3" or "3,4"
    chunk_text: str
    citation: str               # e.g. "[1]"


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    retrieved: Optional[List[SourceChunk]] = None   # raw retrieval (debug)
    reranked: Optional[List[SourceChunk]] = None     # after reranking (debug)


class GenerateRequest(BaseModel):
    question: str
    top_k: int = 10
    top_n: int = 5
    use_reranker: bool = True


class GenerateResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    pipeline: str               # "retrieve → rerank → generate" or "retrieve → generate"


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_reranker: bool = True


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple health check."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest):
    """
    Ingest a document: extract text → chunk → embed → upsert into Pinecone.
    """
    try:
        records = ingest_document(req.file_path)
        upserted = upsert_chunks(records)
        return IngestResponse(
            file=req.file_path,
            chunks=upserted,
            message="Document ingested successfully",
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Full RAG pipeline: retrieve → (rerank) → generate answer.
    """
    try:
        # Step 1: Always fetch raw retrieval results
        retrieved_chunks = search(req.question)

        # Step 2: Rerank if enabled
        if req.use_reranker:
            reranked_chunks = rerank(req.question)
            chunks = reranked_chunks
        else:
            chunks = retrieved_chunks

        if not chunks:
            return ChatResponse(
                answer="I couldn't find any relevant information in the documents.",
                sources=[],
            )

        # Step 3: Generate answer with citations
        answer = generate_answer(req.question, chunks)

        def _to_source(c, idx):
            return SourceChunk(
                id=c["id"],
                score=c["score"],
                source=c["source"],
                pages=c.get("pages", ""),
                chunk_text=c["chunk_text"][:200] + "...",
                citation=f"[{idx}]",
            )

        sources = [_to_source(c, i) for i, c in enumerate(chunks, 1)]

        # Debug: include raw retrieval + reranked lists for comparison
        debug_retrieved = None
        debug_reranked = None
        if req.debug:
            debug_retrieved = [_to_source(c, i) for i, c in enumerate(retrieved_chunks, 1)]
            if req.use_reranker:
                debug_reranked = [_to_source(c, i) for i, c in enumerate(reranked_chunks, 1)]

        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieved=debug_retrieved,
            reranked=debug_reranked,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_endpoint(req: SearchRequest):
    """
    Search only (no generation) — useful for debugging retrieval.
    """
    try:
        if req.use_reranker:
            results = rerank(req.query, top_k=req.top_k)
        else:
            results = search(req.query, top_k=req.top_k)
        return {"query": req.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    """
    Standalone generation endpoint: retrieve → (rerank) → generate.
    Returns the final LLM answer with cited, reranked sources.
    """
    try:
        # Step 1: Retrieve
        retrieved_chunks = search(req.question, top_k=req.top_k)

        # Step 2: Rerank (optional)
        if req.use_reranker:
            chunks = rerank(req.question, top_k=req.top_k, top_n=req.top_n)
            pipeline = "retrieve → rerank → generate"
        else:
            chunks = retrieved_chunks[:req.top_n]
            pipeline = "retrieve → generate"

        if not chunks:
            return GenerateResponse(
                question=req.question,
                answer="No relevant information found in the documents.",
                sources=[],
                pipeline=pipeline,
            )

        # Step 3: Generate
        answer = generate_answer(req.question, chunks)

        sources = [
            SourceChunk(
                id=c["id"],
                score=c["score"],
                source=c["source"],
                pages=c.get("pages", ""),
                chunk_text=c["chunk_text"][:200] + "...",
                citation=f"[{i}]",
            )
            for i, c in enumerate(chunks, 1)
        ]

        return GenerateResponse(
            question=req.question,
            answer=answer,
            sources=sources,
            pipeline=pipeline,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
