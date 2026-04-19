"""
Reranker module — uses Pinecone's hosted BGE reranker to reorder
retrieval results by relevance to the query.
"""

from typing import List, Dict
from pinecone import Pinecone

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_RERANK_MODEL,
    RERANK_TOP_N,
    TOP_K,
)
from app.retrieval import search as vector_search

_pc = Pinecone(api_key=PINECONE_API_KEY)


def rerank(query: str, top_k: int = TOP_K, top_n: int = RERANK_TOP_N) -> List[Dict]:
    """
    Two-stage rerank: first retrieve diverse candidates via vector_search,
    then rerank them with Pinecone's hosted BGE reranker.

    Returns a list of the top_n most relevant chunks.
    """
    # Stage 1: Get diverse candidates (source-balanced retrieval)
    candidates = vector_search(query, top_k=top_k)

    if not candidates:
        return []

    # Stage 2: Rerank candidates using Pinecone inference API
    docs = [{"text": c["chunk_text"]} for c in candidates]

    reranked = _pc.inference.rerank(
        model=PINECONE_RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=top_n,
        rank_fields=["text"],
    )

    hits = []
    for item in reranked.data:
        idx = item["index"]
        original = candidates[idx]
        hits.append(
            {
                "id": original["id"],
                "score": item["score"],
                "chunk_text": original["chunk_text"],
                "source": original["source"],
                "pages": original["pages"],
            }
        )

    return hits


if __name__ == "__main__":
    import sys

    print("=== Reranker Test ===")
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What was Apple's revenue?"
    print(f"Query: {query}\n")

    results = rerank(query, top_k=5, top_n=3)
    print(f"Reranked {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['score']:.4f} | source={r['source']}")
        print(f"      {r['chunk_text'][:150]}...\n")
    print("✅ Reranker test passed!")
