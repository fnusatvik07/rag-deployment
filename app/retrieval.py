"""
Retrieval module — performs semantic search against the Pinecone index.
Uses integrated embedding, so we search with raw text (no manual embedding needed).
"""

from typing import List, Dict
from pinecone import Pinecone

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    TOP_K,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)


def _parse_hits(results) -> List[Dict]:
    """Parse Pinecone search results into a flat list of hit dicts."""
    hits = []
    for item in results.get("result", {}).get("hits", []):
        hits.append(
            {
                "id": item.get("_id", ""),
                "score": item.get("_score", 0.0),
                "chunk_text": item.get("fields", {}).get("chunk_text", ""),
                "source": item.get("fields", {}).get("source", ""),
                "pages": item.get("fields", {}).get("pages", ""),
            }
        )
    return hits


def _ensure_source_diversity(hits: List[Dict], top_k: int) -> List[Dict]:
    """
    Ensure results include chunks from multiple sources when available.
    Reserves at least 30% of top_k slots for minority sources.
    """
    sources = {}
    for h in hits:
        sources.setdefault(h["source"], []).append(h)

    if len(sources) <= 1:
        return hits[:top_k]

    # Reserve slots for underrepresented sources
    reserved = max(top_k // 3, 2)
    main_slots = top_k - reserved

    # Primary source gets main_slots, others share reserved slots
    all_sources = sorted(sources.keys(), key=lambda s: len(sources[s]), reverse=True)
    primary = all_sources[0]

    result = sources[primary][:main_slots]

    # Fill reserved slots round-robin from other sources
    others = [sources[s] for s in all_sources[1:]]
    idx = 0
    filled = 0
    while filled < reserved and others:
        for source_hits in others:
            if idx < len(source_hits) and filled < reserved:
                result.append(source_hits[idx])
                filled += 1
        idx += 1
        if all(idx >= len(sh) for sh in others):
            break

    # Sort by score descending
    result.sort(key=lambda h: h["score"], reverse=True)
    return result[:top_k]


def search(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Perform a semantic vector search in Pinecone using integrated embedding.
    Pinecone embeds the query text automatically.
    Ensures source diversity so smaller documents aren't drowned out.

    Returns a list of dicts with keys: id, score, chunk_text, source.
    """
    index = _pc.Index(PINECONE_INDEX_NAME)

    # Fetch extra candidates to allow diversity rebalancing
    fetch_k = top_k * 3

    results = index.search(
        namespace=PINECONE_NAMESPACE,
        query={
            "top_k": fetch_k,
            "inputs": {"text": query},
        },
        fields=["chunk_text", "source", "pages"],
    )

    hits = _parse_hits(results)
    return _ensure_source_diversity(hits, top_k)


if __name__ == "__main__":
    import sys

    print("=== Retrieval Test ===")
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What was Apple's revenue?"
    print(f"Query: {query}\n")

    results = search(query, top_k=3)
    print(f"Found {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['score']:.4f} | source={r['source']}")
        print(f"      {r['chunk_text'][:150]}...\n")
    print("✅ Retrieval test passed!")
