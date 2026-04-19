"""
Agentic RAG module — an LLM agent that decomposes complex queries into
sub-queries, retrieves context for each, merges results, and generates
a synthesized answer.

Pipeline: Query → Decompose → Multi-Retrieve → Rerank → Generate
"""

import json

from langchain_openai import ChatOpenAI

from app.config import MAX_TOKENS, OPENAI_API_KEY, OPENAI_MODEL
from app.generation import generate_answer
from app.reranker import rerank
from app.retrieval import search

_llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    max_tokens=MAX_TOKENS,
    temperature=0.0,
)

# ── Query Decomposition ─────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """You are a query decomposition agent for a RAG system.
Given a user question, decide whether it should be broken into multiple sub-queries
for better retrieval, or kept as a single query.

RULES:
- If the question mentions multiple entities, topics, or asks for a comparison, decompose it.
- If the question is simple and focused on one topic, return it as-is.
- Each sub-query should be self-contained and specific.
- Return 1-4 sub-queries maximum.

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{"sub_queries": ["query1", "query2"]}

Examples:
- "What was Apple's revenue?" → {"sub_queries": ["What was Apple's revenue?"]}
- "Compare Apple and Nike revenue" → {"sub_queries": ["Apple's total revenue?", "Nike's total revenue?"]}
- "Nike's gross margin and operating expenses?" → {"sub_queries": ["Nike's margin?", "Nike's expenses?"]}
"""


def decompose_query(question: str) -> list[str]:
    """Use the LLM to decompose a complex query into sub-queries."""
    messages = [
        ("system", DECOMPOSE_PROMPT),
        ("human", question),
    ]

    response = _llm.invoke(messages)
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        parsed = json.loads(content)
        sub_queries = parsed.get("sub_queries", [question])
        if not sub_queries:
            return [question]
        return sub_queries
    except (json.JSONDecodeError, KeyError):
        return [question]


# ── Multi-Query Retrieval ────────────────────────────────────────────────────


def multi_retrieve(sub_queries: list[str], use_reranker: bool = True, top_k: int = 20, top_n: int = 5) -> list[dict]:
    """
    Run retrieval for each sub-query and merge results with deduplication.
    Keeps the highest score when a chunk appears in multiple sub-query results.
    """
    seen_ids = {}

    for query in sub_queries:
        hits = rerank(query, top_k=top_k, top_n=top_n) if use_reranker else search(query, top_k=top_n)

        for hit in hits:
            chunk_id = hit["id"]
            if chunk_id not in seen_ids or hit["score"] > seen_ids[chunk_id]["score"]:
                seen_ids[chunk_id] = hit

    # Sort by score descending
    merged = sorted(seen_ids.values(), key=lambda h: h["score"], reverse=True)
    return merged


# ── Agentic RAG Pipeline ────────────────────────────────────────────────────


def agentic_rag(
    question: str,
    use_reranker: bool = True,
    top_k: int = 20,
    top_n: int = 5,
    debug: bool = False,
) -> dict:
    """
    Full agentic RAG pipeline:
    1. Decompose the query into sub-queries
    2. Retrieve context for each sub-query
    3. Merge and deduplicate results
    4. Generate a synthesized answer

    Returns a dict with: answer, sources, sub_queries, and optionally debug info.
    """
    # Step 1: Decompose
    sub_queries = decompose_query(question)

    # Step 2 & 3: Multi-retrieve + merge
    chunks = multi_retrieve(sub_queries, use_reranker=use_reranker, top_k=top_k, top_n=top_n)

    if not chunks:
        return {
            "answer": "I couldn't find any relevant information in the documents.",
            "sources": [],
            "sub_queries": sub_queries,
        }

    # Step 4: Generate answer from merged context
    answer = generate_answer(question, chunks)

    result = {
        "answer": answer,
        "sources": chunks,
        "sub_queries": sub_queries,
        "pipeline": (
            "decompose → multi-retrieve → rerank → generate"
            if use_reranker
            else "decompose → multi-retrieve → generate"
        ),
    }

    return result


if __name__ == "__main__":
    import sys

    print("=== Agentic RAG Test ===\n")

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Compare Apple and Nike revenue"
    print(f"Question: {question}\n")

    # Step 1: Decompose
    sub_queries = decompose_query(question)
    print(f"Sub-queries ({len(sub_queries)}):")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")

    # Step 2: Multi-retrieve
    print("\nRetrieving for each sub-query...")
    chunks = multi_retrieve(sub_queries)
    print(f"Merged: {len(chunks)} unique chunks")

    sources = set(c["source"] for c in chunks)
    print(f"Sources covered: {sources}")

    # Step 3: Generate
    print("\nGenerating answer...")
    result = agentic_rag(question)

    print(f"\n{'=' * 60}")
    print(f"  Question: {question}")
    print(f"  Sub-queries: {result['sub_queries']}")
    print(f"  Pipeline: {result['pipeline']}")
    print(f"{'=' * 60}")
    print(f"\n{result['answer']}\n")
    print(f"{'=' * 60}")
    print(f"  Sources ({len(result['sources'])}):")
    for i, c in enumerate(result["sources"], 1):
        pages = c.get("pages", "")
        page_label = f", p.{pages}" if pages else ""
        print(f"  [{i}] {c['source']}{page_label} (score: {c['score']:.4f})")
    print(f"{'=' * 60}")
    print("\n Agentic RAG test passed!")
