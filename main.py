"""
RAG Classic — entry point.

Usage:
    # Start the API server
    python main.py serve

    # Ingest a document from the command line
    python main.py ingest docs/Apple_Q24.pdf
    python main.py ingest docs/Nike-Inc-2025_10K.pdf

    # Ask a question from the command line
    python main.py ask "What was Apple's revenue in Q4 2024?"
"""

import os
import sys

import uvicorn


def serve():
    """Start the FastAPI server."""
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)  # noqa: S104


def ingest(file_path: str):
    """Ingest a single document into the vector store."""
    from app.embedding import upsert_chunks
    from app.ingestion import ingest_document

    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"❌ File not found: {abs_path}")
        sys.exit(1)

    records = ingest_document(abs_path)
    upsert_chunks(records)
    print("🎉 Done!")


def _print_hits(label: str, hits):
    """Pretty-print a list of retrieved/reranked chunks."""
    print(f"\n{'=' * 60}")
    print(f"  {label} ({len(hits)} results)")
    print(f"{'=' * 60}")
    for i, h in enumerate(hits, 1):
        pages = h.get("pages", "")
        page_label = f" | p.{pages}" if pages else ""
        print(f"  [{i}] score={h['score']:.4f} | source={h['source']}{page_label}")
        preview = h["chunk_text"][:200].replace("\n", " ")
        print(f"      {preview}...")
    print()


def ask(question: str, use_reranker: bool = True, debug: bool = False):
    """
    Run the agentic RAG pipeline from the command line.
    The agent decomposes complex queries, retrieves per sub-query, and synthesizes.

    Flags:
        --no-rerank : skip reranking, use raw retrieval only
        --debug     : show decomposition + retrieval details
    """
    from app.agent import agentic_rag

    print(f"\n🔍 Question: {question}")

    result = agentic_rag(
        question,
        use_reranker=use_reranker,
        debug=debug,
    )

    # ── Show sub-queries ──────────────────────────────────────────────
    sub_queries = result.get("sub_queries", [])
    if len(sub_queries) > 1:
        print(f"\n🧠 Agent decomposed into {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"   {i}. {sq}")
    else:
        print("\n🧠 Agent: single query (no decomposition needed)")

    chunks = result.get("sources", [])

    if debug and chunks:
        _print_hits("📡 Merged results (all sub-queries)", chunks)
        sources = set(c["source"] for c in chunks)
        print(f"📊 Source coverage: {', '.join(sources)}\n")

    if not chunks:
        print("No relevant results found.")
        return

    # ── Generated answer ──────────────────────────────────────────────
    print(f"\n💬 Answer:\n{result['answer']}\n")

    # ── Source summary ────────────────────────────────────────────────
    print(f"📄 Sources used ({result.get('pipeline', '')}):")
    for i, c in enumerate(chunks, 1):
        pages = c.get("pages", "")
        page_label = f", p.{pages}" if pages else ""
        print(f"  [{i}] {c['source']}{page_label} (score: {c['score']:.4f})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "serve":
        serve()
    elif command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <file_path>")
            sys.exit(1)
        ingest(sys.argv[2])
    elif command == "ask":
        # Parse flags
        args = sys.argv[2:]
        use_reranker = True
        debug = False
        question_parts = []
        for arg in args:
            if arg == "--no-rerank":
                use_reranker = False
            elif arg == "--debug":
                debug = True
            else:
                question_parts.append(arg)

        if not question_parts:
            print('Usage: python main.py ask "<question>" [--no-rerank] [--debug]')
            sys.exit(1)

        ask(" ".join(question_parts), use_reranker=use_reranker, debug=debug)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: serve, ingest, ask")
        sys.exit(1)


if __name__ == "__main__":
    main()
