"""
Shared fixtures and helpers for unit tests.
All external services (Pinecone, OpenAI) are mocked.
"""


def make_chunk(idx=0, source="test.pdf", pages="1", score=0.9, text="chunk text"):
    """Build a fake chunk dict matching the shape used throughout the pipeline."""
    return {
        "id": f"{source}::chunk-{idx}",
        "score": score,
        "chunk_text": text,
        "source": source,
        "pages": pages,
    }


def make_pinecone_hit(idx=0, source="test.pdf", pages="1", score=0.9, text="chunk text"):
    """Build a raw Pinecone search-result hit."""
    return {
        "_id": f"{source}::chunk-{idx}",
        "_score": score,
        "fields": {
            "chunk_text": text,
            "source": source,
            "pages": pages,
        },
    }


def make_pinecone_response(hits):
    """Wrap hits in the Pinecone search response shape."""
    return {"result": {"hits": hits}}
