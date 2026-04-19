"""
Generation module — sends the retrieved context + user query to an LLM
and returns the final answer.
Uses LangChain's ChatOpenAI for LLM interaction.
"""

from langchain_openai import ChatOpenAI

from app.config import MAX_TOKENS, OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE

_llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Answer using ONLY the information in the context chunks below.
Synthesize information across multiple chunks when needed to build a complete answer.
If the context contains partial information relevant to the question, provide what you can.
Only say "I don't have enough information" if the context contains absolutely nothing relevant.

CITATION RULES:
- Each context chunk is labeled [1], [2], etc. with its source document and page number(s).
- When you use information from a chunk, cite it inline like [1], [2], etc.
- Combine data from multiple chunks and sources when the question requires it.
- At the end of your answer, add a "References" section listing each cited source with page numbers.
- Format: [n] source_filename, p.X

Example:
Apple's Q4 revenue was $94.9 billion [1], with Services reaching a record $25 billion [2].

References:
[1] Apple_Q24.pdf, p.3
[2] Apple_Q24.pdf, p.5"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context string with page info."""
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "unknown")
        pages = c.get("pages", "")
        text = c.get("chunk_text", "")
        page_label = f", p.{pages}" if pages else ""
        parts.append(f"[{i}] (source: {source}{page_label})\n{text}")
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Generate an answer with inline citations using LangChain ChatOpenAI.
    """
    context = build_context_block(chunks)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\n---\nQuestion: {query}"),
    ]

    ai_msg = _llm.invoke(messages)
    return ai_msg.content


if __name__ == "__main__":
    print("=== Generation Test (LangChain) ===")
    print(f"   Using: ChatOpenAI(model={OPENAI_MODEL})")

    # Test with dummy context
    test_chunks = [
        {
            "chunk_text": "Apple reported Q4 2024 revenue of $94.9 billion, up 6% YoY.",
            "source": "Apple_Q24.pdf",
        },
        {
            "chunk_text": "Services revenue reached an all-time record of $25 billion.",
            "source": "Apple_Q24.pdf",
        },
    ]
    test_query = "What was Apple's revenue in Q4 2024?"

    print(f"Query: {test_query}")
    print(f"Context chunks: {len(test_chunks)}\n")

    answer = generate_answer(test_query, test_chunks)
    print(f"💬 Answer:\n{answer}")
    print("\n✅ Generation test passed (LangChain)!")
