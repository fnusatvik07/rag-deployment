# RAG Classic — Simple RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) chatbot built with **Pinecone**, **LangChain**, and **FastAPI**.

It ingests PDF documents, chunks them with page-number tracking, stores them in a Pinecone serverless index with integrated embedding, and answers questions with inline citations and page references.

---

## Architecture

```
  ┌────────────────── INGESTION PIPELINE ──────────────────┐
  │                                                        │
  │  PDF ──▶ Page Extraction ──▶ Chunking ──▶ Pinecone     │
  │          (per-page text)    (512 chars,   (upsert with  │
  │                              64 overlap)  integrated    │
  │                                           embedding)    │
  └────────────────────────────────────────────────────────┘

  ┌────────────────── QUERY PIPELINE ──────────────────────┐
  │                                                        │
  │  Question ──▶ Retrieval ──▶ Reranker ──▶ Generation    │
  │               (Pinecone     (BGE-M3)     (LangChain    │
  │                search)                    ChatOpenAI)   │
  │                                           with [1][2]   │
  └────────────────────────────────────────────────────────┘
```

| Stage       | What it does                                       | Model / Service                  |
|-------------|----------------------------------------------------|----------------------------------|
| Ingestion   | Extracts text per page from PDF, splits into chunks | `pypdf`                          |
| Embedding   | Stores chunks with server-side embedding            | Pinecone `multilingual-e5-large` |
| Retrieval   | Semantic vector search over stored chunks           | Pinecone integrated search       |
| Reranking   | Re-orders results by true relevance to the query    | Pinecone `bge-reranker-v2-m3`    |
| Generation  | Produces an answer with inline citations            | LangChain `ChatOpenAI` (`gpt-4o-mini`) |

---

## Project Structure

```
rag-classic/
├── app/
│   ├── __init__.py        # Package marker
│   ├── config.py          # Settings & environment variables
│   ├── ingestion.py       # PDF/TXT loading, page extraction, chunking
│   ├── embedding.py       # Create Pinecone index & upsert records
│   ├── retrieval.py       # Semantic vector search
│   ├── reranker.py        # Rerank with bge-reranker-v2-m3
│   ├── generation.py      # LLM answer generation with citations
│   └── api.py             # FastAPI REST endpoints
├── docs/                  # Source documents
│   ├── Apple_Q24.pdf
│   └── Nike-Inc-2025_10K.pdf
├── rag_test.py            # Bare-minimum pipeline walkthrough (no functions)
├── main.py                # CLI entry point (ingest / ask / serve)
├── pyproject.toml         # Dependencies
├── .env                   # API keys (not committed — see .gitignore)
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/fnusatvik07/rag-classic.git
cd rag-classic

# Install with uv (recommended)
uv pip install -e .
```

### 2. Add API keys

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
```

### 3. Verify config

```bash
python -m app.config
```

```
=== Config Test ===
PINECONE_API_KEY : ✅ set
OPENAI_API_KEY   : ✅ set
Index name       : rag-classic
Embed model      : multilingual-e5-large
Rerank model     : bge-reranker-v2-m3
Chunk size       : 512, overlap: 64
LLM model        : gpt-4o-mini
✅ Config loaded successfully!
```

---

## Quick Start

```bash
# 1. Install
uv pip install -e .

# 2. Ingest documents
python main.py ingest docs/Apple_Q24.pdf
python main.py ingest docs/Nike-Inc-2025_10K.pdf

# 3. Ask a question
python main.py ask "What was Apple's revenue in Q4 2024?"

# 4. Or start the API server
python main.py serve
```

---

## Pipeline Walkthrough (`rag_test.py`)

A single-file, no-functions walkthrough of the entire RAG pipeline — great for learning or demos:

```bash
python rag_test.py
```

It runs all 9 steps linearly:

| Step | Description                                      |
|------|--------------------------------------------------|
| 1    | Extract text from PDF (page by page)             |
| 2    | Chunk text with page number tracking             |
| 3    | Create Pinecone index (skips if exists)          |
| 4    | Upsert chunks (skips if already done)            |
| 5    | Semantic search (retrieval)                      |
| 6    | Rerank results (shows position changes)          |
| 7    | Generate answer with LangChain ChatOpenAI        |
| 8    | Generate clean final answer from reranked chunks |
| 9    | Test `/generate` API endpoint (if server running)|

---

## CLI Usage

### Ingest a document

```bash
python main.py ingest docs/Apple_Q24.pdf
python main.py ingest docs/Nike-Inc-2025_10K.pdf
```

### Ask a question

```bash
# Default (with reranking + citations)
python main.py ask "What was Apple's revenue in Q4 2024?"

# Debug mode — shows retrieval vs reranked comparison
python main.py ask "What was Apple's revenue in Q4 2024?" --debug

# Skip reranking — raw vector search only
python main.py ask "What was Apple's revenue in Q4 2024?" --no-rerank
```

**Example output:**

```
💬 Answer:
Apple's total revenue in Q4 2024 was $94,930 million [1].

References:
[1] Apple_Q24.pdf, p.1

📄 Sources used:
  [1] Apple_Q24.pdf, p.1 (score: 0.9225)
  [2] Apple_Q24.pdf, p.1,2 (score: 0.9035)
  [3] Apple_Q24.pdf, p.3,4 (score: 0.5195)
```

### Test individual modules

```bash
python -m app.config                              # Verify env vars
python -m app.ingestion docs/Apple_Q24.pdf        # Test PDF extraction & chunking
python -m app.embedding                           # Test upsert (dummy data)
python -m app.retrieval "What was Apple's revenue" # Test vector search
python -m app.reranker "What was Apple's revenue"  # Test reranking
python -m app.generation                           # Test LLM generation
```

---

## API Server

```bash
python main.py serve
```

Server runs at **http://localhost:8000** — Swagger docs at **http://localhost:8000/docs**.

### Endpoints

| Method | Endpoint     | Description                                          |
|--------|--------------|------------------------------------------------------|
| GET    | `/health`    | Health check                                         |
| POST   | `/ingest`    | Ingest a document by file path                       |
| POST   | `/chat`      | Ask a question (full RAG pipeline + debug mode)      |
| POST   | `/generate`  | Retrieve → rerank → generate (clean response)        |
| POST   | `/search`    | Search only (no generation)                          |

### Examples

**Ingest:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/absolute/path/to/docs/Apple_Q24.pdf"}'
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Nike total revenue?", "use_reranker": true}'
```

**Chat with debug (shows retrieval vs reranked):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Nike total revenue?", "use_reranker": true, "debug": true}'
```

**Generate (retrieve → rerank → generate):**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Nike total revenue?", "use_reranker": true, "top_k": 10, "top_n": 3}'
```

**Search only (no LLM):**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Apple earnings", "top_k": 5, "use_reranker": true}'
```

---

## Key Features

- **LangChain Integration** — Uses `langchain-openai` `ChatOpenAI` for LLM generation (latest API: `llm.invoke()`)
- **Integrated Embedding** — Pinecone handles embedding server-side via `multilingual-e5-large`; no local embedding model needed
- **Page Number Tracking** — Each chunk carries its source page number(s) through the entire pipeline
- **Reranking** — `bge-reranker-v2-m3` reorders retrieval results for better relevance
- **Inline Citations** — LLM answers include `[1]`, `[2]` references with source file and page numbers
- **Debug Mode** — `--debug` flag shows retrieval vs reranked comparison and position changes
- **`/generate` Endpoint** — Clean API endpoint returning answer, sources, and pipeline info
- **Skip Duplicate Ingestion** — `rag_test.py` checks if documents are already indexed before re-ingesting
