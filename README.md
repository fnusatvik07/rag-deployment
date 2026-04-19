# RAG Classic - Agentic RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) chatbot built with **Pinecone**, **LangChain**, and **FastAPI**.

It ingests PDF documents, chunks them with page-number tracking, stores them in a Pinecone serverless index with integrated embedding, and answers questions with inline citations and page references. Features an **agentic pipeline** that decomposes complex queries into sub-queries for multi-source retrieval.

## Architecture

```
  INGESTION PIPELINE

  PDF --> Page Extraction --> Chunking --> Pinecone
          (per-page text)    (512 chars,   (upsert with
                              64 overlap)  integrated
                                           embedding)

  AGENTIC QUERY PIPELINE

  Question --> Agent --> Sub-queries --> Multi-Retrieve --> Rerank --> Generate
               (LLM      (per-entity     (Pinecone         (BGE-M3)   (ChatOpenAI
               decompose)  queries)        search)                      with [1][2])
```

| Stage          | What it does                                       | Model / Service                  |
|----------------|----------------------------------------------------|----------------------------------|
| Ingestion      | Extracts text per page from PDF, splits into chunks | `pypdf`                          |
| Embedding      | Stores chunks with server-side embedding            | Pinecone `multilingual-e5-large` |
| Decomposition  | Breaks complex queries into focused sub-queries     | OpenAI `gpt-4o-mini`             |
| Retrieval      | Semantic vector search with source diversity        | Pinecone integrated search       |
| Reranking      | Re-orders results by true relevance to the query    | Pinecone `bge-reranker-v2-m3`    |
| Generation     | Produces an answer with inline citations            | LangChain `ChatOpenAI` (`gpt-4o-mini`) |

## Project Structure

```
rag-classic/
├── app/
│   ├── __init__.py        # Package marker
│   ├── config.py          # Settings & environment variables
│   ├── ingestion.py       # PDF/TXT loading, page extraction, chunking
│   ├── embedding.py       # Create Pinecone index & upsert records
│   ├── retrieval.py       # Semantic vector search with source diversity
│   ├── reranker.py        # Two-stage rerank with BGE reranker
│   ├── generation.py      # LLM answer generation with citations
│   ├── agent.py           # Agentic RAG: query decomposition & multi-retrieve
│   └── api.py             # FastAPI REST endpoints
├── docs/                  # Source documents
│   ├── Apple_Q24.pdf
│   └── Nike-Inc-2025_10K.pdf
├── rag_test.py            # Bare-minimum pipeline walkthrough (no functions)
├── main.py                # CLI entry point (ingest / ask / serve)
├── pyproject.toml         # Dependencies
├── .env                   # API keys (not committed)
├── .gitignore
└── README.md
```

## Setup

### 1. Clone & install

```bash
git clone https://github.com/fnusatvik07/rag-cicd.git
cd rag-cicd

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

## Quick Start

```bash
# 1. Install
uv pip install -e .

# 2. Ingest documents
python main.py ingest docs/Apple_Q24.pdf
python main.py ingest docs/Nike-Inc-2025_10K.pdf

# 3. Ask a question (agentic mode)
python main.py ask "What was Apple's revenue in Q4 2024?"

# 4. Cross-document comparison (auto-decomposes)
python main.py ask "Compare Apple and Nike revenue"

# 5. Or start the API server
python main.py serve
```

## Pipeline Walkthrough (`rag_test.py`)

A single-file, no-functions walkthrough of the entire RAG pipeline, great for learning or demos:

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

## CLI Usage

### Ingest a document

```bash
python main.py ingest docs/Apple_Q24.pdf
python main.py ingest docs/Nike-Inc-2025_10K.pdf
```

### Ask a question

```bash
# Default (agentic mode with reranking + citations)
python main.py ask "What was Apple's revenue in Q4 2024?"

# Cross-document queries (agent auto-decomposes)
python main.py ask "Compare Apple and Nike revenue"

# Debug mode — shows sub-queries, merged results, source coverage
python main.py ask "Compare Apple and Nike revenue" --debug

# Skip reranking — raw vector search only
python main.py ask "What was Apple's revenue in Q4 2024?" --no-rerank
```

**Example output:**

```
🔍 Question: Compare Apple and Nike revenue

🧠 Agent decomposed into 2 sub-queries:
   1. What was Apple's total revenue?
   2. What was Nike's total revenue?

💬 Answer:
Apple's total net sales were $416.2 billion [1], while Nike's total
revenues were $46.3 billion [2].

References:
[1] Apple_Q24.pdf, p.1
[2] Nike-Inc-2025_10K.pdf, p.32

📄 Sources used (decompose → multi-retrieve → rerank → generate):
  [1] Apple_Q24.pdf, p.1 (score: 0.9147)
  [2] Nike-Inc-2025_10K.pdf, p.32 (score: 0.9938)
```

### Test individual modules

```bash
python -m app.config                              # Verify env vars
python -m app.ingestion docs/Apple_Q24.pdf        # Test PDF extraction & chunking
python -m app.embedding                           # Test upsert (dummy data)
python -m app.retrieval "What was Apple's revenue" # Test vector search
python -m app.reranker "What was Apple's revenue"  # Test reranking
python -m app.generation                           # Test LLM generation
python -m app.agent "Compare Apple and Nike"       # Test agentic pipeline
```

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
| POST   | `/chat`      | Ask a question (agentic RAG pipeline + debug mode)   |
| POST   | `/generate`  | Retrieve → rerank → generate (clean response)        |
| POST   | `/search`    | Search only (no generation)                          |

### Examples

**Ingest:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/absolute/path/to/docs/Apple_Q24.pdf"}'
```

**Chat (agentic mode, default):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare Apple and Nike revenue"}'
```

**Chat (classic mode, no agent):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Nike total revenue?", "agentic": false}'
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

## Key Features

- **Agentic Query Decomposition** — LLM agent breaks complex queries into sub-queries for better multi-source retrieval
- **Source Diversity** — Retrieval ensures smaller documents aren't drowned out by larger ones
- **Multi-Source Retrieval** — Runs separate searches per sub-query and merges results with deduplication
- **LangChain Integration** — Uses `langchain-openai` `ChatOpenAI` for LLM generation
- **Integrated Embedding** — Pinecone handles embedding server-side via `multilingual-e5-large`
- **Page Number Tracking** — Each chunk carries its source page number(s) through the entire pipeline
- **Two-Stage Reranking** — Diverse retrieval followed by `bge-reranker-v2-m3` reranking
- **Inline Citations** — LLM answers include `[1]`, `[2]` references with source file and page numbers
- **Debug Mode** — `--debug` flag shows sub-queries, merged results, and source coverage
- **Classic Fallback** — Set `agentic: false` in the API to use the classic single-query pipeline
