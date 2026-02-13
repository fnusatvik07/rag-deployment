"""
RAG Pipeline Walkthrough — Bare minimum, step-by-step.
Run: python rag_test.py

This file walks through the entire RAG pipeline in plain, linear code:
  Step 1: Ingest a PDF → extract text per page
  Step 2: Chunk the text (with page tracking)
  Step 3: Create Pinecone index (with integrated embedding)
  Step 4: Upsert chunks into Pinecone
  Step 5: Semantic search (retrieval)
  Step 6: Rerank results
  Step 7: Generate answer with citations (OpenAI — inline)
  Step 8: Generate final answer from reranked chunks (clean)
  Step 9: Test /generate API endpoint (if server running)
"""

import os
import re
import time
from pypdf import PdfReader
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "rag-test"
NAMESPACE = "documents"
EMBED_MODEL = "multilingual-e5-large"
RERANK_MODEL = "bge-reranker-v2-m3"
LLM_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

DOC_PATH = "docs/Apple_Q24.pdf"  # change to any doc you want to test

pc = Pinecone(api_key=PINECONE_API_KEY)
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    max_tokens=1024,
    temperature=0.2,
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Extract text from PDF (page by page)
# ─────────────────────────────────────────────────────────────────────────────
print("\n📖 STEP 1: Extracting text from PDF...")

reader = PdfReader(DOC_PATH)
pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if text.strip():
        pages.append({"page": i + 1, "text": text})

print(f"   Extracted {len(pages)} pages from '{DOC_PATH}'")
print(f"   Page 1 preview: {pages[0]['text'][:100]}...")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Chunk the text (with page tracking)
# ─────────────────────────────────────────────────────────────────────────────
print("\n✂️  STEP 2: Chunking text...")

# Build flat text + track which character belongs to which page
full_text = ""
char_to_page = []
for p in pages:
    cleaned = re.sub(r"\s+", " ", p["text"]).strip()
    if cleaned:
        if full_text:
            full_text += " "
            char_to_page.append(p["page"])
        full_text += cleaned
        char_to_page.extend([p["page"]] * len(cleaned))

# Slide a window over the text to create overlapping chunks
chunks = []
start = 0
while start < len(full_text):
    end = min(start + CHUNK_SIZE, len(full_text))
    chunk_text = full_text[start:end].strip()
    if chunk_text:
        page_set = sorted(set(char_to_page[start:end]))
        chunks.append({
            "chunk_text": chunk_text,
            "pages": ",".join(str(p) for p in page_set),
        })
    start += CHUNK_SIZE - CHUNK_OVERLAP

file_name = os.path.basename(DOC_PATH)
records = [
    {
        "_id": f"{file_name}::chunk-{i}",
        "chunk_text": c["chunk_text"],
        "source": file_name,
        "pages": c["pages"],
    }
    for i, c in enumerate(chunks)
]

print(f"   Created {len(records)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
print(f"   Chunk 0: pages={records[0]['pages']} | {records[0]['chunk_text'][:80]}...")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Create Pinecone index (skip if already exists)
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏗️  STEP 3: Setting up Pinecone index...")

if pc.has_index(INDEX_NAME):
    print(f"   Index '{INDEX_NAME}' already exists — skipping creation.")
else:
    print(f"   Creating index '{INDEX_NAME}' with integrated embedding ({EMBED_MODEL})...")
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": EMBED_MODEL,
            "field_map": {"text": "chunk_text"},
        },
    )
    # Wait until ready
    while not pc.describe_index(INDEX_NAME).status.get("ready", False):
        time.sleep(1)
    print("   ✅ Index ready!")

index = pc.Index(INDEX_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Upsert chunks (skip if document already ingested)
# ─────────────────────────────────────────────────────────────────────────────
print("\n📦 STEP 4: Upserting chunks into Pinecone...")

# Quick check: see if first chunk of this doc already exists
stats = index.describe_index_stats()
ns_count = stats.get("namespaces", {}).get(NAMESPACE, {}).get("record_count", 0)

# Simple heuristic: if namespace has records AND first chunk id exists, skip
already_ingested = False
if ns_count > 0:
    try:
        fetch_result = index.fetch(ids=[records[0]["_id"]], namespace=NAMESPACE)
        if fetch_result.get("vectors", {}):
            already_ingested = True
    except Exception:
        pass

if already_ingested:
    print(f"   '{file_name}' already ingested ({ns_count} records in namespace) — skipping.")
else:
    # Upsert in batches of 96 (Pinecone limit for text records)
    for i in range(0, len(records), 96):
        batch = records[i : i + 96]
        index.upsert_records(NAMESPACE, batch)
    print(f"   Upserted {len(records)} chunks")

    # Wait a few seconds for indexing to complete
    print("   ⏳ Waiting for vectors to be indexed...")
    time.sleep(10)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Semantic search (retrieval)
# ─────────────────────────────────────────────────────────────────────────────
QUESTION = "What was Apple's total revenue in Q4 2024?"

print(f"\n🔍 STEP 5: Searching for: \"{QUESTION}\"")

search_results = index.search(
    namespace=NAMESPACE,
    query={"top_k": 10, "inputs": {"text": QUESTION}},
    fields=["chunk_text", "source", "pages"],
)

retrieved = []
for hit in search_results.get("result", {}).get("hits", []):
    retrieved.append({
        "id": hit["_id"],
        "score": hit["_score"],
        "chunk_text": hit["fields"]["chunk_text"],
        "source": hit["fields"]["source"],
        "pages": hit["fields"].get("pages", ""),
    })

print(f"   Found {len(retrieved)} results:\n")
for i, r in enumerate(retrieved[:5], 1):  # show top 5
    print(f"   [{i}] score={r['score']:.4f} | {r['source']}, p.{r['pages']}")
    print(f"       {r['chunk_text'][:120]}...\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Rerank results
# ─────────────────────────────────────────────────────────────────────────────
print(f"🔀 STEP 6: Reranking with {RERANK_MODEL}...")

reranked_results = index.search(
    namespace=NAMESPACE,
    query={"top_k": 10, "inputs": {"text": QUESTION}},
    rerank={
        "model": RERANK_MODEL,
        "top_n": 5,
        "rank_fields": ["chunk_text"],
    },
    fields=["chunk_text", "source", "pages"],
)

reranked = []
for hit in reranked_results.get("result", {}).get("hits", []):
    reranked.append({
        "id": hit["_id"],
        "score": hit["_score"],
        "chunk_text": hit["fields"]["chunk_text"],
        "source": hit["fields"]["source"],
        "pages": hit["fields"].get("pages", ""),
    })

print(f"   Reranked to {len(reranked)} results:\n")
for i, r in enumerate(reranked, 1):
    print(f"   [{i}] score={r['score']:.4f} | {r['source']}, p.{r['pages']}")
    print(f"       {r['chunk_text'][:120]}...\n")

# Compare: did reranking change the order?
print("   📊 Rerank impact:")
retrieved_ids = [h["id"] for h in retrieved]
for i, r in enumerate(reranked, 1):
    old_pos = retrieved_ids.index(r["id"]) + 1 if r["id"] in retrieved_ids else "?"
    print(f"      [{i}] was position {old_pos} → now position {i}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Generate answer with citations (OpenAI)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n💬 STEP 7: Generating answer with LangChain ChatOpenAI ({LLM_MODEL})...")

# Build context from reranked chunks
context_parts = []
for i, c in enumerate(reranked, 1):
    page_label = f", p.{c['pages']}" if c["pages"] else ""
    context_parts.append(f"[{i}] (source: {c['source']}{page_label})\n{c['chunk_text']}")

context = "\n\n".join(context_parts)

system_msg = (
    "You are a helpful assistant. Answer based ONLY on the provided context. "
    "Cite sources inline as [1], [2], etc. "
    "End with a References section: [n] filename, p.X"
)

ai_msg = llm.invoke([
    ("system", system_msg),
    ("human", f"Context:\n{context}\n\n---\nQuestion: {QUESTION}"),
])

answer = ai_msg.content

print(f"\n{'='*60}")
print(f"  Question: {QUESTION}")
print(f"{'='*60}")
print(f"\n{answer}\n")
print(f"{'='*60}")
print("  Sources:")
for i, c in enumerate(reranked, 1):
    print(f"  [{i}] {c['source']}, p.{c['pages']} (score: {c['score']:.4f})")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Generate using reranked results (clean version)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n🔧 STEP 8: Generating a clean final answer from reranked chunks...")

# Build context from reranked results (same as Step 7 but cleaner)
ctx_parts = []
for i, c in enumerate(reranked, 1):
    page_label = f", p.{c['pages']}" if c["pages"] else ""
    ctx_parts.append(f"[{i}] (source: {c['source']}{page_label})\n{c['chunk_text']}")

ctx = "\n\n".join(ctx_parts)

system_prompt = (
    "You are a helpful assistant. Answer based ONLY on the provided context. "
    "Cite sources inline as [1], [2], etc. "
    "End with a References section: [n] filename, p.X"
)

final_msg = llm.invoke([
    ("system", system_prompt),
    ("human", f"Context:\n{ctx}\n\n---\nQuestion: {QUESTION}"),
])

final_answer = final_msg.content

print(f"\n{'='*60}")
print(f"  Final Generated Answer (from reranked chunks)")
print(f"  Question: {QUESTION}")
print(f"  Pipeline: retrieve → rerank → generate")
print(f"{'='*60}")
print(f"\n{final_answer}\n")
print(f"{'='*60}")
print(f"  Sources used:")
for i, c in enumerate(reranked, 1):
    print(f"  [{i}] {c['source']}, p.{c['pages']} (score: {c['score']:.4f})")
print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Test /generate API endpoint (if server is running)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n🌐 STEP 9: Testing /generate API endpoint...")

import json
import urllib.request

API_URL = "http://localhost:8000/generate"

payload = json.dumps({
    "question": QUESTION,
    "use_reranker": True,
    "top_k": 10,
    "top_n": 5,
}).encode("utf-8")

try:
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    print(f"\n{'='*60}")
    print(f"  /generate API Response")
    print(f"  Pipeline: {data['pipeline']}")
    print(f"{'='*60}")
    print(f"\n{data['answer']}\n")
    print(f"{'='*60}")
    print(f"  Sources:")
    for s in data["sources"]:
        print(f"  {s['citation']} {s['source']}, p.{s['pages']} (score: {s['score']:.4f})")
    print(f"{'='*60}")

except Exception as e:
    print(f"   ⚠️  API test skipped ({e})")
    print(f"   Start the server first: python main.py serve")


# ─────────────────────────────────────────────────────────────────────────────
# 🧪 TRY MORE QUESTIONS — uncomment any line below and re-run
# ─────────────────────────────────────────────────────────────────────────────
# QUESTION = "What was Apple's services revenue?"
# QUESTION = "How did iPhone sales perform?"
# QUESTION = "What were Apple's operating expenses?"
# QUESTION = "What is Apple's gross margin?"
