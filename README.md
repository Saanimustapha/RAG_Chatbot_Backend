# RAG Chatbot Backend — Production-style RAG + Custom Indexing (HNSW) 🚀

A FastAPI backend that ingests real documents (PDF/DOCX/TXT/HTML + pasted text), builds a **per-user Retrieval-Augmented Generation (RAG)** corpus, and answers questions with **citations**.  
Unlike “just call an API”, this backend implements a **custom vector store format** and a **from-scratch HNSW (Hierarchical Navigable Small World) index**, with evaluation scripts to measure **latency vs brute-force** and **recall@k**.

> ✅ Built for multi-user usage: isolation is **per user**, with a **single corpus + index per user**.

---

## Why this repo is worth your attention

### What makes it different
- **End-to-end RAG**: ingest → chunk → embed → store → retrieve → generate → cite.
- **Own the storage layer**: vectors stored in a compact binary format (`vectors.f32`) with mapping files (`ids.txt`, `docstore.jsonl`).
- **Own the retrieval layer**: custom **HNSW** ANN index + exact brute-force baseline.
- **Hybrid retrieval**: semantic (HNSW) + lexical (BM25), optional cross-encoder reranking, query rewrites, and MMR diversity.
- **Repeatable evaluation**: scripts to benchmark p50/p95/QPS and recall@k.

---

## Key Features

### ✅ Document ingestion (multi-format)
- PDF (`.pdf`)
- Word (`.docx`)
- Text / Markdown (`.txt`, `.md`)
- HTML (`.html`, `.htm`)
- Raw pasted text

### ✅ Robust pipeline (Phase 2 quality)
- Cleaning (including removal of problematic control characters)
- Configurable chunking strategy:
  - token-based chunking (default)
  - sentence chunking
- Metadata per chunk:
  - document_id, doc_version, chunk_index
  - filename, source_type
  - page range (PDF)
  - section (HTML/structured parsing)

### ✅ Per-user indexing (Phase 3–4)
- Per-doc artifacts written to:
  - `artifacts/user_<user_id>/doc_<doc_id>/v<version>/...`
- Per-user merged corpus stored in:
  - `artifacts/user_<user_id>/_corpus/...`
- Per-user HNSW index stored in:
  - `artifacts/user_<user_id>/_corpus/_hnsw/...`

### ✅ Retrieval options
- **HNSW semantic search** (fast ANN)
- **BM25 keyword search** (lexical)
- **Hybrid merge** (semantic + keyword)
- **Cross-encoder reranking**
- **Query rewriting** (via Ollama)
- **MMR diversity** to reduce near-duplicate chunks
- Optional doc-level filtering (`document_id` / `document_ids`)

### ✅ LLM integration (Open Source friendly)
- Default: **Ollama** (`qwen2.5:7b-instruct`) via HTTP
- Embeddings: **SentenceTransformers** (`BAAI/bge-small-en-v1.5`)

> Pinecone and OpenAI libraries are present in dependencies from earlier phases, but current core flow is designed around **local embeddings + local indexing**.

---

## Architecture (high level)
