# RAG Chatbot Backend — Production-style RAG +  custom retrieval system on top of hnswlib 🚀

A FastAPI backend that ingests real documents (PDF/DOCX/TXT/HTML + pasted text), builds a **per-user Retrieval-Augmented Generation (RAG)** corpus, and answers questions with **citations**.  
Unlike “just call an API”, this project implements a custom per-user vector indexing pipeline for RAG. Vectors are stored in a compact binary format and indexed using an HNSW-based ANN index (via an HNSW library). The system supports per-user corpus merges, index persistence, incremental updates (tombstones + append-only vectors), and benchmarking against an exact brute-force baseline (recall@k, p50/p95 latency).

> ✅ Built for multi-user usage: isolation is **per user**, with a **single corpus + index per user**.

---

## Why this repo is worth your attention

### What makes it different
- **End-to-end RAG**: ingest → chunk → embed → store → retrieve → generate → cite.
- **Own the storage layer**: vectors stored in a compact binary format (`vectors.f32`) with mapping files (`ids.txt`, `docstore.jsonl`).
- **Own the retrieval layer**: **HNSW** ANN index + exact brute-force baseline.
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
```
Upload / Paste
     │
     ▼
┌────────────────────────────────────┐
│         Ingestion Pipeline         │
│  - load (pdf / docx / html / txt)  │
│  - clean + normalize               │
│  - chunk + metadata                │
│  - embed (SentenceTransformers)    │
└───────────────┬────────────────────┘
                │
                │  writes per-doc artifacts
                ▼
artifacts/user_<id>/doc_<docid>/v<version>/
    ├── chunks.jsonl
    ├── docstore.jsonl
    └── embeddings.bin
                │
                │  merge into per-user corpus
                ▼
artifacts/user_<id>/_corpus/
    ├── vectors.f32
    ├── ids.txt
    ├── docstore.jsonl
    ├── manifest.json
    ├── deleted_ids.txt
    └── _hnsw/
        ├── hnsw_meta.json
        └── hnsw_graph.jsonl
```
Chat Query → embed(query) → retrieve (HNSW/BM25/Hybrid/Rerank) → build prompt → Ollama → answer + citations


---

## Tech Stack

- **API**: FastAPI, Uvicorn
- **DB**: PostgreSQL (async SQLAlchemy + asyncpg)
- **Auth**: JWT (python-jose), password hashing (argon2 via passlib)
- **Embeddings**: SentenceTransformers (`BAAI/bge-small-en-v1.5`)
- **LLM**: Ollama (`qwen2.5:7b-instruct`)
- **Indexing**:  Built on top of hnswlib
- **Keyword Retrieval**: BM25 (`rank-bm25`)
- **Reranking**: CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Artifacts + Storage**: binary float32 vectors + JSONL metadata
- **Evaluation**: numpy + matplotlib + benchmark scripts

---

## Getting Started

### 1) Prerequisites
- Python **3.11+**
- PostgreSQL running locally or via Docker
- (Recommended) Ollama installed and running

### 2) Install dependencies
```bash
pip install -r requirements.txt
```
### 3) Configure environment
#### Database
- DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb
- DATABASE_URL_SYNC=postgresql://postgres:postgres@localhost:5432/ragdb

#### Auth
- JWT_SECRET=your-super-secret
- JWT_ALG=HS256

#### Embeddings
- EMBEDDING_PROVIDER=local
- EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
- EMBEDDING_DIM=384

#### LLM (Ollama)
- LLM_PROVIDER=ollama
- OLLAMA_BASE_URL=http://localhost:11434
- OLLAMA_MODEL=qwen2.5:7b-instruct

#### Retrieval + chunking
- TOP_K=6
- CHUNK_STRATEGY=tokens
- CHUNK_TOKENS=450
- CHUNK_OVERLAP=80

#### HNSW parameters
- HNSW_M=16
- HNSW_EFC=200
- HNSW_EFS=80
- HNSW_METRIC=cosine

#### Artifacts
- ARTIFACTS_DIR=artifacts
