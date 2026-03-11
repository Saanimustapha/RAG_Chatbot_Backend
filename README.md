# RAG Chatbot Backend

A fully **local, privacy-preserving** Retrieval-Augmented Generation (RAG) backend that lets users upload documents and chat with them through a natural-language API. No data leaves your machine — embeddings, retrieval, and generation all run on-premise.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Benchmark Results](#benchmark-results)
- [Configuration Reference](#configuration-reference)
- [Scripts](#scripts)

---

## Overview

This backend powers a document-grounded Q&A system. Users upload PDF, DOCX, HTML, or plain-text files; the system ingests, chunks, and embeds them locally using [sentence-transformers](https://www.sbert.net/). At query time, a multi-stage retrieval pipeline finds the most relevant passages and passes them to a locally-served LLM ([Qwen2.5:7B-Instruct](https://ollama.com/library/qwen2.5) via [Ollama](https://ollama.com/)) to generate a grounded, citation-backed answer.

All computation is local. No user data is sent to any external API.

---

## Key Features

| Feature | Details |
|---|---|
| **Local-first** | All embeddings and generation run on-device via sentence-transformers + Ollama |
| **Multi-format ingestion** | PDF (page-aware + header/footer stripping), DOCX, HTML (section-structured), plain text / Markdown |
| **Hybrid retrieval** | Dense HNSW search (hnswlib) fused with BM25 sparse keyword search |
| **Cross-encoder re-ranking** | `ms-marco-MiniLM-L-6-v2` re-ranks candidates for precision |
| **MMR diversity** | Maximal Marginal Relevance deduplication prevents redundant context chunks |
| **Multi-query expansion** | LLM rewrites the query into N alternatives to improve recall |
| **Per-user corpus isolation** | Each user has their own binary vector store, HNSW index, and BM25 index |
| **Document versioning** | Re-uploading a changed file bumps the version and soft-deletes old chunks |
| **Citation-backed answers** | Every answer references the source document, page, and section |
| **JWT auth** | Argon2-hashed passwords, HS256 signed tokens, 7-day expiry |

---

## Architecture

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
    ├── vectors.f32          ← raw float32 embedding matrix
    ├── ids.txt              ← chunk IDs (positionally aligned)
    ├── docstore.jsonl       ← chunk metadata (text, page, section, citation)
    ├── manifest.json        ← per-doc version tracking
    ├── deleted_ids.txt      ← soft-deletion tombstones
    └── _hnsw/
        ├── hnsw.bin         ← serialised hnswlib index
        └── hnsw_params.json ← M, ef_construction, ef_search, metric
```

---

## Retrieval Pipeline

At query time, `smart_retrieve()` orchestrates five stages:

```
User Question
     │
     ▼
┌─────────────────────────┐
│  1. Multi-Query Rewrite  │  LLM generates N alternative phrasings
└────────────┬────────────┘
             │  (per query variant)
             ▼
┌────────────────────────────────────────────────┐
│  2. Hybrid Retrieval                           │
│   ├── Dense: HNSW knn_query (hnswlib, cosine)  │
│   └── Sparse: BM25Okapi keyword search         │
│   → min-max normalise → weighted fusion        │
│      score = 0.65 × semantic + 0.35 × keyword  │
└────────────────────┬───────────────────────────┘
                     │  merge best score per row across all query variants
                     ▼
        ┌────────────────────────┐
        │  3. MMR Diversity      │  λ=0.7, sim_threshold=0.92
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
        │  4. Cross-Encoder      │  ms-marco-MiniLM-L-6-v2
        │     Re-ranking         │
        └────────────┬───────────┘
                     ▼
              Top-K Chunks
                     │
                     ▼
        ┌────────────────────────┐
        │  5. LLM Generation     │  Qwen2.5:7B via Ollama
        │     (grounded answer   │  temp=0.2, citations inline
        │      + citations)      │
        └────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI 0.115.6 + Uvicorn |
| Database | PostgreSQL + SQLAlchemy 2.0 (async, asyncpg) |
| Migrations | Alembic |
| Auth | JWT (python-jose) + Argon2 (passlib) |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers 3.3.1 · dim=384 |
| ANN index | hnswlib 0.8.0 (C++ backend, cosine space) |
| Keyword index | rank-bm25 (BM25Okapi) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Qwen2.5:7B-Instruct via Ollama |
| PDF | pypdf 5.2.0 |
| DOCX | python-docx 1.1.2 |
| HTML | BeautifulSoup4 + lxml |
| Tokenisation | tiktoken 0.8.0 |
| Numerics | NumPy 1.26.4 (OpenBLAS) |
| Visualisation | matplotlib 3.8.2 |

---

## Project Structure

```
RAG_Chatbot_Backend/
├── main.py                          # FastAPI app entry point
├── requirements.txt
├── core/
│   ├── config.py                    # Pydantic Settings (.env-driven)
│   ├── security.py                  # JWT + Argon2 helpers
│   └── logging.py
├── api/
│   ├── deps.py                      # get_current_user dependency
│   └── routes/
│       ├── auth.py                  # /auth/register, /auth/login
│       ├── chat.py                  # /chat/query
│       └── documents.py             # /docs/upload, /docs/pasted
├── db/
│   ├── models.py                    # User, Document, Chunk, IngestionJob
│   ├── session.py
│   └── crud.py
├── schemas/                         # Pydantic request/response models
├── services/
│   ├── embeddings.py                # embed_query() / embed_passages()
│   ├── rag.py                       # build_prompt() + generate_answer()
│   ├── local_retriever.py           # filtered subset cosine search
│   ├── Ingestion/
│   │   ├── pipeline.py              # main ingest_bytes() orchestrator
│   │   ├── loaders.py               # pdf/docx/html/txt extraction
│   │   ├── chunking.py              # token + sentence chunking strategies
│   │   ├── cleaning.py              # unicode normalisation
│   │   ├── pdf_blocks.py            # heading/paragraph block segmentation
│   │   ├── pdf_cleaning.py          # header/footer frequency filter
│   │   └── summarize.py             # extractive page summaries
│   ├── corpus/
│   │   ├── corpus_store.py          # append-only binary corpus management
│   │   └── updater.py               # corpus + HNSW rebuild/update
│   ├── hnsw/
│   │   └── hnsw_store.py            # HNSWIndex wrapper + save/load
│   ├── local_vector_store/
│   │   └── store.py                 # LocalVectorStore (NumPy brute-force)
│   ├── bm25/
│   │   └── bm25_store.py            # UserBM25 with persistence
│   └── retrieval/
│       ├── smart_retrieve.py        # top-level retrieval orchestrator
│       ├── hybrid_retriever.py      # HNSW + BM25 fusion
│       ├── rerank.py                # cross-encoder re-ranking
│       ├── diversity.py             # MMR selection
│       ├── query_rewrite.py         # LLM multi-query expansion
│       └── filters.py               # document-scoped row filtering
├── utils/
│   ├── citations.py                 # citation string formatter
│   └── text_sanitize.py
└── scripts/
    ├── eval_hnsw_vs_bruteforce.py   # scalability benchmark (see below)
    ├── build_hnsw.py                # rebuild HNSW index for a user
    └── build_user_corpus_from_existing.py
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL (running locally or via Docker)
- [Ollama](https://ollama.com/) with `qwen2.5:7b-instruct` pulled

```bash
ollama pull qwen2.5:7b-instruct
```

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/RAG_Chatbot_Backend.git
cd RAG_Chatbot_Backend

# 2. Create and activate a virtual environment
python -m venv rag_venv
# Windows
rag_venv\Scripts\activate
# macOS / Linux
source rag_venv/bin/activate

# 3. Install dependencies (CPU-only PyTorch)
pip install -r requirements.txt

# 4. Apply database migrations
alembic upgrade head
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ragdb
DATABASE_URL_SYNC=postgresql+psycopg2://user:password@localhost:5432/ragdb

# Auth
JWT_SECRET=your-secret-key-at-least-32-chars

# Embeddings (local)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384

# LLM (Ollama)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct

# HNSW index
HNSW_M=16
HNSW_EFC=200
HNSW_EFS=80

# Ingestion
ARTIFACTS_DIR=artifacts
MAX_UPLOAD_MB=20
CHUNK_STRATEGY=tokens      # or: sentences
CHUNK_TOKENS=450
CHUNK_OVERLAP=80
TOP_K=6
```

### Running the Server

```bash
uvicorn RAG_Chatbot_Backend.main:app --reload --port 8000
```

Interactive API docs available at `http://localhost:8000/docs`.

---

## API Reference

All protected endpoints require:
```
Authorization: Bearer <access_token>
```

### Authentication

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/auth/register` | `{ "email": "...", "password": "..." }` | Register a new user, returns JWT |
| `POST` | `/auth/login` | `{ "email": "...", "password": "..." }` | Login, returns JWT |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/docs/upload` | Upload a file (`multipart/form-data`, field: `file`; query param: `title`) |
| `POST` | `/docs/pasted` | Ingest pasted text (`{ "title": "...", "text": "..." }`) |

**Supported file types:** `.pdf`, `.docx`, `.html`, `.htm`, `.txt`, `.md`

Re-uploading a changed file automatically increments its version and soft-deletes old chunks.

### Chat

**`POST /chat/query`**

```json
{
  "question": "What were the key findings?",
  "document_ids": ["uuid-1", "uuid-2"],   // optional: restrict to specific docs
  "top_k": 6,
  "use_hybrid": true,
  "use_rerank": true,
  "use_query_rewrite": true,
  "n_rewrites": 3
}
```

**Response:**
```json
{
  "answer": "The key findings were... [Doc: Report 2024, p.4]",
  "chunks": [
    {
      "citation": "Report 2024 · Page 4 · Section: Results",
      "citation_id": "<doc_uuid>:v1:12",
      "text": "The study found that...",
      "score": 0.94,
      "source": "rerank"
    }
  ]
}
```

---

## Benchmark Results

The scalability of the HNSW index versus brute-force search was evaluated using the `eval_hnsw_vs_bruteforce.py` script. Two measurement regimes were used:

- **\[FAIR\]** — Both sides use raw `hnswlib.knn_query()` with identical C++ overhead. The *exact* path sets `ef_search = N`; the *ANN* path uses `ef_search = 200`. This is the pure **algorithmic** comparison.
- **\[PROD\]** — Real production code paths: `LocalVectorStore.search()` (NumPy/BLAS) vs `HNSWIndex.search()` (hnswlib). This reflects actual deployment latency including Python and library call overhead.

> **Hardware:** HP EliteBook 840 G4 · Intel Core i5-7300U (2 cores / 4 threads, 2.60 GHz) · 8 GB DDR4 RAM · NVMe SSD · Windows 10 Pro 64-bit · CPU-only (no GPU) · Python 3.11 · NumPy 1.26.4 (OpenBLAS) · hnswlib 0.8.0
>
> Each reported value is the **median (p50) of 500 iterations** after 50 warm-up runs, single-threaded.

---

### \[FAIR\] Algorithmic Comparison — hnswlib exact vs ANN (ef=200)

> These charts isolate the **pure algorithmic cost** of O(N) exact search vs O(log N) ANN graph traversal. Both sides pay identical hnswlib C++ call overhead, making this a valid apples-to-apples comparison.

| Metric | Chart |
|---|---|
| **p50 Median Latency** | ![FAIR p50 Latency](fair_p50.png) |
| **p95 Tail Latency** | ![FAIR p95 Latency](fair_p95.png) |
| **Queries Per Second** | ![FAIR QPS](fair_qps.png) |

**Key observations:**
- The algorithmic crossover occurs at **N ≈ 500** (slightly above `ef_search = 200`). Below this point, exact search scans fewer vectors than HNSW's fixed candidate budget and is therefore faster.
- Exact search latency scales **linearly** with N: 0.008 ms at N = 10 → 16.4 ms at N = 50,000 (a 2,050× increase for a 5,000× corpus growth).
- ANN latency remains **near-constant** below the crossover and grows only sub-linearly above it: 0.007 ms → 1.26 ms over the same range (a 180× increase).
- **Recall@5 = 100%** is maintained for all corpus sizes up to N = 25,000 with `ef_search = 200`. At N = 50,000 recall drops to 60%, indicating `ef_search` should be raised for very large corpora.
- Peak speedup: **13.8× lower median latency** at N = 25,000.

---

### \[PROD\] Production Paths — LocalVectorStore (BF) vs HNSWIndex (ANN)

> These charts show **real deployment latency** including all Python and library overhead. On this hardware, NumPy's OpenBLAS dispatch carries ~80 µs fixed overhead per call, while hnswlib's C++ entry point carries ~40 µs. This constant 2× gap makes HNSW appear faster even at very small N — an important platform-level caveat when interpreting small-corpus results.

| Metric | Chart |
|---|---|
| **p50 Median Latency** | ![PROD p50 Latency](prod_p50.png) |
| **p95 Tail Latency** | ![PROD p95 Latency](prod_p95.png) |
| **Queries Per Second** | ![PROD QPS](prod_qps.png) |

**Key observations:**
- The BF QPS curve is non-monotone at small N (visible in the QPS chart). This is a direct consequence of the ~80 µs BLAS dispatch floor on the i5-7300U: at N = 10–300, BLAS overhead dominates compute time, flattening BF latency regardless of corpus size. This is a platform artefact, not an algorithmic property — see the \[FAIR\] charts for the true comparison.
- For N ≥ 5,000, HNSW wins decisively in production: **1.76× at N = 5,000, 4.63× at N = 50,000**.
- At N = 50,000, BF p95 tail latency reaches **~12 ms** while HNSW p95 stays bounded at **~3.5 ms** — a 3.4× tail latency advantage relevant for consistent response times under load.
- The production crossover (where HNSW first becomes meaningfully faster, independent of call-overhead effects) sits at approximately **N = 2,000–5,000** on this hardware.

---

### Summary Table

| N | Exact p50 | ANN p50 | Speedup | Recall@5 |
|---|---|---|---|---|
| 10 | 0.008 ms | 0.007 ms | 1.05× | 1.00 |
| 100 | 0.027 ms | 0.029 ms | 0.94× | 1.00 |
| 500 | 0.124 ms | 0.080 ms | **1.56× ◄** | 1.00 |
| 1,000 | 0.244 ms | 0.106 ms | 2.31× | 1.00 |
| 5,000 | 2.154 ms | 0.322 ms | 6.68× | 1.00 |
| 10,000 | 3.453 ms | 0.532 ms | 6.49× | 1.00 |
| 25,000 | 9.071 ms | 0.656 ms | **13.82×** | 1.00 |
| 50,000 | 16.452 ms | 1.259 ms | 13.07× | 0.60 |

*◄ marks the algorithmic crossover point (N ≈ ef_search = 200). All timings from the \[FAIR\] regime.*

---

### Reproducing the Benchmark

```bash
# Run the full sweep (requires a populated user corpus)
python -m RAG_Chatbot_Backend.scripts.eval_hnsw_vs_bruteforce

# Generate charts from the terminal output (no project environment needed)
# 1. Copy the full terminal output
# 2. Paste it into the RAW_OUTPUT string in plot_benchmark.py
# 3. Run:
python plot_benchmark.py
# → produces fair_p50.png, fair_p95.png, fair_qps.png,
#              prod_p50.png, prod_p95.png, prod_qps.png
```

---

## Configuration Reference

All settings are read from `.env` via Pydantic Settings (`core/config.py`).

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | *(required)* | Async PostgreSQL DSN (`postgresql+asyncpg://...`) |
| `DATABASE_URL_SYNC` | *(required)* | Sync DSN for Alembic (`postgresql+psycopg2://...`) |
| `JWT_SECRET` | *(required)* | HS256 signing secret (min 32 chars recommended) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers model name |
| `EMBEDDING_DIM` | `384` | Output embedding dimension |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model tag for generation |
| `HNSW_M` | `16` | Max connections per HNSW layer |
| `HNSW_EFC` | `200` | `ef_construction` — index build quality |
| `HNSW_EFS` | `80` | `ef_search` — query-time recall/speed trade-off |
| `ARTIFACTS_DIR` | `artifacts` | Root directory for per-user corpus files |
| `CHUNK_TOKENS` | `450` | Token window size for chunking |
| `CHUNK_OVERLAP` | `80` | Overlapping tokens between consecutive chunks |
| `CHUNK_STRATEGY` | `tokens` | `tokens` or `sentences` |
| `TOP_K` | `6` | Number of chunks passed to the LLM |
| `MAX_UPLOAD_MB` | `20` | Maximum upload file size |

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/eval_hnsw_vs_bruteforce.py` | Full scalability benchmark: BF vs HNSW across N = 10–50,000. Generates 6 PNG charts. |
| `scripts/build_hnsw.py` | Rebuild the HNSW index for a specific user from their existing corpus. |
| `scripts/build_user_corpus_from_existing.py` | Compact and rebuild a user's corpus store from existing document artifacts (removes deleted chunks). |
| `scripts/eval_vector_store.py` | Standalone benchmark for the LocalVectorStore brute-force search. |

---

## Corpus File Layout

Each user's data lives under `artifacts/user_<uuid>/`:

```
artifacts/user_<uuid>/
└── _corpus/
    ├── vectors.f32          float32 binary matrix, shape (N, 384)
    ├── ids.txt              chunk ID per row: <doc_uuid>:v<ver>:<chunk_idx>
    ├── docstore.jsonl       JSON metadata per chunk (text, page, section, citation)
    ├── meta.json            { "dim": 384, "metric": "cosine", "count": N }
    ├── manifest.json        per-document version and active chunk tracking
    ├── deleted_ids.txt      soft-deletion tombstones
    ├── _hnsw/
    │   ├── hnsw.bin         serialised hnswlib index
    │   └── hnsw_params.json { M, ef_construction, ef_search, metric, seed }
    └── _bm25/
        ├── tokens.jsonl     tokenised chunk texts for BM25 reconstruction
        └── bm25_meta.json   { "count": N }
```
