import os
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete


from RAG_Chatbot_Backend.services.corpus.updater import update_user_corpus_and_hnsw
from RAG_Chatbot_Backend.services.hnsw.hnsw_store import HNSWParams
from RAG_Chatbot_Backend.core.config import settings  
from RAG_Chatbot_Backend.db.models import Document, Chunk  
from RAG_Chatbot_Backend.services.embeddings import embed_passages
from RAG_Chatbot_Backend.services.Ingestion.hashing import sha256_bytes
from RAG_Chatbot_Backend.utils.text_sanitize import sanitize_text
from RAG_Chatbot_Backend.services.Ingestion.pdf_cleaning import build_repeated_line_filter, clean_pdf_page
from RAG_Chatbot_Backend.services.Ingestion.pdf_blocks import split_into_blocks
from RAG_Chatbot_Backend.services.Ingestion.summarize import page_summary
from RAG_Chatbot_Backend.services.Ingestion.loaders import (
    load_pdf_bytes, load_docx_bytes, load_text_bytes, load_html_bytes
)
from RAG_Chatbot_Backend.services.Ingestion.chunking import chunk_tokens, chunk_sentences
from RAG_Chatbot_Backend.services.Ingestion.artifacts import (
    ensure_dir, write_chunks_jsonl, write_embeddings_bin, write_docstore_jsonl
)

def _pick_chunker():
    if getattr(settings, "CHUNK_STRATEGY", "tokens") == "sentences":
        return chunk_sentences
    return chunk_tokens

async def ingest_bytes(
    *,
    db: AsyncSession,
    owner_id,
    title: str,
    filename: str | None,
    source_type: str,
    file_bytes: bytes,
) -> dict:
    checksum = sha256_bytes(file_bytes)
    now = datetime.utcnow()

    safe_title = (title or "").strip() or (filename or "").strip() or "untitled"
    safe_filename = (filename or "").strip() or safe_title


    # find existing document (same owner + same filename OR title)
    q = select(Document).where(Document.owner_id == owner_id)
    if filename:
        q = q.where(Document.original_filename == filename)
    else:
        q = q.where(Document.title == title, Document.source_type == source_type)

    res = await db.execute(q)
    doc = res.scalars().first()

    if doc and doc.checksum == checksum:
        return {"skipped": True, "document_id": str(doc.id), "version": doc.version, "chunks": 0}

    # create or update doc (bump version)
    if not doc:
        doc = Document(
            owner_id=owner_id,
            title=safe_title,
            source_type=source_type or "",
            original_filename=filename or "",
            checksum=checksum,
            version=1,
            created_at=now,
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
    else:
        doc.version += 1
        doc.checksum = checksum
        doc.updated_at = now
        await db.commit()
        await db.refresh(doc)

        # remove old chunks for this doc (keeping only latest version)
        await db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
        await db.commit()
        await db.refresh(doc)   # re-attach doc to the session cleanly
        db.expire_all()


    # load -> structured blocks
    chunker = _pick_chunker()
    chunk_tokens_n = settings.CHUNK_TOKENS
    overlap = settings.CHUNK_OVERLAP

    chunks_with_meta = []

    if source_type == "pdf":
        pages = load_pdf_bytes(file_bytes)

        # 1) detect repeated header/footer lines across pages
        repeated = build_repeated_line_filter(pages, min_freq_ratio=0.35)

        for p in pages:
            page_num = p["page"]

            # 2) clean the page (remove repeated lines + artifacts)
            cleaned_page = clean_pdf_page(p["text"], repeated)

            if not cleaned_page:
                continue

            # 3) split into blocks by heading/paragraph
            blocks = split_into_blocks(cleaned_page)

            # 4) Optional: add one summary chunk per page (helps conceptual Q&A)
            summary = page_summary(cleaned_page, max_sentences=3)
            if summary:
                chunks_with_meta.append({
                    "text": f"Page {page_num} summary: {summary}",
                    "page_start": page_num,
                    "page_end": page_num,
                    "section": "Page Summary",
                })

            # 5) Chunk each block using your existing token chunker
            for b in blocks:
                section = b.get("section")
                for chunk_text in chunker(b["text"], chunk_tokens_n, overlap):
                    item = {
                        "text": chunk_text,
                        "page_start": page_num,
                        "page_end": page_num,
                    }
                    if section:
                        item["section"] = section
                    chunks_with_meta.append(item)

    elif source_type == "html":
        blocks = load_html_bytes(file_bytes)
        for b in blocks:
            section = (b.get("section") or "").strip() or None
            for chunk_text in chunker(b["text"], chunk_tokens_n, overlap):
                item = {"text": chunk_text}
                if section:  # only include if non-empty
                    item["section"] = section
                chunks_with_meta.append(item)

    elif source_type == "docx":
        text = load_docx_bytes(file_bytes)
        for chunk_text in chunker(text, chunk_tokens_n, overlap):
            chunks_with_meta.append({"text": chunk_text})

    else:  # txt/md/pasted
        text = load_text_bytes(file_bytes)
        for chunk_text in chunker(text, chunk_tokens_n, overlap):
            chunks_with_meta.append({"text": chunk_text})


    texts = [c["text"] for c in chunks_with_meta]
    if not texts:
        return {"skipped": False, "document_id": str(doc.id), "version": doc.version, "chunks": 0}

    # embed
    embeddings = embed_passages(texts)

    # prepare Pinecone vectors + DB rows + artifact rows
    vectors = []
    chunk_rows = []
    chunks_jsonl = []
    docstore_jsonl = []

    for i, (meta, emb) in enumerate(zip(chunks_with_meta, embeddings)):
        citation = f"{doc.id}:v{doc.version}:{i}"
        pinecone_id = f"{doc.id}:{doc.version}:{i}:{uuid4().hex[:8]}"

        raw_md = {
            "document_id": str(doc.id),
            "doc_version": doc.version,
            "chunk_index": i,
            "title": safe_title,
            "source_type": doc.source_type or "unknown",
            "filename": safe_filename,
            "citation": citation,

            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "section": meta.get("section"),
        }

        vectors.append((pinecone_id, emb, raw_md))

        chunk_rows.append(Chunk(
            document_id=doc.id,
            doc_version=doc.version,
            chunk_index=i,
            text=sanitize_text(meta.get("text", "")),      
            page_start=meta.get("page_start"),
            page_end=meta.get("page_end"),
            section=sanitize_text(meta.get("section")),        
            pinecone_id=pinecone_id
        ))

        chunks_jsonl.append({
            "chunk_id": citation,
            "document_id": str(doc.id),
            "doc_version": doc.version,
            "chunk_index": i,
            "text": sanitize_text(meta.get("text", "")),       
            "title": sanitize_text(doc.title),                 
            "source_type": doc.source_type or "",
            "filename": sanitize_text(filename or ""),        
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "section": sanitize_text(meta.get("section")),     
            "ingested_at": now.isoformat(),
            "checksum": checksum,
        })

        docstore_jsonl.append({
            "chunk_id": citation,
            "document_id": str(doc.id),
            "doc_version": doc.version,
            "chunk_index": i,
            "title": sanitize_text(doc.title),
            "text": sanitize_text(meta.get("text", "")),
            "source_type": doc.source_type or "",
            "filename": sanitize_text(filename or ""),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "section": sanitize_text(meta.get("section")),
            })

    # persist chunks to postgres
    db.add_all(chunk_rows)
    await db.commit()

    # write artifacts
    base_dir = os.path.join(settings.ARTIFACTS_DIR, f"user_{owner_id}", f"doc_{doc.id}", f"v{doc.version}")
    ensure_dir(base_dir)
    write_chunks_jsonl(chunks_jsonl, os.path.join(base_dir, "chunks.jsonl"))
    write_docstore_jsonl(docstore_jsonl, os.path.join(base_dir, "docstore.jsonl"))
    write_embeddings_bin(embeddings, os.path.join(base_dir, "embeddings.bin"))

    # --- update per-user corpus + HNSW index ---
    user_dir = Path(settings.ARTIFACTS_DIR) / f"user_{owner_id}"
    doc_leaf = Path(base_dir)  # this is .../doc_<id>/v<version>

    # set these in .env or hardcode for now
    embed_dim = int(getattr(settings, "EMBEDDING_DIM", 384))

    hnsw_params = HNSWParams(
        M=int(getattr(settings, "HNSW_M", 16)),
        ef_construction=int(getattr(settings, "HNSW_EFC", 200)),
        ef_search=int(getattr(settings, "HNSW_EFS", 50)),
        metric=str(getattr(settings, "HNSW_METRIC", "cosine")),
        seed=42,
    )

    update_user_corpus_and_hnsw(
        user_dir=user_dir,
        doc_leaf=doc_leaf,
        embed_dim=embed_dim,
        hnsw_params=hnsw_params,
    )

    return {"skipped": False, "document_id": str(doc.id), "version": doc.version, "chunks": len(texts)}
