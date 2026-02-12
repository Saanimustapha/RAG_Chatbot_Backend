import os
from datetime import datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from core.config import settings  # adjust import
from db.models import Document, Chunk  # adjust import
from services.embeddings import embed_texts
from services.pinecone_store import upsert_vectors
from hashing import sha256_bytes
from loaders import (
    load_pdf_bytes, load_docx_bytes, load_text_bytes, load_html_bytes
)
from chunking import chunk_tokens, chunk_sentences
from artifacts import (
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

    # find existing document (same owner + same filename OR title)
    q = select(Document).where(Document.owner_id == owner_id)
    if filename:
        q = q.where(Document.original_filename == filename)
    else:
        q = q.where(Document.title == title)

    res = await db.execute(q)
    doc = res.scalar_one_or_none()

    if doc and doc.checksum == checksum:
        return {"skipped": True, "document_id": str(doc.id), "version": doc.version, "chunks": 0}

    # create or update doc (bump version)
    if not doc:
        doc = Document(
            owner_id=owner_id,
            title=title,
            source_type=source_type,
            original_filename=filename,
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
        # NOTE: You may also want to delete old vectors in Pinecone later.

    # load -> structured blocks
    chunker = _pick_chunker()
    chunk_tokens_n = settings.CHUNK_TOKENS
    overlap = settings.CHUNK_OVERLAP

    chunks_with_meta = []

    if source_type == "pdf":
        pages = load_pdf_bytes(file_bytes)
        for p in pages:
            page_num = p["page"]
            for chunk_text in chunker(p["text"], chunk_tokens_n, overlap):
                chunks_with_meta.append({
                    "text": chunk_text,
                    "page_start": page_num,
                    "page_end": page_num,
                    "section": None,
                })
    elif source_type == "html":
        blocks = load_html_bytes(file_bytes)
        for b in blocks:
            for chunk_text in chunker(b["text"], chunk_tokens_n, overlap):
                chunks_with_meta.append({
                    "text": chunk_text,
                    "page_start": None,
                    "page_end": None,
                    "section": b.get("section"),
                })
    elif source_type == "docx":
        text = load_docx_bytes(file_bytes)
        for chunk_text in chunker(text, chunk_tokens_n, overlap):
            chunks_with_meta.append({"text": chunk_text, "page_start": None, "page_end": None, "section": None})
    else:  # txt/md/pasted
        text = load_text_bytes(file_bytes)
        for chunk_text in chunker(text, chunk_tokens_n, overlap):
            chunks_with_meta.append({"text": chunk_text, "page_start": None, "page_end": None, "section": None})

    texts = [c["text"] for c in chunks_with_meta]
    if not texts:
        return {"skipped": False, "document_id": str(doc.id), "version": doc.version, "chunks": 0}

    # embed
    embeddings = embed_texts(texts)

    # prepare Pinecone vectors + DB rows + artifact rows
    vectors = []
    chunk_rows = []
    chunks_jsonl = []
    docstore_jsonl = []

    for i, (meta, emb) in enumerate(zip(chunks_with_meta, embeddings)):
        citation = f"{doc.id}:{i}"
        pinecone_id = f"{doc.id}:{doc.version}:{i}:{uuid4().hex[:8]}"

        vectors.append((pinecone_id, emb, {
            "document_id": str(doc.id),
            "doc_version": doc.version,
            "chunk_index": i,
            "title": doc.title,
            "source_type": doc.source_type,
            "filename": filename,
            "citation": citation,
            "page_start": meta["page_start"],
            "page_end": meta["page_end"],
            "section": meta["section"],
        }))

        chunk_rows.append(Chunk(
            document_id=doc.id,
            doc_version=doc.version,
            chunk_index=i,
            text=meta["text"],
            page_start=meta["page_start"],
            page_end=meta["page_end"],
            section=meta["section"],
            pinecone_id=pinecone_id
        ))

        chunks_jsonl.append({
            "chunk_id": citation,
            "document_id": str(doc.id),
            "doc_version": doc.version,
            "chunk_index": i,
            "text": meta["text"],
            "title": doc.title,
            "source_type": doc.source_type,
            "filename": filename,
            "page_start": meta["page_start"],
            "page_end": meta["page_end"],
            "section": meta["section"],
            "ingested_at": now.isoformat(),
            "checksum": checksum,
        })

        docstore_jsonl.append({
            "chunk_id": citation,
            "document_id": str(doc.id),
            "doc_version": doc.version,
            "chunk_index": i,
            "title": doc.title,
            "source_type": doc.source_type,
            "filename": filename,
            "page_start": meta["page_start"],
            "page_end": meta["page_end"],
            "section": meta["section"],
        })

    # upsert to pinecone
    upsert_vectors(str(owner_id), vectors)

    # persist chunks to postgres
    db.add_all(chunk_rows)
    await db.commit()

    # write artifacts
    base_dir = os.path.join(settings.ARTIFACTS_DIR, f"user_{owner_id}", f"doc_{doc.id}", f"v{doc.version}")
    ensure_dir(base_dir)
    write_chunks_jsonl(chunks_jsonl, os.path.join(base_dir, "chunks.jsonl"))
    write_docstore_jsonl(docstore_jsonl, os.path.join(base_dir, "docstore.jsonl"))
    write_embeddings_bin(embeddings, os.path.join(base_dir, "embeddings.bin"))

    return {"skipped": False, "document_id": str(doc.id), "version": doc.version, "chunks": len(texts)}
