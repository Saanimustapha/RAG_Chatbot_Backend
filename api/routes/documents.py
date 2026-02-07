from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4

from RAG_Chatbot_Backend.api.deps import get_current_user
from RAG_Chatbot_Backend.db.session import get_db
from RAG_Chatbot_Backend.db.models import Document, Chunk
from RAG_Chatbot_Backend.schemas.documents import PastedTextIn, IngestOut, DocumentOut
from RAG_Chatbot_Backend.services.extractor import extract_pdf, extract_docx, extract_txt
from RAG_Chatbot_Backend.services.chunker import chunk_text
from RAG_Chatbot_Backend.services.embeddings import embed_texts
from RAG_Chatbot_Backend.services.pinecone_store import upsert_vectors

router = APIRouter()

@router.post("/pasted", response_model=IngestOut)
async def ingest_pasted(
    payload: PastedTextIn,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = Document(owner_id=user.id, title=payload.title, source_type="pasted")
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    chunks = chunk_text(payload.text)
    embeddings = embed_texts(chunks)

    vectors = []
    chunk_rows = []
    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
        pinecone_id = f"{doc.id}:{i}:{uuid4().hex[:8]}"
        vectors.append((pinecone_id, emb, {
            "document_id": str(doc.id),
            "chunk_index": i,
            "title": doc.title,
            "source_type": doc.source_type,
            "citation": f"{doc.id}:{i}",
        }))
        chunk_rows.append(Chunk(
            document_id=doc.id,
            chunk_index=i,
            text=text,
            pinecone_id=pinecone_id
        ))

    upsert_vectors(str(user.id), vectors)
    db.add_all(chunk_rows)
    await db.commit()

    return IngestOut(
        document=DocumentOut(id=doc.id, title=doc.title, source_type=doc.source_type),
        chunks_indexed=len(chunks),
    )

@router.post("/upload", response_model=IngestOut)
async def ingest_upload(
    title: str,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    data = await file.read()
    filename = file.filename or "upload"
    lower = filename.lower()

    if lower.endswith(".pdf"):
        text, _pages = extract_pdf(data)
        source_type = "pdf"
    elif lower.endswith(".docx"):
        text = extract_docx(data)
        source_type = "docx"
    elif lower.endswith(".txt") or lower.endswith(".md"):
        text = extract_txt(data)
        source_type = "txt"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    doc = Document(
        owner_id=user.id, title=title, source_type=source_type, original_filename=filename
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    vectors = []
    chunk_rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        pinecone_id = f"{doc.id}:{i}:{uuid4().hex[:8]}"
        vectors.append((pinecone_id, emb, {
            "document_id": str(doc.id),
            "chunk_index": i,
            "title": doc.title,
            "source_type": doc.source_type,
            "citation": f"{doc.id}:{i}",
            "filename": filename,
        }))
        chunk_rows.append(Chunk(
            document_id=doc.id,
            chunk_index=i,
            text=chunk,
            pinecone_id=pinecone_id
        ))

    upsert_vectors(str(user.id), vectors)
    db.add_all(chunk_rows)
    await db.commit()

    return IngestOut(
        document=DocumentOut(id=doc.id, title=doc.title, source_type=doc.source_type),
        chunks_indexed=len(chunks),
    )
