from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from RAG_Chatbot_Backend.api.deps import get_current_user
from RAG_Chatbot_Backend.db.session import get_db
from RAG_Chatbot_Backend.db.models import Chunk, Document
from RAG_Chatbot_Backend.schemas.chat import ChatQueryIn, ChatOut, RetrievedChunk
from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.services.rag import generate_answer
from RAG_Chatbot_Backend.utils.citations import format_citation 
from RAG_Chatbot_Backend.services.retrieval.smart_retrieve import smart_retrieve

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/query", response_model=ChatOut)
async def chat_query(
    payload: ChatQueryIn,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    top_k = payload.top_k or settings.TOP_K

    # normalize doc ids (multi + legacy single)
    doc_ids = payload.document_ids or ([] if not payload.document_id else [payload.document_id])

    # ✅ enforce user access at DB level (prevents cross-user doc_id abuse)
    if doc_ids:
        try:
            doc_uuids = [uuid.UUID(str(d)) for d in doc_ids]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid document_id(s).")

        stmt = select(Document.id).where(Document.owner_id == user.id, Document.id.in_(doc_uuids))
        res = await db.execute(stmt)
        allowed = {row[0] for row in res.all()}
        doc_ids = [str(d) for d in allowed]
        if not doc_ids:
            raise HTTPException(status_code=404, detail="No accessible documents found.")

    # smart retrieval returns candidates with corpus_row + metadata (+ possibly text)
    retrieved = smart_retrieve(
        user_id=str(user.id),
        question=payload.question,
        top_k=top_k,
        document_ids=doc_ids if doc_ids else None,
        use_hybrid=payload.use_hybrid,
        use_rerank=payload.use_rerank,
        use_query_rewrite=payload.use_query_rewrite,
        n_rewrites=payload.n_rewrites,
    )

    contexts = []
    for r in retrieved:
        md = r.get("metadata") or {}
        citation_id = md.get("citation") or md.get("chunk_id")
        if not citation_id:
            continue

        display_citation = format_citation(md)

        # Parse "docid:v1:chunk_index"
        try:
            parts = citation_id.split(":")
            doc_uuid = uuid.UUID(parts[0])
            doc_version = int(parts[-2][1:]) if len(parts) >= 3 and parts[-2].startswith("v") else None
            chunk_index = int(parts[-1])
        except Exception:
            continue

        # Fetch chunk text from DB (authoritative)
        stmt = select(Chunk).where(Chunk.document_id == doc_uuid, Chunk.chunk_index == chunk_index)
        if doc_version is not None:
            stmt = stmt.where(Chunk.doc_version == doc_version)

        res = await db.execute(stmt)
        chunk = res.scalar_one_or_none()
        if not chunk:
            continue

        contexts.append({
            "citation": display_citation,
            "citation_id": citation_id,
            "text": chunk.text,
            "title": md.get("title"),
            "score": r.get("rerank_score", r.get("score")),
            "source": r.get("source"),
        })

    answer = generate_answer(payload.question, contexts)

    return ChatOut(
        answer=answer,
        chunks=[
            RetrievedChunk(
                citation=c["citation"],
                citation_id=c["citation_id"],
                text=c["text"],
                title=c.get("title"),
                score=c.get("score"),
                source=c.get("source"),
            )
            for c in contexts
        ],
    )