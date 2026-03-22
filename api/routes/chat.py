import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from RAG_Chatbot_Backend.api.deps import get_current_user
from RAG_Chatbot_Backend.db.session import get_db
from RAG_Chatbot_Backend.db.models import Chunk, Document
from RAG_Chatbot_Backend.schemas.chat import ChatQueryIn, ChatOut, RetrievedChunk
from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.core.logging import logger
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
    doc_ids = payload.document_ids or ([] if not payload.document_id else [payload.document_id])

    if doc_ids:
        try:
            doc_uuids = [uuid.UUID(str(d)) for d in doc_ids]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid document_id(s).",
            )

        stmt = select(Document.id).where(Document.owner_id == user.id, Document.id.in_(doc_uuids))
        res = await db.execute(stmt)
        allowed = {row[0] for row in res.all()}
        doc_ids = [str(d) for d in allowed]

        if not doc_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No accessible documents found.",
            )

    retrieved = await smart_retrieve(
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

        try:
            parts = citation_id.split(":")
            doc_uuid = uuid.UUID(parts[0])
            doc_version = int(parts[-2][1:]) if len(parts) >= 3 and parts[-2].startswith("v") else None
            chunk_index = int(parts[-1])
        except (ValueError, IndexError):
            logger.warning("Skipping malformed citation_id: %s", citation_id)
            continue

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

    if not contexts:
        return ChatOut(
            answer="I could not find relevant context for that question in your indexed documents.",
            chunks=[],
        )

    answer = await generate_answer(payload.question, contexts)

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