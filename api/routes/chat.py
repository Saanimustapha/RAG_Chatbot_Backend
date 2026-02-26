from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from RAG_Chatbot_Backend.api.deps import get_current_user
from RAG_Chatbot_Backend.db.session import get_db
from RAG_Chatbot_Backend.db.models import Chunk
from RAG_Chatbot_Backend.schemas.chat import ChatQueryIn, ChatOut, RetrievedChunk
from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.services.embeddings import embed_query
from RAG_Chatbot_Backend.services.local_retriever import query_user_index
from RAG_Chatbot_Backend.services.rag import generate_answer
from RAG_Chatbot_Backend.utils.citations import format_citation 


router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/query", response_model=ChatOut)
async def chat_query(
    payload: ChatQueryIn,
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    top_k = payload.top_k or settings.TOP_K

    # Normalize doc_ids (multi + backward compatible single)
    doc_ids = payload.document_ids or ([] if not payload.document_id else [payload.document_id])

    # Validate UUID formatting early (helps client errors)
    if doc_ids:
        try:
            _ = [uuid.UUID(str(d)) for d in doc_ids]
        except Exception:
            raise HTTPException(status_code=400, detail="One or more document_ids are not valid UUIDs.")

    # Embed query
    q_emb = embed_query(payload.question)

    # Retrieve strictly within the specified docs (if provided)
    matches = query_user_index(
        user_id=str(user.id),
        query_embedding=q_emb,
        top_k=top_k,
        document_ids=doc_ids if doc_ids else None,
        source_type=payload.source_type,
        filename_contains=payload.filename_contains,
    )

    contexts = []

    for m in matches.get("matches", []):
        md = m.get("metadata") or {}
        citation_id = md.get("citation") or md.get("chunk_id")
        if not citation_id:
            continue

        display_citation = format_citation(md)

        # parse "doc_uuid:v1:chunk_index"
        try:
            parts = citation_id.split(":")
            doc_uuid = uuid.UUID(parts[0])

            doc_version = None
            if len(parts) >= 3 and parts[-2].startswith("v"):
                doc_version = int(parts[-2][1:])

            chunk_index = int(parts[-1])
        except Exception:
            continue

        stmt = select(Chunk).where(
            Chunk.document_id == doc_uuid,
            Chunk.chunk_index == chunk_index,
        )
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
            )
            for c in contexts
        ],
    )