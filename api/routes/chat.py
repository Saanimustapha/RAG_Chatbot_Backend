from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

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
    db: AsyncSession = Depends(get_db),
):
    top_k = payload.top_k or settings.TOP_K
    q_emb = embed_query(payload.question)
    matches = query_user_index(str(user.id), q_emb, top_k=top_k)

    contexts = []
    for m in matches["matches"]:
        md = m.get("metadata") or {}

        citation_id = md.get("citation") or md.get("chunk_id")
        if not citation_id:
            continue

        display_citation = format_citation(md)

        try:
            parts = citation_id.split(":")
            doc_id = parts[0]
            # supports "doc_id:v1:264"
            doc_version = None
            if len(parts) >= 3 and parts[-2].startswith("v"):
                doc_version = int(parts[-2][1:])
            chunk_index = int(parts[-1])
        except Exception:
            continue

        stmt = select(Chunk).where(
            Chunk.document_id == doc_id,
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
                citation_id=c.get("citation_id"),
                text=c["text"],
                title=c.get("title"),
            )
            for c in contexts
        ],
    )