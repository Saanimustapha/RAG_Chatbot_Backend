from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from RAG_Chatbot_Backend.api.deps import get_current_user
from RAG_Chatbot_Backend.db.session import get_db
from RAG_Chatbot_Backend.db.models import Chunk
from RAG_Chatbot_Backend.schemas.chat import ChatQueryIn, ChatOut, RetrievedChunk
from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.services.embeddings import embed_query
from RAG_Chatbot_Backend.services.pinecone_store import query_vectors
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
    matches = query_vectors(str(user.id), q_emb, top_k=top_k)

    contexts = []
    for m in matches["matches"]:
        md = m.get("metadata") or {}

        citation_id = md.get("citation")  # internal: "{doc_id}:{chunk_index}"
        if not citation_id:
            continue

        # ✅ Human-friendly citation from Pinecone metadata
        display_citation = format_citation(md)

        try:
            doc_id, chunk_index = citation_id.split(":")
            chunk_index = int(chunk_index)
        except Exception:
            continue

        res = await db.execute(
            select(Chunk).where(Chunk.document_id == doc_id, Chunk.chunk_index == chunk_index)
        )
        chunk = res.scalar_one_or_none()
        if not chunk:
            continue

        contexts.append({
            "citation": display_citation,   # ✅ for prompt + response
            "citation_id": citation_id,     # optional
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