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

    # Get chunk texts (we store text in Postgres)
    contexts = []
    for m in matches["matches"]:
        md = m.get("metadata") or {}
        citation = md.get("citation", "unknown")
        # citation format: "{doc_id}:{chunk_index}"
        try:
            doc_id, chunk_index = citation.split(":")
            chunk_index = int(chunk_index)
            res = await db.execute(
                select(Chunk).where(Chunk.document_id == doc_id, Chunk.chunk_index == chunk_index)
            )
            chunk = res.scalar_one_or_none()
            if chunk:
                contexts.append({
                    "citation": citation,
                    "text": chunk.text,
                    "title": md.get("title"),
                })
        except Exception:
            continue

    answer = generate_answer(payload.question, contexts)

    return ChatOut(
        answer=answer,
        chunks=[RetrievedChunk(citation=c["citation"], text=c["text"], title=c.get("title")) for c in contexts],
    )
