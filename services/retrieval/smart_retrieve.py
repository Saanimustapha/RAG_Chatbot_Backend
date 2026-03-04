import json
import numpy as np
from pathlib import Path
from typing import Any, Optional

from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.services.embeddings import embed_query
from RAG_Chatbot_Backend.services.retrieval.hybrid_retriever import hybrid_retrieve, _user_corpus_dir, _load_meta, _load_vectors
from RAG_Chatbot_Backend.services.retrieval.diversity import mmr_select
from RAG_Chatbot_Backend.services.retrieval.query_rewrite import rewrite_queries
from RAG_Chatbot_Backend.services.retrieval.rerank import rerank

def smart_retrieve(
    *,
    user_id: str,
    question: str,
    top_k: int,
    document_ids: Optional[list[str]] = None,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    use_query_rewrite: bool = True,
    n_rewrites: int = 3,
) -> list[dict[str, Any]]:
    # multi-query expansion
    queries = [question]
    if use_query_rewrite:
        queries = rewrite_queries(question, n=n_rewrites)
        if question not in queries:
            queries = [question] + queries

    # candidates from multiple queries merged by best score
    merged_by_row: dict[int, dict[str, Any]] = {}

    for qtext in queries:
        q_emb = embed_query(qtext)

        if use_hybrid:
            cands = hybrid_retrieve(
                user_id=user_id,
                query_emb=q_emb,
                query_text=qtext,
                top_k=top_k,
                document_ids=document_ids,
            )
        else:
            # fallback: semantic-only (reuse hybrid_retrieve with keyword weight 0)
            cands = hybrid_retrieve(
                user_id=user_id,
                query_emb=q_emb,
                query_text=qtext,
                top_k=top_k,
                document_ids=document_ids,
                w_sem=1.0,
                w_kw=0.0,
            )

        for c in cands:
            r = c["corpus_row"]
            if r not in merged_by_row or c["score"] > merged_by_row[r]["score"]:
                merged_by_row[r] = c

    candidates = list(merged_by_row.values())
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # diversity / dedup
    corpus_dir = _user_corpus_dir(user_id)
    meta = _load_meta(corpus_dir)
    dim = int(meta["dim"])
    vectors = _load_vectors(corpus_dir, dim)

    diverse = mmr_select(candidates, vectors=vectors, top_k=max(top_k * 3, top_k))

    # attach text if present in metadata (optional)
    # (best practice: store text in docstore.jsonl)
    for c in diverse:
        md = c.get("metadata") or {}
        c["text"] = md.get("text", "")

    # rerank
    if use_rerank:
        return rerank(question, diverse, top_k=top_k)

    return diverse[:top_k]