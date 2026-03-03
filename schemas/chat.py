from pydantic import BaseModel
from typing import Optional, List

class ChatQueryIn(BaseModel):
    question: str
    top_k: int | None = None

    # allow multiple docs; keep legacy single doc
    document_ids: Optional[List[str]] = None
    document_id: Optional[str] = None

    # filters
    source_type: Optional[str] = None
    filename_contains: Optional[str] = None
    tags: Optional[List[str]] = None
    date_from: Optional[str] = None  # ISO date/time string
    date_to: Optional[str] = None

    # retrieval tuning
    use_hybrid: bool = True
    use_rerank: bool = True
    use_query_rewrite: bool = True
    n_rewrites: int = 3

class RetrievedChunk(BaseModel):
    citation: str
    citation_id: str | None = None
    text: str
    title: str | None = None
    score: float | None = None     # expose score
    source: str | None = None      # "hnsw" | "bm25" | "hybrid" | "rerank"

class ChatOut(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]