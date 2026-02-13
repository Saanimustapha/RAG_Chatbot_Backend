from pydantic import BaseModel

class ChatQueryIn(BaseModel):
    question: str
    top_k: int | None = None

class RetrievedChunk(BaseModel):
    citation: str                 # human-friendly: "weekly_report.pdf p. 3"
    citation_id: str | None = None  # internal: "doc_id:chunk_index"
    text: str
    title: str | None = None

class ChatOut(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]
