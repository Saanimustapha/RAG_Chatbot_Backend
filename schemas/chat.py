from pydantic import BaseModel

class ChatQueryIn(BaseModel):
    question: str
    top_k: int | None = None

class RetrievedChunk(BaseModel):
    citation: str
    text: str
    title: str | None = None

class ChatOut(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]
