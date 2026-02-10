from pydantic import BaseModel
from uuid import UUID

class PastedTextIn(BaseModel):
    title: str | None = None
    text: str

class DocumentOut(BaseModel):
    id: UUID
    title: str | None = None
    source_type: str

class IngestOut(BaseModel):
    document: DocumentOut
    chunks_indexed: int
