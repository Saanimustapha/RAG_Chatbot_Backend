import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from RAG_Chatbot_Backend.db.base import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    documents: Mapped[list["Document"]] = relationship(back_populates="owner")

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # pdf, docx, txt, pasted
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    owner: Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # citation metadata
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # vector pointer info
    pinecone_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    doc_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    section: Mapped[str | None] = mapped_column(String(255), nullable=True)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")  # queued, processing, done, failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
