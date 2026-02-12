from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from RAG_Chatbot_Backend.api.deps import get_current_user
from RAG_Chatbot_Backend.db.session import get_db
from RAG_Chatbot_Backend.schemas.documents import PastedTextIn
from RAG_Chatbot_Backend.services.Ingestion.pipeline import ingest_bytes

router = APIRouter(prefix="/docs", tags=["docs"])

@router.post("/pasted")
async def ingest_pasted(
    payload: PastedTextIn,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    data = payload.text.encode("utf-8", errors="ignore")
    result = await ingest_bytes(
        db=db,
        owner_id=user.id,
        title=payload.title,
        filename=None,
        source_type="pasted",
        file_bytes=data,
    )
    return result

@router.post("/upload")
async def ingest_upload(
    title: str,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    data = await file.read()
    filename = file.filename or "upload"
    lower = filename.lower()

    if lower.endswith(".pdf"):
        source_type = "pdf"
    elif lower.endswith(".docx"):
        source_type = "docx"
    elif lower.endswith(".html") or lower.endswith(".htm"):
        source_type = "html"
    elif lower.endswith(".txt") or lower.endswith(".md"):
        source_type = "txt"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    result = await ingest_bytes(
    db=db,
    owner_id=user.id,
    title=title,
    filename=filename,
    source_type=source_type,
    file_bytes=data,
    )
    return result