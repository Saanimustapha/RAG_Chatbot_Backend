from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from RAG_Chatbot_Backend.api.deps import get_current_user
from RAG_Chatbot_Backend.db.session import get_db
from RAG_Chatbot_Backend.schemas.documents import PastedTextIn
from RAG_Chatbot_Backend.services.Ingestion.pipeline import ingest_bytes
from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.core.rate_limit import limiter


router = APIRouter(prefix="/docs", tags=["docs"])


@router.post("/pasted")
@limiter.limit(lambda: settings.UPLOAD_LIMIT)
async def ingest_pasted(
    request: Request,
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
@limiter.limit(lambda: settings.UPLOAD_LIMIT)
async def ingest_upload(
    request: Request,
    title: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    data = await file.read()

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max size of {settings.MAX_UPLOAD_MB} MB",
        )

    filename = (file.filename or "upload").strip()
    lower = filename.lower()

    allowed = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".html": "html",
        ".htm": "html",
        ".txt": "txt",
        ".md": "txt",
    }

    source_type = None
    for ext, mapped in allowed.items():
        if lower.endswith(ext):
            source_type = mapped
            break

    if not source_type:
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