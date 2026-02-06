from fastapi import APIRouter
from RAG_Chatbot_Backend.api.routes import auth, documents, chat

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(documents.router, prefix="/documents", tags=["documents"])
router.include_router(chat.router, prefix="/chat", tags=["chat"])
