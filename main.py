from fastapi import FastAPI
from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.api import router as api_router

app = FastAPI(title=settings.APP_NAME)
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/health")
def health():
    return {"status": "ok"}
