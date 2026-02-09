from fastapi import FastAPI
from RAG_Chatbot_Backend.core.config import settings
# from RAG_Chatbot_Backend.api import router as api_router
from RAG_Chatbot_Backend.api.routes import auth, chat, documents

app = FastAPI()

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(documents.router)

@app.get("/health")
def health():
    return {"status": "ok"}
