from openai import OpenAI
from RAG_Chatbot_Backend.core.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
