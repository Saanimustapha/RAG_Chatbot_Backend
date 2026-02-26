# services/embeddings.py
from sentence_transformers import SentenceTransformer
from RAG_Chatbot_Backend.core.config import settings

_model = SentenceTransformer(settings.EMBEDDING_MODEL)

def embed_passages(texts: list[str]) -> list[list[float]]:
    texts = [f"passage: {t}" for t in texts]
    embs = _model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return embs.tolist()

def embed_query(text: str) -> list[float]:
    text = f"query: {text}"
    emb = _model.encode([text], normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return emb[0].tolist()