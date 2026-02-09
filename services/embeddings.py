from sentence_transformers import SentenceTransformer
from RAG_Chatbot_Backend.core.config import settings

_model = SentenceTransformer(settings.EMBEDDING_MODEL)

def embed_texts(texts: list[str]) -> list[list[float]]:
    # normalize_embeddings True improves cosine similarity
    embs = _model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return embs.tolist()

def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
