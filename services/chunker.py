import tiktoken
from RAG_Chatbot_Backend.core.config import settings

enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str, chunk_tokens: int | None = None, overlap: int | None = None) -> list[str]:
    chunk_tokens = chunk_tokens or settings.CHUNK_TOKENS
    overlap = overlap or settings.CHUNK_OVERLAP

    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(tokens):
            break
    return chunks
