import re
from RAG_Chatbot_Backend.services.chunker import chunk_text  

_sentence_re = re.compile(r"(?<=[.!?])\s+")

def chunk_tokens(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    return chunk_text(text, chunk_tokens=chunk_tokens, overlap=overlap)

def chunk_sentences(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    """
    Sentence-based packing into token windows.
    Split sentences then join until token chunker size.
    We reuse token chunker by packing paragraphs, but keep sentence boundaries cleaner.
    """
    sentences = [s.strip() for s in _sentence_re.split(text) if s.strip()]
    if not sentences:
        return []

    # pack sentences into rough blocks (by character length heuristic)
    blocks = []
    buf = []
    buf_len = 0
    target_chars = chunk_tokens * 4  # rough heuristic: 1 token ~ 4 chars
    for s in sentences:
        if buf_len + len(s) + 1 > target_chars and buf:
            blocks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(s)
        buf_len += len(s) + 1
    if buf:
        blocks.append(" ".join(buf))

    # then token-chunk each block
    chunks = []
    for b in blocks:
        chunks.extend(chunk_text(b, chunk_tokens=chunk_tokens, overlap=overlap))
    return chunks
