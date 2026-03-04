from typing import Any
from sentence_transformers import CrossEncoder

# lightweight, strong baseline
_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not candidates:
        return []

    pairs = [(question, (c.get("text") or "")) for c in candidates]
    scores = _MODEL.predict(pairs)  # higher is better

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
        c["source"] = "rerank"

    candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return candidates[:top_k]