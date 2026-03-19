import json
import httpx
from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.core.logging import logger


async def rewrite_queries(question: str, n: int = 3) -> list[str]:
    prompt = f"""
Rewrite the following user question into {n} alternative search queries.
Return ONLY a JSON array of strings.

Question: {question}
""".strip()

    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You write search query rewrites."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }

    try:
        async with httpx.AsyncClient(timeout=settings.QUERY_REWRITE_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            text = (response.json().get("message") or {}).get("content", "").strip()

        arr = json.loads(text)
        if isinstance(arr, list):
            cleaned = [str(x).strip() for x in arr if str(x).strip()]
            return cleaned[:n] if cleaned else [question]

    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Query rewrite failed: %s", exc)

    return [question]