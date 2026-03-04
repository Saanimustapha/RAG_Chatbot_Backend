import requests
from RAG_Chatbot_Backend.core.config import settings

def rewrite_queries(question: str, n: int = 3) -> list[str]:
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

    r = requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    text = (r.json().get("message") or {}).get("content", "").strip()

    # tiny safe parse (expects JSON array)
    try:
        import json
        arr = json.loads(text)
        if isinstance(arr, list):
            return [str(x) for x in arr][:n]
    except Exception:
        pass

    # fallback: return original only
    return [question]