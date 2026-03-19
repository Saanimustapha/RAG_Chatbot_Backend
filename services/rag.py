import httpx
from RAG_Chatbot_Backend.core.config import settings


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join([f"[{c['citation']}]\n{c['text']}" for c in contexts])

    return f"""
You are a helpful assistant. Answer using ONLY the provided context.
If the answer isn't in the context, say you don't know.
Cite sources using bracket citations at the end of sentences.

Context:
{context_block}

Question: {question}
Answer:
""".strip()


async def generate_answer(question: str, contexts: list[dict]) -> str:
    prompt = build_prompt(question, contexts)

    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a grounded RAG assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    return (data.get("message") or {}).get("content", "").strip()