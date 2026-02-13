import requests
from RAG_Chatbot_Backend.core.config import settings

def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join([f"[{c['citation']}]\n{c['text']}" for c in contexts])

    return f"""
You are a helpful assistant. Answer using ONLY the provided context. 
If the answer isn't in the context, say you don't know. 
Cite sources using bracket citations like [weekly_report.pdf p. 3] at the end of sentences.

Context:
{context_block}

Question: {question}
Answer:
""".strip()

def generate_answer(question: str, contexts: list[dict]) -> str:
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

    r = requests.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "").strip()
