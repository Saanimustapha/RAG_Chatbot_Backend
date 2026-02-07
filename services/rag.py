from openai import OpenAI
from RAG_Chatbot_Backend.core.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def build_prompt(question: str, contexts: list[dict]) -> list[dict]:
    # contexts: [{text, citation}]
    context_block = "\n\n".join([f"[{c['citation']}]\n{c['text']}" for c in contexts])

    system = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know. "
        "Cite sources using bracket citations like [doc:chunk] at the end of sentences."
    )

    user = f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def generate_answer(question: str, contexts: list[dict]) -> str:
    messages = build_prompt(question, contexts)
    resp = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content
