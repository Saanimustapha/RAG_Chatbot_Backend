import redis.asyncio as redis
from RAG_Chatbot_Backend.core.config import settings


async def check_rate_limit_redis() -> None:
    client = redis.from_url(settings.RATE_LIMIT_STORAGE_URI, decode_responses=True)
    try:
        await client.ping()
    finally:
        await client.aclose()