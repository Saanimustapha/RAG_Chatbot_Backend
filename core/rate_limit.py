from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from RAG_Chatbot_Backend.core.config import settings


def rate_limit_key(request: Request) -> str:
    """
    Prefer authenticated user identity when available.
    Fall back to client IP for anonymous endpoints.
    """
    user = getattr(request.state, "user", None)
    if user is not None and getattr(user, "id", None):
        return f"user:{user.id}"

    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
        if client_ip:
            return f"ip:{client_ip}"

    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


limiter = Limiter(
    key_func=rate_limit_key,
    enabled=settings.RATE_LIMIT_ENABLED,
    headers_enabled=settings.RATE_LIMIT_HEADERS_ENABLED,
    strategy=settings.RATE_LIMIT_STRATEGY,
    storage_uri=settings.RATE_LIMIT_STORAGE_URI,
    key_prefix=settings.RATE_LIMIT_KEY_PREFIX,
)

upload_shared_limit = limiter.shared_limit(
    lambda: settings.UPLOAD_LIMIT,
    scope="document-ingestion",
)

rate_limit_exceeded_handler = _rate_limit_exceeded_handler
rate_limit_exception = RateLimitExceeded