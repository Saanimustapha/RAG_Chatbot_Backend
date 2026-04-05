from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from slowapi.middleware import SlowAPIMiddleware
from RAG_Chatbot_Backend.core.rate_limit import (
    limiter,
    rate_limit_exception,
    rate_limit_exceeded_handler,
)

from RAG_Chatbot_Backend.api.routes import auth, chat, documents
from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.core.logging import configure_logging, logger
from RAG_Chatbot_Backend.db.session import engine
from RAG_Chatbot_Backend.services.rate_limit_health import check_rate_limit_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting application")

    # startup checks
    required = {
        "DATABASE_URL": settings.DATABASE_URL,
        "DATABASE_URL_SYNC": settings.DATABASE_URL_SYNC,
        "JWT_SECRET": settings.JWT_SECRET,
        "OLLAMA_BASE_URL": settings.OLLAMA_BASE_URL,
        "OLLAMA_MODEL": settings.OLLAMA_MODEL,
    }

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required settings: {', '.join(missing)}")
    
    if settings.RATE_LIMIT_ENABLED:
        await check_rate_limit_redis()

    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))

    yield

    logger.info("Shutting down application")
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Local-first RAG chatbot backend",
    docs_url="/docs" if settings.ENV != "prod" else None,
    redoc_url="/redoc" if settings.ENV != "prod" else None,
    openapi_url="/openapi.json" if settings.ENV != "prod" else None,
    lifespan=lifespan,
)

app.include_router(auth.router, prefix=settings.API_PREFIX)
app.include_router(chat.router, prefix=settings.API_PREFIX)
app.include_router(documents.router, prefix=settings.API_PREFIX)

app.state.limiter = limiter
app.add_exception_handler(rate_limit_exception, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("request.start", extra={"method": request.method, "path": request.url.path})
    response = await call_next(request)
    logger.info(
        "request.end",
        extra={"method": request.method, "path": request.url.path, "status_code": response.status_code},
    )
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})



@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.ENV}
