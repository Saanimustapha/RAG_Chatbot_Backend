from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "rag-backend"
    ENV: str = "dev"
    API_PREFIX: str = "/api"

    DATABASE_URL: str
    DATABASE_URL_SYNC: str

    JWT_SECRET: str
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384

    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"

    HNSW_M: int = 16
    HNSW_EFC: int = 200
    HNSW_EFS: int = 80
    HNSW_METRIC: str = "cosine"

    ARTIFACTS_DIR: str = "artifacts"

    MAX_UPLOAD_MB: int = 20
    CHUNK_STRATEGY: str = "tokens"
    CHUNK_TOKENS: int = 450
    CHUNK_OVERLAP: int = 80
    TOP_K: int = 6

    CORS_ORIGINS: List[str] = ["*"]
    OLLAMA_TIMEOUT_SECONDS: int = 300
    QUERY_REWRITE_TIMEOUT_SECONDS: int = 120

    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_HEADERS_ENABLED: bool = True
    RATE_LIMIT_STORAGE_URI: str = "redis://localhost:6379/1"
    RATE_LIMIT_STRATEGY: str = "fixed-window"
    RATE_LIMIT_KEY_PREFIX: str = "rag-chatbot"

    AUTH_LOGIN_LIMIT: str = "10/minute"
    AUTH_REGISTER_LIMIT: str = "5/minute"
    CHAT_QUERY_LIMIT: str = "30/minute"
    UPLOAD_LIMIT: str = "20/minute"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    @field_validator("JWT_SECRET")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        if not v or len(v.strip()) < 16:
            raise ValueError("JWT_SECRET must be set and at least 16 characters long")
        return v

    @field_validator("CHUNK_STRATEGY")
    @classmethod
    def validate_chunk_strategy(cls, v: str) -> str:
        allowed = {"tokens", "sentences"}
        if v not in allowed:
            raise ValueError(f"CHUNK_STRATEGY must be one of {allowed}")
        return v


settings = Settings()