from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "rag-backend"
    ENV: str = "dev"
    API_PREFIX: str = "/api"

    DATABASE_URL: str
    DATABASE_URL_SYNC: str

    JWT_SECRET: str
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384

    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"

    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str
    PINECONE_NAMESPACE_PREFIX: str = "org_"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_DIM: int = 1536

    ARTIFACTS_DIR: str = "artifacts"

    MAX_UPLOAD_MB: int = 20
    CHUNK_STRATEGY: str = "tokens"
    CHUNK_TOKENS: int = 450
    CHUNK_OVERLAP: int = 80
    TOP_K: int = 6

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
