from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "rag-backend"
    ENV: str = "dev"
    API_PREFIX: str = "/api"

    DATABASE_URL: str

    JWT_SECRET: str
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str
    PINECONE_NAMESPACE_PREFIX: str = "org_"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_DIM: int = 1536

    MAX_UPLOAD_MB: int = 20
    CHUNK_TOKENS: int = 450
    CHUNK_OVERLAP: int = 80
    TOP_K: int = 6

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
