from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from RAG_Chatbot_Backend.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
