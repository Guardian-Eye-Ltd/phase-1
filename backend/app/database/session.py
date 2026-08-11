from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

# Create async engine with postgresql+asyncpg database URL
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,  # Automatically detect and recover lost connections
)

# Configure the sessionmaker for asynchronous database sessions
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

# FastAPI dependency to yield database sessions per request
async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
