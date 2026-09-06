from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

is_sqlite = settings.DATABASE_URL.startswith("sqlite")
engine_kwargs = {"echo": False, "future": True}
if not is_sqlite:
    engine_kwargs["pool_pre_ping"] = True

engine = create_async_engine(
    settings.DATABASE_URL,
    **engine_kwargs
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
