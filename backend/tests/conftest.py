import pytest
import pytest_asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from httpx import AsyncClient, ASGITransport

from app.database.base import Base
from app.database.session import get_db
from app.database.init_db import create_tables, seed_roles
from app.models.role import Role
from main import app

# In-memory SQLite async engine for isolated test environment
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test engine and populate schema/seed roles."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    await create_tables(engine)
    
    # Seed roles in test database
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        roles = [
            Role(id=1, role_name="Admin"),
            Role(id=2, role_name="Operator"),
            Role(id=3, role_name="Investigator"),
            Role(id=4, role_name="Viewer")
        ]
        session.add_all(roles)
        await session.commit()

    yield engine
    await engine.dispose()

@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional DB session for testing."""
    async_session = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session

@pytest_asyncio.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide an HTTPX AsyncClient configured with app dependency overrides."""
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
        
    app.dependency_overrides.clear()
