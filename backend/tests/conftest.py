import atexit
import os
import shutil
import tempfile

# Isolate the whole test session BEFORE any app module is imported. Several
# tests build their own client or use TestClient (which runs the app lifespan
# and init_db), and the analysis runner opens its own SessionLocal — all of
# which previously reached the developer's real guardianeye.db and storage/.
# Environment variables take priority over backend/.env in pydantic-settings.
_TEST_ROOT = tempfile.mkdtemp(prefix="guardianeye_test_").replace("\\", "/")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_ROOT}/test.db"
os.environ["STORAGE_DIR"] = os.path.join(_TEST_ROOT, "storage", "evidence")
os.environ["DERIVED_STORAGE_DIR"] = os.path.join(_TEST_ROOT, "storage", "derived")
os.environ["VECTOR_DB_PATH"] = os.path.join(_TEST_ROOT, "storage", "chroma_db")
atexit.register(shutil.rmtree, _TEST_ROOT, True)

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
from app.core.config import settings as _settings

assert _settings.DATABASE_URL.startswith("sqlite") and _TEST_ROOT in _settings.DATABASE_URL, (
    f"Test session is not isolated: DATABASE_URL={_settings.DATABASE_URL}"
)
assert _TEST_ROOT in os.path.abspath(_settings.STORAGE_DIR).replace("\\", "/"), "Test storage is not isolated"

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
