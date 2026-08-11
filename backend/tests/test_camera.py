import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.camera import Camera
from app.models.user import User
from app.api.dependencies.auth import require_active_user
from main import app

pytestmark = pytest.mark.asyncio

async def test_list_cameras(async_client: AsyncClient, db_session: AsyncSession):
    # Mock auth
    app.dependency_overrides[require_active_user] = lambda: User(id=1, username="test", is_active=True)

    # Setup test cameras
    camera1 = Camera(
        name="Cam 1", camera_code="C-01", location="Zone A", 
        stream_url="rtsp://test/1", is_active=True
    )
    camera2 = Camera(
        name="Cam 2", camera_code="C-02", location="Zone B", 
        stream_url="rtsp://test/2", is_active=False
    )
    db_session.add(camera1)
    db_session.add(camera2)
    await db_session.commit()

    response = await async_client.get("/api/v1/cameras")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["camera_code"] == "C-01"

async def test_get_camera_status(async_client: AsyncClient, db_session: AsyncSession):
    app.dependency_overrides[require_active_user] = lambda: User(id=1, username="test", is_active=True)
    camera1 = Camera(
        name="Cam 1", camera_code="C-03", location="Zone A", 
        stream_url="rtsp://test/1", is_active=True
    )
    db_session.add(camera1)
    await db_session.commit()

    response = await async_client.get(f"/api/v1/cameras/{camera1.id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["camera_code"] == "C-03"
    assert data["fps"] == 0.0
