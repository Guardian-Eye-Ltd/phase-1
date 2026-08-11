import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User
from app.models.camera import Camera
from app.models.alert import Alert
from app.models.incident import Incident
from app.api.dependencies.auth import require_active_user
from main import app

pytestmark = pytest.mark.asyncio

async def test_get_camera_stats(async_client: AsyncClient, db_session: AsyncSession):
    app.dependency_overrides[require_active_user] = lambda: User(id=1, username="admin", is_active=True)
    
    # Add mock data
    camera1 = Camera(name="C1", camera_code="C-1", location="L1", stream_url="rtsp://test/1", is_active=True)
    camera2 = Camera(name="C2", camera_code="C-2", location="L2", stream_url="rtsp://test/2", is_active=False)
    db_session.add_all([camera1, camera2])
    await db_session.commit()

    response = await async_client.get("/api/v1/cameras/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 2
    assert data["active"] >= 1
    assert data["offline"] >= 1

async def test_get_system_health(async_client: AsyncClient):
    app.dependency_overrides[require_active_user] = lambda: User(id=1, username="admin", is_active=True)
    response = await async_client.get("/api/v1/health/system")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_usage" in data
    assert "memory_usage" in data

async def test_get_alerts(async_client: AsyncClient, db_session: AsyncSession):
    app.dependency_overrides[require_active_user] = lambda: User(id=1, username="admin", is_active=True)
    
    camera1 = Camera(name="C3", camera_code="C-3", location="L3", stream_url="rtsp://test/3", is_active=True)
    db_session.add(camera1)
    await db_session.commit()
    
    alert = Alert(camera_id=camera1.id, alert_type="TEST_ALERT", severity="HIGH", description="Test")
    db_session.add(alert)
    await db_session.commit()

    response = await async_client.get("/api/v1/alerts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["alert_type"] == "TEST_ALERT"

async def test_get_incidents(async_client: AsyncClient, db_session: AsyncSession):
    app.dependency_overrides[require_active_user] = lambda: User(id=1, username="admin", is_active=True)
    
    incident = Incident(title="Test Incident", description="Test Desc", severity="LOW")
    db_session.add(incident)
    await db_session.commit()

    response = await async_client.get("/api/v1/incidents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["title"] == "Test Incident"

async def test_toggle_camera(async_client: AsyncClient, db_session: AsyncSession):
    app.dependency_overrides[require_active_user] = lambda: User(id=1, username="admin", is_active=True)
    
    camera1 = Camera(name="C4", camera_code="C-4", location="L4", stream_url="rtsp://test/4", is_active=True)
    db_session.add(camera1)
    await db_session.commit()

    response = await async_client.post(f"/api/v1/cameras/{camera1.id}/toggle")
    assert response.status_code == 200
    data = response.json()
    assert data["is_active"] is False
    assert data["status"] == "OFFLINE"
