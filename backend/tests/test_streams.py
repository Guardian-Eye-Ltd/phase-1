import pytest
import asyncio
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from app.streaming.stream_manager import stream_manager
from app.models.camera import Camera
from sqlalchemy.ext.asyncio import AsyncSession
from main import app # assuming we can import app

pytestmark = pytest.mark.asyncio

async def test_websocket_auth_failure(db_session: AsyncSession):
    # Test websocket connection without token
    # We use TestClient from httpx for WS testing but starlette TestClient is better for WS
    from fastapi.testclient import TestClient
    sync_client = TestClient(app)
    
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with sync_client.websocket_connect("/api/v1/streams/1") as websocket:
            pass
    assert exc_info.value.code == 1008 # Policy violation (no token)

# A simple mock worker test
async def test_stream_manager_creates_worker():
    worker = await stream_manager.get_or_create_worker(999, "rtsp://mock")
    assert worker.camera_id == 999
    assert worker.status == "CONNECTING"
    assert worker.is_running == True
    
    await stream_manager.shutdown_all()
    assert len(stream_manager._workers) == 0
