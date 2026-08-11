import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import get_db
from app.models.user import User
from app.api.dependencies.ws_auth import get_ws_current_user
from app.services.camera_service import CameraService
from app.streaming.stream_manager import stream_manager
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/streams", tags=["Streams"])

@router.websocket("/{camera_id}")
async def stream_camera(
    websocket: WebSocket,
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_ws_current_user)
):
    await websocket.accept()
    
    # 1. Verify camera exists and is active
    camera = await CameraService.get_camera(db, camera_id)
    if not camera or not camera.is_active:
        await websocket.send_json({"type": "error", "message": "Camera not found or inactive"})
        await websocket.close(code=1008)
        return
        
    stream_url = camera.stream_url
    if camera.username and camera.password:
        # Construct authenticated URL if needed (PyAV handles embedded auth well)
        # Assuming format rtsp://user:pass@host/... if not already present
        if "@" not in stream_url and "://" in stream_url:
            proto, rest = stream_url.split("://", 1)
            stream_url = f"{proto}://{camera.username}:{camera.password}@{rest}"
            
    # 2. Get or create worker
    worker = await stream_manager.get_or_create_worker(camera_id, stream_url)
    
    # 3. Register client queue
    queue = asyncio.Queue(maxsize=settings.STREAM_MAX_BUFFER_SIZE)
    worker.add_client(queue)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status", 
            "status": worker.status, 
            "fps": worker.fps
        })
        
        last_status_time = 0
        
        while True:
            # Wait for next frame from worker
            frame_bytes = await queue.get()
            
            # Send binary frame
            await websocket.send_bytes(frame_bytes)
            
            # Periodically send status update
            now = asyncio.get_event_loop().time()
            if now - last_status_time > 1.0:
                await websocket.send_json({
                    "type": "status", 
                    "status": worker.status, 
                    "fps": worker.fps
                })
                last_status_time = now
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from camera {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket stream error for camera {camera_id}: {e}")
    finally:
        worker.remove_client(queue)
        await stream_manager.remove_worker_if_idle(camera_id)
