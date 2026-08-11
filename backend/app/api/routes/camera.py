from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.database.session import get_db
from app.models.user import User
from app.api.dependencies.auth import get_current_user, require_active_user
from app.schemas.camera import CameraResponse, CameraStatusResponse
from app.services.camera_service import CameraService
from app.streaming.stream_manager import stream_manager
from datetime import datetime, timezone

router = APIRouter(prefix="/cameras", tags=["Cameras"])

@router.get("", response_model=List[CameraResponse])
async def list_cameras(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_active_user)
):
    """
    Retrieve a paginated list of cameras.
    """
    cameras, _ = await CameraService.list_cameras(db, skip=skip, limit=limit)
    return cameras

@router.get("/stats", summary="Get overall camera statistics")
async def get_camera_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_active_user)
):
    """
    Returns aggregated statistics about the camera network.
    """
    from sqlalchemy import func
    from sqlalchemy.future import select
    from app.models.camera import Camera
    
    # Get total and active counts
    stmt = select(func.count(Camera.id))
    total_cameras = (await db.execute(stmt)).scalar() or 0
    
    stmt_active = select(func.count(Camera.id)).where(Camera.is_active == True)
    active_cameras = (await db.execute(stmt_active)).scalar() or 0
    
    return {
        "total": total_cameras,
        "active": active_cameras,
        "offline": total_cameras - active_cameras
    }

@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_active_user)
):
    """
    Get detailed information about a specific camera.
    """
    camera = await CameraService.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera

@router.get("/{camera_id}/status", response_model=CameraStatusResponse)
async def get_camera_status(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_active_user)
):
    """
    Get live operational status of a camera stream.
    """
    camera = await CameraService.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    # Check if we have an active worker for real-time stats
    worker = stream_manager._workers.get(camera_id)
    
    fps = 0.0
    frame_count = 0
    current_status = camera.status
    
    if worker:
        fps = worker.fps
        frame_count = worker.frame_count
        current_status = worker.status

    return CameraStatusResponse(
        id=camera.id,
        camera_code=camera.camera_code,
        status=current_status,
        is_active=camera.is_active,
        last_seen_at=camera.last_seen_at or datetime.now(timezone.utc),
        fps=fps,
        frame_count=frame_count,
        last_error=None
    )


@router.post("/{camera_id}/toggle", response_model=CameraResponse, summary="Toggle camera stream status")
async def toggle_camera_stream(
    camera_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_active_user)
):
    """
    Toggles the camera's active status (software start/stop).
    """
    camera = await CameraService.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found."
        )
    
    # Toggle active status
    camera.is_active = not camera.is_active
    if not camera.is_active:
        camera.status = "OFFLINE"
    else:
        camera.status = "ONLINE"
        
    await db.commit()
    await db.refresh(camera)
    
    return camera
