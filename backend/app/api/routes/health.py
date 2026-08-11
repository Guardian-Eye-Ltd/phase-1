from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
import psutil

from app.database.session import get_db
from app.api.dependencies.auth import require_active_user
from app.models.user import User

router = APIRouter(prefix="/health", tags=["System Health"])

@router.get("/system", summary="Get system health metrics")
async def get_system_health(
    current_user: User = Depends(require_active_user)
):
    """
    Returns basic system CPU, memory, and disk usage for the dashboard.
    """
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory.percent,
        "memory_total": memory.total,
        "memory_used": memory.used,
        "disk_usage": disk.percent,
        "disk_total": disk.total,
        "disk_used": disk.used
    }
