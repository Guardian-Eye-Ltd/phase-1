from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.database.session import get_db
from app.schemas.alert import AlertResponse
from app.models.alert import Alert
from app.api.dependencies.auth import require_active_user
from app.models.user import User

router = APIRouter(prefix="/alerts", tags=["Alerts"])

@router.get("", response_model=List[AlertResponse], summary="Get recent alerts")
async def get_recent_alerts(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_active_user)
):
    """
    Fetch the most recent system alerts for the dashboard.
    """
    stmt = select(Alert).order_by(Alert.timestamp.desc()).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()
