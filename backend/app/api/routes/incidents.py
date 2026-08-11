from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.database.session import get_db
from app.schemas.incident import IncidentResponse
from app.models.incident import Incident
from app.api.dependencies.auth import require_active_user
from app.models.user import User

router = APIRouter(prefix="/incidents", tags=["Incidents"])

@router.get("", response_model=List[IncidentResponse], summary="Get open incidents")
async def get_incidents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_active_user)
):
    """
    Fetch active incidents.
    """
    stmt = select(Incident).order_by(Incident.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()
