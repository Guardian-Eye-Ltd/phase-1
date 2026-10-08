import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import RequireAdmin
from app.database.session import get_db
from app.models.user import User
from app.services.system_reset_service import (
    RESET_CONFIRMATION_PHRASE, ResetBlockedError, SystemResetService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Administration"])


class SystemResetRequest(BaseModel):
    confirmation: str = Field(
        ..., description=f"Must be exactly '{RESET_CONFIRMATION_PHRASE}'."
    )


@router.get("/reset/preview")
async def preview_system_reset(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(RequireAdmin),
):
    """Show exactly what a reset would delete. Deletes nothing."""
    return await SystemResetService.preview(db)


@router.post("/reset")
async def execute_system_reset(
    req: SystemResetRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(RequireAdmin),
):
    """
    Irreversibly delete all evidence, analysis results, derived files and the
    vector index. Users, roles, cameras and the audit log are preserved.
    """
    if req.confirmation != RESET_CONFIRMATION_PHRASE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Confirmation phrase mismatch. Type exactly: {RESET_CONFIRMATION_PHRASE}",
        )
    try:
        return await SystemResetService.reset(db, user_id=current_user.id)
    except ResetBlockedError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
