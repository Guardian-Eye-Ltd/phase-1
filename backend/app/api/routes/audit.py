from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.database.session import get_db
from app.schemas.audit import AuditLogListResponse, AuditLogResponse
from app.services.audit_service import AuditService
from app.api.dependencies.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/audit", tags=["Audit & Forensics Chain of Custody"])

@router.get(
    "",
    response_model=AuditLogListResponse,
    summary="Query investigator chain-of-custody audit logs"
)
async def list_audit_logs(
    page: int = Query(1, ge=1, description="Page index (1-based)"),
    size: int = Query(50, ge=1, le=200, description="Items per page"),
    action: Optional[str] = Query(None, description="Filter by action (LOGIN, UPLOAD_EVIDENCE, VIEW_EVIDENCE, etc.)"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    skip = (page - 1) * size
    logs, total = await AuditService.get_audit_logs(
        db=db,
        skip=skip,
        limit=size,
        action_filter=action,
        user_id_filter=user_id
    )

    return AuditLogListResponse(
        items=[AuditLogResponse.model_validate(log) for log in logs],
        total=total,
        page=page,
        size=size
    )
