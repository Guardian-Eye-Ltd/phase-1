"""Investigation-layer entry point; delegates to the canonical resolver."""
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.analysis_jobs import get_active_analysis_job_id


async def resolve_analysis_job(
    db: AsyncSession,
    evidence_id: int,
    requested_job_id: Optional[int] = None,
) -> Optional[int]:
    return await get_active_analysis_job_id(db, evidence_id, requested_job_id)
