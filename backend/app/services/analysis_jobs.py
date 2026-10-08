"""
The single source of truth for "which analysis run am I looking at?".

An evidence file can be analysed many times; every derived record belongs to
exactly one (evidence_id, analysis_job_id) pair. Every route, tool and service
that reads analysis-derived data must resolve its job through this module so
two runs can never be merged by accident.
"""
import logging
from typing import Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import AnalysisJob, JobStatus

logger = logging.getLogger(__name__)


class AnalysisJobNotFound(Exception):
    """The requested job does not exist, belongs to other evidence, or is not completed."""


async def get_active_analysis_job(
    db: AsyncSession,
    evidence_id: int,
    requested_job_id: Optional[int] = None,
    *,
    strict: bool = False,
) -> Optional[AnalysisJob]:
    """
    Resolve the analysis job to read from.

    - With `requested_job_id`: that job, if it belongs to `evidence_id` and is
      COMPLETED. If not, raise AnalysisJobNotFound when `strict`, otherwise fall
      back to the latest completed job (logged).
    - Without it: the latest COMPLETED job. Ties on completed_at break on id so
      the answer is deterministic.

    QUEUED / PROCESSING / FAILED / CANCELLED jobs are never returned: a run in
    progress must not hide the last good result, and a failed run has no
    trustworthy data.
    """
    if requested_job_id is not None:
        job = (await db.execute(
            select(AnalysisJob).where(
                AnalysisJob.id == requested_job_id,
                AnalysisJob.evidence_id == evidence_id,
            )
        )).scalars().first()
        if job is not None and job.status == JobStatus.COMPLETED:
            return job
        reason = (
            "not found for this evidence" if job is None
            else f"not completed (status={job.status.value})"
        )
        if strict:
            raise AnalysisJobNotFound(
                f"Analysis job #{requested_job_id} is {reason}."
            )
        logger.warning(
            "[JOB_RESOLVE] evidence=%s requested job #%s is %s; using latest completed.",
            evidence_id, requested_job_id, reason,
        )

    return (await db.execute(
        select(AnalysisJob)
        .where(
            AnalysisJob.evidence_id == evidence_id,
            AnalysisJob.status == JobStatus.COMPLETED,
        )
        .order_by(desc(AnalysisJob.completed_at), desc(AnalysisJob.id))
        .limit(1)
    )).scalars().first()


async def get_active_analysis_job_id(
    db: AsyncSession,
    evidence_id: int,
    requested_job_id: Optional[int] = None,
    *,
    strict: bool = False,
) -> Optional[int]:
    job = await get_active_analysis_job(db, evidence_id, requested_job_id, strict=strict)
    return job.id if job else None
