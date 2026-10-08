import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, ConfigDict

from app.database.session import get_db
from app.api.dependencies.auth import get_current_active_user
from app.models.user import User
from app.services.agents.investigation_orchestrator import InvestigationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/investigation", tags=["Agentic Investigation"])


class InvestigationQueryRequest(BaseModel):
    evidence_id: int = Field(..., description="ID of the evidence video to investigate")
    query: str = Field(..., min_length=3, description="Natural language investigation query")
    analysis_job_id: Optional[int] = Field(
        None,
        description="Optional explicit analysis job id. If omitted, the latest "
                    "completed job for the evidence is used.",
    )


class InvestigationQueryResponse(BaseModel):
    """
    Loose response model — the orchestrator attaches intent-specific extras
    (result, verification, diagnostic, analysis_job_id, ...). Extra fields are
    passed through so clients can keep working as the schema grows.
    """
    model_config = ConfigDict(extra="allow")

    investigation_id: str
    evidence_id: int
    query: str
    status: str
    execution_time_ms: float
    plan: Dict[str, Any] = {}
    progress: List[Any] = []
    events: List[Any] = []
    findings: List[Any] = []
    report_markdown: str


@router.post("/query", response_model=InvestigationQueryResponse)
async def execute_agent_investigation(
    req: InvestigationQueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Submits an investigator natural language query to the GuardianEye intent-first
    multi-agent system. The orchestrator parses intent, resolves exactly one
    analysis job, and dispatches to the correct tool (deterministic DB
    aggregation / EventEngine / hybrid semantic retrieval).
    """
    try:
        res = await InvestigationOrchestrator.run_investigation(
            db=db,
            evidence_id=req.evidence_id,
            query_text=req.query,
            user_id=current_user.id,
            analysis_job_id=req.analysis_job_id,
        )
        return res
    except Exception as e:
        logger.exception(f"[API] Error executing agent investigation for evidence #{req.evidence_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent investigation failed: {str(e)}",
        )
