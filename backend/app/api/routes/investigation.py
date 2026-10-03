import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.database.session import get_db
from app.api.dependencies.auth import get_current_active_user
from app.models.user import User
from app.services.agents.investigation_orchestrator import InvestigationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/investigation", tags=["Agentic Investigation"])

class InvestigationQueryRequest(BaseModel):
    evidence_id: int = Field(..., description="ID of the evidence video to investigate")
    query: str = Field(..., min_length=3, description="Natural language investigation query")

class InvestigationQueryResponse(BaseModel):
    investigation_id: str
    evidence_id: int
    query: str
    status: str
    execution_time_ms: float
    plan: Dict[str, Any]
    progress: list
    events: list
    findings: list
    report_markdown: str

@router.post("/query", response_model=InvestigationQueryResponse)
async def execute_agent_investigation(
    req: InvestigationQueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submits an investigator natural language query to the GuardianEye Multi-Agent System.
    Executes Planner -> Investigator Tool System -> Evidence Verifier -> Forensic Report Generator.
    """
    try:
        res = await InvestigationOrchestrator.run_investigation(
            db=db,
            evidence_id=req.evidence_id,
            query_text=req.query,
            user_id=current_user.id
        )
        return res
    except Exception as e:
        logger.exception(f"[API] Error executing agent investigation for evidence #{req.evidence_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent investigation failed: {str(e)}"
        )
