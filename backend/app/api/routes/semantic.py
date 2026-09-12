import logging
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.session import get_db
from app.models.user import User
from app.models.evidence import Evidence
from app.models.analysis import AnalysisJob, JobStatus, Track, Keyframe, ActivityInterval, PossibleInteraction
from app.models.semantic import (
    ForensicDocument, VLMObservation, SearchQuery, SearchResult, AIModelExecution
)
from app.models.audit import AuditLog
from app.api.dependencies.auth import get_current_active_user
from app.schemas.semantic import (
    SemanticIndexRequest, SemanticStatusResponse,
    SearchRequest, SearchResponse,
    ForensicDocumentResponse, VLMObservationResponse,
    EnhancedTimelineResponse, EnhancedTimelineEvent
)
from app.services.semantic.indexer import SemanticIndexer
from app.services.semantic.hybrid_search_engine import HybridSearchEngine
from app.services.semantic.document_generator import ForensicDocumentGenerator

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/evidence/{id}/semantic-index", response_model=SemanticStatusResponse)
async def trigger_semantic_indexing(
    id: int,
    req: SemanticIndexRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Triggers asynchronous background semantic indexing (Document Generation -> VLM -> Embeddings -> Vector DB).
    """
    ev_res = await db.execute(select(Evidence).where(Evidence.id == id))
    evidence = ev_res.scalar_one_or_none()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence record not found.")

    # Find latest completed analysis job
    job_res = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.evidence_id == id, AnalysisJob.status == JobStatus.COMPLETED)
        .order_by(AnalysisJob.completed_at.desc())
    )
    job = job_res.scalars().first()

    if not job:
        raise HTTPException(
            status_code=400, 
            detail="No completed Phase 1B analysis job found for this evidence. Please run analysis first."
        )

    # Launch background indexing
    background_tasks.add_task(SemanticIndexer.index_evidence_async, id, job.id)

    # Log audit entry
    audit = AuditLog(
        user_id=current_user.id,
        action="TRIGGER_SEMANTIC_INDEX",
        resource_type="EVIDENCE",
        resource_id=str(id),
        metadata_json=f"Triggered background semantic indexing for evidence #{id}"
    )
    db.add(audit)
    await db.commit()

    return SemanticStatusResponse(
        evidence_id=id,
        status="PROCESSING",
        progress=10.0,
        stage="QUEUED",
        total_documents=0,
        total_vlm_observations=0,
        total_embeddings=0
    )

@router.get("/evidence/{id}/semantic-status", response_model=SemanticStatusResponse)
async def get_semantic_indexing_status(
    id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Poll background semantic indexing stage and percentage progress.
    """
    status_dict = SemanticIndexer.get_indexing_status(id)
    if status_dict["status"] == "NOT_STARTED":
        # Check DB if documents already exist
        doc_count_res = await db.execute(
            select(ForensicDocument).where(ForensicDocument.evidence_id == id)
        )
        docs = doc_count_res.scalars().all()
        if docs:
            vlm_res = await db.execute(
                select(VLMObservation).where(VLMObservation.evidence_id == id)
            )
            vlms = vlm_res.scalars().all()
            return SemanticStatusResponse(
                evidence_id=id,
                status="COMPLETED",
                progress=100.0,
                stage="READY",
                total_documents=len(docs),
                total_vlm_observations=len(vlms),
                total_embeddings=len(docs)
            )

    return SemanticStatusResponse(**status_dict)

@router.post("/evidence/{id}/search", response_model=SearchResponse)
async def search_evidence(
    id: int,
    req: SearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Executes natural language semantic forensic search grounded strictly in evidence observations.
    """
    ev_res = await db.execute(select(Evidence).where(Evidence.id == id))
    if not ev_res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Evidence not found.")

    # Execute Hybrid Search Engine
    search_output = await HybridSearchEngine.execute_search(
        db=db,
        evidence_id=id,
        query_text=req.query,
        user_id=current_user.id,
        top_k=req.top_k
    )

    # Log forensic audit event
    audit = AuditLog(
        user_id=current_user.id,
        action="SEMANTIC_SEARCH",
        resource_type="EVIDENCE",
        resource_id=str(id),
        metadata_json=f"Query: '{req.query}'. Results count: {search_output['total_results']}"
    )
    db.add(audit)
    await db.commit()

    return search_output

@router.get("/evidence/{id}/timeline/enhanced", response_model=EnhancedTimelineResponse)
async def get_enhanced_timeline(
    id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves enhanced chronological forensic event timeline (Clickable events, timestamps, keyframe jump references).
    """
    timeline_events: List[EnhancedTimelineEvent] = []

    # Fetch tracks
    tr_res = await db.execute(select(Track).where(Track.evidence_id == id).order_by(Track.first_seen_timestamp))
    tracks = tr_res.scalars().all()

    for trk in tracks:
        ts_str = ForensicDocumentGenerator.format_timestamp(trk.first_seen_timestamp)
        timeline_events.append(EnhancedTimelineEvent(
            id=f"track_start_{trk.id}",
            timestamp=trk.first_seen_timestamp,
            timestamp_str=ts_str,
            event_type="TRACK_ENTRY",
            title=f"Track {trk.track_number} ({trk.class_name.capitalize()}) Detected",
            description=f"First seen at {ts_str}. Duration: {int(trk.duration)}s across {trk.observation_count} frames.",
            track_id=trk.track_number,
            confidence_score=0.92,
            source_type="OBJECT_DETECTION_TRACKER",
            is_clickable=True
        ))

    # Fetch keyframes
    kf_res = await db.execute(select(Keyframe).where(Keyframe.evidence_id == id).order_by(Keyframe.timestamp))
    keyframes = kf_res.scalars().all()

    for kf in keyframes:
        ts_str = ForensicDocumentGenerator.format_timestamp(kf.timestamp)
        timeline_events.append(EnhancedTimelineEvent(
            id=f"keyframe_{kf.id}",
            timestamp=kf.timestamp,
            timestamp_str=ts_str,
            event_type="KEYFRAME_CAPTURED",
            title=f"Keyframe KF-{kf.id} Extracted",
            description=f"Reason: {kf.selection_reason.value} at frame {kf.frame_number}.",
            keyframe_id=kf.id,
            image_path=kf.image_path,
            confidence_score=0.95,
            source_type="KEYFRAME_EXTRACTOR",
            is_clickable=True
        ))

    # Fetch spatial interactions
    inter_res = await db.execute(select(PossibleInteraction).where(PossibleInteraction.evidence_id == id).order_by(PossibleInteraction.start_time))
    interactions = inter_res.scalars().all()

    for inter in interactions:
        ts_str = ForensicDocumentGenerator.format_timestamp(inter.start_time)
        timeline_events.append(EnhancedTimelineEvent(
            id=f"spatial_interaction_{inter.id}",
            timestamp=inter.start_time,
            timestamp_str=ts_str,
            event_type="SPATIAL_INTERACTION",
            title=f"Spatial Interaction: Track {inter.entity_a_track_id} & Track {inter.entity_b_track_id}",
            description=f"Proximity detected near Track {inter.entity_b_track_id} between {ts_str} and {ForensicDocumentGenerator.format_timestamp(inter.end_time)}.",
            track_id=inter.entity_a_track_id,
            confidence_score=inter.confidence_score,
            source_type="INTERACTION_DETECTOR",
            is_clickable=True
        ))

    # Sort all events chronologically
    timeline_events.sort(key=lambda x: x.timestamp)

    return EnhancedTimelineResponse(
        evidence_id=id,
        total_events=len(timeline_events),
        timeline=timeline_events
    )

@router.get("/evidence/{id}/forensic-documents", response_model=List[ForensicDocumentResponse])
async def get_forensic_documents(
    id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieve evidence-grounded ForensicDocument records generated from Phase 1B observations.
    """
    res = await db.execute(
        select(ForensicDocument).where(ForensicDocument.evidence_id == id).order_by(ForensicDocument.created_at.desc())
    )
    return res.scalars().all()

@router.get("/evidence/{id}/ai-observations", response_model=List[VLMObservationResponse])
async def get_vlm_observations(
    id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieve VLM keyframe visual observations with provenance metadata.
    """
    res = await db.execute(
        select(VLMObservation).where(VLMObservation.evidence_id == id).order_by(VLMObservation.created_at.desc())
    )
    return res.scalars().all()

@router.get("/search/{search_id}")
async def get_search_query_details(
    search_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieve audit history details for a specific search query.
    """
    res = await db.execute(select(SearchQuery).where(SearchQuery.id == search_id))
    sq = res.scalar_one_or_none()
    if not sq:
        raise HTTPException(status_code=404, detail="Search query record not found.")

    res_results = await db.execute(select(SearchResult).where(SearchResult.search_query_id == search_id))
    results = res_results.scalars().all()

    return {
        "search_query_id": sq.id,
        "evidence_id": sq.evidence_id,
        "query_text": sq.query_text,
        "extracted_intent": sq.extracted_intent,
        "execution_time_ms": sq.execution_time_ms,
        "created_at": sq.created_at,
        "results_count": len(results),
        "results": [
            {
                "id": r.id,
                "track_id": r.track_id,
                "keyframe_id": r.keyframe_id,
                "start_time": r.start_time,
                "relevance_score": r.relevance_score,
                "confidence_level": r.confidence_level.value,
                "confidence_reason": r.confidence_reason,
                "summary_text": r.summary_text,
                "why_explanation": r.explanation_json
            } for r in results
        ]
    }
