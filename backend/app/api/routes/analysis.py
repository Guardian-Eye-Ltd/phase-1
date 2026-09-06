import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.database.session import get_db
from app.models.user import User
from app.models.evidence import Evidence
from app.models.audit import AuditLog
from app.models.analysis import (
    AnalysisJob, JobStatus, JobStage, ActivityInterval,
    FrameObservation, Detection, Track, Keyframe, PossibleInteraction
)
from app.core.config import settings
from app.api.routes.auth import get_current_user
from app.services.analysis_runner import run_analysis_job_async, active_job_cancellations
from app.services.pipeline.manifest_generator import ManifestGenerator
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evidence", tags=["Video Analysis Engine"])

class AnalysisCreateRequest(BaseModel):
    sampling_fps: float = Field(default=2.0, ge=0.5, le=10.0)
    confidence_threshold: float = Field(default=0.4, ge=0.1, le=0.9)
    tracking_enabled: bool = Field(default=True)

@router.post("/{evidence_id}/analysis", status_code=status.HTTP_202_ACCEPTED)
async def start_evidence_analysis(
    evidence_id: int,
    request_data: AnalysisCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Creates and enqueues a post-event computer vision video analysis job for an evidence file.
    """
    res = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = res.scalar_one_or_none()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence record not found")

    # Create Analysis Job record
    job = AnalysisJob(
        evidence_id=evidence.id,
        status=JobStatus.QUEUED,
        current_stage=JobStage.VALIDATING,
        progress=0.0,
        sampling_fps=request_data.sampling_fps,
        confidence_threshold=request_data.confidence_threshold,
        tracking_enabled=request_data.tracking_enabled
    )
    db.add(job)

    # Log audit entry
    audit = AuditLog(
        user_id=current_user.id,
        action="ANALYSIS_STARTED",
        resource_type="EVIDENCE",
        resource_id=str(evidence.id),
        metadata_json=f"Analysis job enqueued for evidence '{evidence.original_filename}' (FPS={request_data.sampling_fps}, Conf={request_data.confidence_threshold})"
    )
    db.add(audit)
    await db.commit()
    await db.refresh(job)

    # Dispatch to FastAPI background tasks
    background_tasks.add_task(run_analysis_job_async, job.id)

    return {
        "message": "Video analysis job successfully enqueued",
        "job_id": job.id,
        "evidence_id": evidence.id,
        "status": job.status,
        "progress": job.progress,
        "current_stage": job.current_stage
    }

@router.get("/analysis/{job_id}")
async def get_analysis_job_status(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves real-time status and progress for an active analysis job.
    """
    res = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    return {
        "job_id": job.id,
        "evidence_id": job.evidence_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "progress": job.progress,
        "sampling_fps": job.sampling_fps,
        "confidence_threshold": job.confidence_threshold,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message,
        "model_name": job.model_name,
        "manifest_hash": job.manifest_hash
    }

@router.post("/analysis/{job_id}/cancel")
async def cancel_analysis_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Cancels an active or processing video analysis job.
    """
    res = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        return {"message": f"Job is already in terminal state: {job.status}"}

    active_job_cancellations.add(job_id)
    job.status = JobStatus.CANCELLED
    job.error_message = "Cancelled by user"

    audit = AuditLog(
        user_id=current_user.id,
        action="ANALYSIS_CANCELLED",
        resource_type="ANALYSIS_JOB",
        resource_id=str(job_id),
        metadata_json=f"Investigator cancelled analysis job #{job_id}"
    )
    db.add(audit)
    await db.commit()

    return {"message": "Job cancellation requested", "job_id": job_id}

@router.get("/{evidence_id}/detections")
async def get_evidence_detections(
    evidence_id: int,
    class_name: Optional[str] = Query(None),
    track_id: Optional[int] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves paginated object and person detections for an evidence file.
    """
    query = select(Detection).where(Detection.evidence_id == evidence_id)
    if class_name:
        query = query.where(Detection.class_name == class_name.lower())
    if track_id:
        query = query.where(Detection.track_id == track_id)

    query = query.order_by(Detection.frame_number.asc()).offset(offset).limit(limit)
    res = await db.execute(query)
    detections = res.scalars().all()

    return [
        {
            "id": d.id,
            "frame_number": d.frame_number,
            "timestamp": d.timestamp,
            "class_name": d.class_name,
            "confidence": d.confidence,
            "bbox": [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2],
            "track_id": d.track_id
        }
        for d in detections
    ]

@router.get("/{evidence_id}/tracks")
async def get_evidence_tracks(
    evidence_id: int,
    class_name: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves tracked entity summaries (persons, vehicles, objects) for an evidence file.
    """
    query = select(Track).where(Track.evidence_id == evidence_id)
    if class_name:
        query = query.where(Track.class_name == class_name.lower())

    query = query.order_by(Track.track_number.asc())
    res = await db.execute(query)
    tracks = res.scalars().all()

    return [
        {
            "id": t.id,
            "track_number": t.track_number,
            "class_name": t.class_name,
            "first_seen_timestamp": t.first_seen_timestamp,
            "last_seen_timestamp": t.last_seen_timestamp,
            "first_seen_frame": t.first_seen_frame,
            "last_seen_frame": t.last_seen_frame,
            "duration": t.duration,
            "observation_count": t.observation_count
        }
        for t in tracks
    ]

@router.get("/{evidence_id}/keyframes")
async def get_evidence_keyframes(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves extracted forensic keyframe images with selection reasons and cryptographic SHA-256 hashes.
    """
    res = await db.execute(
        select(Keyframe)
        .where(Keyframe.evidence_id == evidence_id)
        .order_by(Keyframe.timestamp.asc())
    )
    keyframes = res.scalars().all()

    return [
        {
            "id": k.id,
            "frame_number": k.frame_number,
            "timestamp": k.timestamp,
            "selection_reason": k.selection_reason,
            "image_path": k.image_path,
            "sha256_hash": k.sha256_hash,
            "track_ids": k.track_ids
        }
        for k in keyframes
    ]

@router.get("/{evidence_id}/activity")
async def get_evidence_activity_intervals(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves motion activity intervals (LOW ACTIVITY vs HIGH ACTIVITY) across video timeline.
    """
    res = await db.execute(
        select(ActivityInterval)
        .where(ActivityInterval.evidence_id == evidence_id)
        .order_by(ActivityInterval.start_time.asc())
    )
    intervals = res.scalars().all()

    return [
        {
            "id": i.id,
            "start_time": i.start_time,
            "end_time": i.end_time,
            "start_frame": i.start_frame,
            "end_frame": i.end_frame,
            "activity_level": i.activity_level,
            "motion_score": i.motion_score
        }
        for i in intervals
    ]

@router.get("/{evidence_id}/timeline")
async def get_evidence_timeline(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves unified chronological forensic observations timeline.
    """
    # Fetch tracks, keyframes, and interactions
    tr_res = await db.execute(select(Track).where(Track.evidence_id == evidence_id))
    tracks = tr_res.scalars().all()

    kf_res = await db.execute(select(Keyframe).where(Keyframe.evidence_id == evidence_id))
    keyframes = kf_res.scalars().all()

    in_res = await db.execute(select(PossibleInteraction).where(PossibleInteraction.evidence_id == evidence_id))
    interactions = in_res.scalars().all()

    timeline_events = []

    # Track appearance events
    for t in tracks:
        label_str = "Person" if t.class_name == "person" else t.class_name.capitalize()
        timeline_events.append({
            "timestamp": t.first_seen_timestamp,
            "frame_number": t.first_seen_frame,
            "type": "TRACK_APPEARANCE",
            "title": f"{label_str} Track #{t.track_number} Detected",
            "description": f"First observed at {t.first_seen_timestamp:.2f}s (observed for {t.duration:.1f}s)",
            "track_id": t.track_number,
            "class_name": t.class_name
        })

    # Keyframe selection events
    for k in keyframes:
        timeline_events.append({
            "timestamp": k.timestamp,
            "frame_number": k.frame_number,
            "type": "KEYFRAME_SELECTED",
            "title": f"Keyframe #{k.frame_number} ({k.selection_reason.replace('_', ' ')})",
            "description": f"Selected at {k.timestamp:.2f}s. Image Hash: {k.sha256_hash[:8]}...",
            "image_path": k.image_path,
            "sha256_hash": k.sha256_hash
        })

    # Spatial Interaction events
    for inter in interactions:
        timeline_events.append({
            "timestamp": inter.start_time,
            "frame_number": int(inter.start_time * 30),
            "type": "POSSIBLE_INTERACTION",
            "title": f"Possible Interaction (Track #{inter.entity_a_track_id} & Track #{inter.entity_b_track_id})",
            "description": f"Proximity distance {inter.min_distance_or_overlap:.2f}m between {inter.start_time:.2f}s and {inter.end_time:.2f}s",
            "track_id": inter.entity_a_track_id,
            "confidence": inter.confidence_score
        })

    # Sort chronologically by timestamp
    timeline_events.sort(key=lambda x: x["timestamp"])
    return timeline_events

@router.get("/{evidence_id}/manifest")
async def get_evidence_manifest(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves the machine-readable cryptographic analysis manifest for evidence provenance verification.
    """
    job_res = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.evidence_id == evidence_id)
        .order_by(desc(AnalysisJob.created_at))
    )
    job = job_res.scalars().first()
    if not job:
        raise HTTPException(status_code=404, detail="No completed analysis job found for evidence")

    ev_res = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = ev_res.scalar_one_or_none()

    kf_res = await db.execute(select(Keyframe).where(Keyframe.evidence_id == evidence_id))
    keyframes = kf_res.scalars().all()

    kf_dicts = [
        {
            "frame_number": k.frame_number,
            "timestamp": k.timestamp,
            "selection_reason": k.selection_reason,
            "image_path": k.image_path,
            "sha256_hash": k.sha256_hash
        }
        for k in keyframes
    ]

    stats = {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "manifest_hash": job.manifest_hash
    }

    manifest_dict, manifest_hash = ManifestGenerator.generate_manifest(
        evidence_id=evidence_id,
        analysis_job_id=job.id,
        source_sha256=evidence.sha256_hash if evidence else "",
        started_at=job.started_at,
        completed_at=job.completed_at or job.updated_at,
        model_name=job.model_name,
        model_version=job.model_version,
        tracker_algorithm=job.tracker_algorithm,
        sampling_fps=job.sampling_fps,
        confidence_threshold=job.confidence_threshold,
        stats=stats,
        keyframes=kf_dicts
    )

    return {
        "manifest_data": manifest_dict,
        "sha256_hash": manifest_hash
    }

@router.get("/{evidence_id}/derived/keyframes/{filename}")
async def serve_derived_keyframe(
    evidence_id: int,
    filename: str
):
    """
    Serves derived keyframe thumbnail image files securely.
    """
    # Prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(settings.DERIVED_STORAGE_DIR, "keyframes", str(evidence_id), safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Keyframe image file not found")

    return FileResponse(file_path, media_type="image/jpeg")
