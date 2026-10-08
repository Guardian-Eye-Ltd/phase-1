import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, delete

from app.database.session import get_db
from app.models.user import User
from app.models.evidence import Evidence, EvidenceStatus
from app.models.audit import AuditLog
from app.models.analysis import (
    AnalysisJob, JobStatus, JobStage, ActivityInterval,
    FrameObservation, Detection, Track, Keyframe, PossibleInteraction
)
from app.core.config import settings
from app.core.permissions import RequireAdmin
from app.api.routes.auth import get_current_user
from app.services.analysis_runner import run_analysis_job_async, active_job_cancellations
from app.services.analysis_jobs import AnalysisJobNotFound, get_active_analysis_job
from app.services.analysis_statistics import compute_analysis_statistics
from app.services.pipeline.manifest_generator import ManifestGenerator
from app.services.system_reset_service import ResetBlockedError, SystemResetService
from pydantic import BaseModel, Field

JOB_ID_QUERY = Query(
    None,
    description="Analysis run to read. Defaults to the latest completed run for this evidence.",
)

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

async def _get_latest_completed_job(
    evidence_id: int, db: AsyncSession, job_id: Optional[int] = None
) -> Optional[AnalysisJob]:
    """
    The analysis run a read route should use. An explicit job_id that is wrong
    for this evidence is a 404, never a silent fallback to another run.
    """
    try:
        return await get_active_analysis_job(db, evidence_id, job_id, strict=job_id is not None)
    except AnalysisJobNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


class ResetAnalysisRequest(BaseModel):
    confirmation: str = Field(..., description="Must be exactly 'RESET EVIDENCE <evidence_id>'.")


@router.post("/{evidence_id}/reset-analysis", status_code=status.HTTP_200_OK)
async def reset_evidence_analysis(
    evidence_id: int,
    req: ResetAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(RequireAdmin),
):
    """
    Admin only. Deletes every analysis run of one evidence file and everything
    derived from them (DB rows, keyframes, sealed manifests, vector index). The
    evidence record and original video are kept so it can be re-analysed.
    """
    expected = f"RESET EVIDENCE {evidence_id}"
    if req.confirmation != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Confirmation phrase mismatch. Type exactly: {expected}",
        )
    try:
        return await SystemResetService.reset_evidence_analysis(db, evidence_id, current_user.id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ResetBlockedError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/{evidence_id}/statistics")
async def get_analysis_statistics(
    evidence_id: int,
    job_id: Optional[int] = JOB_ID_QUERY,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Canonical AnalysisStatistics for one analysis run. Stored-entity counts are
    recomputed from the database; run-time-only counters are None for runs that
    predate their recording.
    """
    job = await _get_latest_completed_job(evidence_id, db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No completed analysis job for this evidence.")
    return (await compute_analysis_statistics(db, job)).to_dict()


_VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "van", "bicycle")
_PERSON_FIELDS = ["upper_garment_presence", "upper_garment_color", "upper_garment_type",
                  "lower_garment_color", "headwear", "carries_bag", "carried_item"]
_VEHICLE_FIELDS = ["vehicle_color", "vehicle_body_style", "license_plate_text"]


@router.get("/{evidence_id}/entities")
async def get_entities(
    evidence_id: int,
    job_id: Optional[int] = JOB_ID_QUERY,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Every tracked person and vehicle with its temporally aggregated attributes
    (value, confidence, status, frame support) and, for anything not
    determined, the reason — so "unknown" is always explained.
    """
    from app.models.observation import TrackAttributeAggregate
    from app.services.pipeline.alpr_service import MIN_VEHICLE_WIDTH_FOR_OCR_PX
    from app.services.pipeline.capability_registry import get_capability

    job = await _get_latest_completed_job(evidence_id, db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No completed analysis job for this evidence.")
    evidence = (await db.execute(select(Evidence).where(Evidence.id == evidence_id))).scalars().first()
    frame_w = None
    if evidence and evidence.resolution and "x" in evidence.resolution:
        try:
            frame_w = int(evidence.resolution.split("x")[0])
        except ValueError:
            frame_w = None

    tracks = (await db.execute(
        select(Track).where(Track.analysis_job_id == job.id).order_by(Track.track_number)
    )).scalars().all()
    aggs = (await db.execute(
        select(TrackAttributeAggregate).where(TrackAttributeAggregate.analysis_job_id == job.id)
    )).scalars().all()
    by_track: Dict[int, Dict[str, Any]] = {}
    for a in aggs:
        by_track.setdefault(a.track_number, {})[a.attribute] = {
            "value": a.value,
            "confidence": round(a.confidence, 3),
            "status": a.status.value,
            "supporting_count": a.supporting_count,
            "observation_count": a.observation_count,
            "withheld_reason": (
                (a.confidence_breakdown or {}).get("contradicted_by")
                or (a.confidence_breakdown or {}).get("withheld_reason")
                or ("frames disagree or confidence too low" if a.status.value == "WITHHELD" else None)
            ),
        }

    max_width = dict((await db.execute(
        select(Detection.track_id, func.max(Detection.bbox_x2 - Detection.bbox_x1))
        .where(Detection.analysis_job_id == job.id, Detection.track_id.isnot(None))
        .group_by(Detection.track_id)
    )).all())

    make_cap = get_capability("vehicle_make")
    plate_cap = get_capability("license_plate_text")
    garment_cap = get_capability("person_garment_color")

    entities = []
    for t in tracks:
        attrs = by_track.get(t.track_number, {})
        not_determined: Dict[str, str] = {}
        if t.class_name == "person":
            kind, fields = "PERSON", _PERSON_FIELDS
            if garment_cap["state"] == "NOT_AVAILABLE":
                not_determined["clothing"] = garment_cap["reason"]
            else:
                if "upper_garment_presence" not in attrs and "upper_garment_color" not in attrs:
                    not_determined["upper_garment"] = "upper body not visible clearly enough in any sampled frame"
                if "lower_garment_color" not in attrs:
                    not_determined["lower_garment"] = "legs not visible (knees/ankles not detected) or bare"
            if "carried_item" not in attrs:
                not_determined["carried_item"] = "no bag detected with this person (not proof that none was carried)"
        elif t.class_name in _VEHICLE_CLASSES:
            kind, fields = "VEHICLE", _VEHICLE_FIELDS
            if "license_plate_text" not in attrs:
                width_px = int(max_width.get(t.track_number, 0) * frame_w) if frame_w else None
                if plate_cap["state"] == "NOT_AVAILABLE":
                    not_determined["license_plate_text"] = plate_cap["reason"]
                elif width_px is not None and width_px < MIN_VEHICLE_WIDTH_FOR_OCR_PX:
                    not_determined["license_plate_text"] = (
                        f"vehicle too small to read a plate (largest view {width_px}px wide; "
                        f"needs at least {MIN_VEHICLE_WIDTH_FOR_OCR_PX}px)"
                    )
                else:
                    not_determined["license_plate_text"] = "no readable plate text in the vehicle's clearest frames"
            not_determined["vehicle_make_model"] = make_cap["reason"]
        else:
            kind, fields = "OBJECT", []
        entities.append({
            "track_id": t.track_number,
            "class_name": t.class_name,
            "entity_type": kind,
            "first_seen": t.first_seen_timestamp,
            "last_seen": t.last_seen_timestamp,
            "duration": t.duration,
            "observation_count": t.observation_count,
            "attributes": {f: attrs[f] for f in fields if f in attrs},
            "not_determined": not_determined,
        })

    return {
        "evidence_id": evidence_id,
        "analysis_job_id": job.id,
        "attributes_extracted": bool(aggs),
        "entities": entities,
    }


@router.get("/{evidence_id}/analysis-jobs")
async def list_analysis_jobs(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Every analysis run of this evidence, newest first, marking the active one."""
    active = await get_active_analysis_job(db, evidence_id)
    jobs = (await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.evidence_id == evidence_id)
        .order_by(desc(AnalysisJob.id))
    )).scalars().all()
    return [
        {
            "job_id": j.id,
            "status": j.status,
            "is_active": active is not None and j.id == active.id,
            "started_at": j.started_at,
            "completed_at": j.completed_at,
            "sampling_fps": j.sampling_fps,
            "confidence_threshold": j.confidence_threshold,
            "model_name": j.model_name,
            "error_message": j.error_message,
        }
        for j in jobs
    ]

@router.get("/{evidence_id}/detections")
async def get_evidence_detections(
    evidence_id: int,
    job_id: Optional[int] = JOB_ID_QUERY,
    class_name: Optional[str] = Query(None),
    track_id: Optional[int] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves paginated object and person detections for the latest analysis job of an evidence file.
    """
    job = await _get_latest_completed_job(evidence_id, db, job_id)
    if not job:
        return []

    query = select(Detection).where(Detection.analysis_job_id == job.id)
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
    job_id: Optional[int] = JOB_ID_QUERY,
    class_name: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves tracked entity summaries for the latest analysis job of an evidence file.
    """
    job = await _get_latest_completed_job(evidence_id, db, job_id)
    if not job:
        return []

    query = select(Track).where(Track.analysis_job_id == job.id)
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
    job_id: Optional[int] = JOB_ID_QUERY,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves extracted forensic keyframe images for the latest analysis job.
    """
    job = await _get_latest_completed_job(evidence_id, db, job_id)
    if not job:
        return []

    res = await db.execute(
        select(Keyframe)
        .where(Keyframe.analysis_job_id == job.id)
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
    job_id: Optional[int] = JOB_ID_QUERY,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves motion activity intervals for the latest analysis job.
    """
    job = await _get_latest_completed_job(evidence_id, db, job_id)
    if not job:
        return []

    res = await db.execute(
        select(ActivityInterval)
        .where(ActivityInterval.analysis_job_id == job.id)
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
    job_id: Optional[int] = JOB_ID_QUERY,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves unified chronological forensic observations timeline for the latest analysis job.
    """
    job = await _get_latest_completed_job(evidence_id, db, job_id)
    if not job:
        return []

    tr_res = await db.execute(select(Track).where(Track.analysis_job_id == job.id))
    tracks = tr_res.scalars().all()

    kf_res = await db.execute(select(Keyframe).where(Keyframe.analysis_job_id == job.id))
    keyframes = kf_res.scalars().all()

    in_res = await db.execute(select(PossibleInteraction).where(PossibleInteraction.analysis_job_id == job.id))
    interactions = in_res.scalars().all()

    timeline_events = []

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

    for k in keyframes:
        timeline_events.append({
            "timestamp": k.timestamp,
            "frame_number": k.frame_number,
            "type": "KEYFRAME_SELECTED",
            "title": f"Keyframe #{k.frame_number} ({k.selection_reason.value.replace('_', ' ')})",
            "description": f"Selected at {k.timestamp:.2f}s. Image Hash: {k.sha256_hash[:8]}...",
            "image_path": k.image_path,
            "sha256_hash": k.sha256_hash
        })

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

    timeline_events.sort(key=lambda x: x["timestamp"])
    return timeline_events

@router.get("/{evidence_id}/manifest")
async def get_evidence_manifest(
    evidence_id: int,
    job_id: Optional[int] = JOB_ID_QUERY,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns the manifest sealed at analysis completion and verifies that its
    SHA-256 still matches the hash recorded on the job. Jobs that predate sealed
    manifests get a reconstruction that is explicitly flagged as unverifiable.
    """
    job = await _get_latest_completed_job(evidence_id, db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No completed analysis job found for this evidence. Run analysis first.")

    sealed_path = ManifestGenerator.manifest_path(evidence_id, job.id)
    if sealed_path.exists():
        manifest_dict, actual_hash, verified = ManifestGenerator.load_and_verify(
            sealed_path, job.manifest_hash
        )
        if not verified:
            logger.error(
                "[INTEGRITY] Manifest for evidence=%s job=%s does not match recorded hash "
                "(recorded=%s actual=%s)", evidence_id, job.id, job.manifest_hash, actual_hash,
            )
        return {
            "manifest_data": manifest_dict,
            "sha256_hash": actual_hash,
            "recorded_hash": job.manifest_hash,
            "integrity_verified": verified,
            "source": "SEALED",
            "analysis_job_id": job.id,
        }

    ev_res = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = ev_res.scalar_one_or_none()

    kf_res = await db.execute(select(Keyframe).where(Keyframe.analysis_job_id == job.id))
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

    det_res = await db.execute(select(Detection).where(Detection.analysis_job_id == job.id))
    detections = det_res.scalars().all()
    det_dicts = [
        {
            "frame_number": d.frame_number,
            "timestamp": d.timestamp,
            "class_name": d.class_name,
            "confidence": d.confidence,
            "bbox_x1": d.bbox_x1,
            "bbox_y1": d.bbox_y1,
            "bbox_x2": d.bbox_x2,
            "bbox_y2": d.bbox_y2,
            "track_id": d.track_id
        }
        for d in detections
    ]

    trk_res = await db.execute(select(Track).where(Track.analysis_job_id == job.id))
    tracks = trk_res.scalars().all()
    trk_dicts = [
        {
            "track_number": t.track_number,
            "class_name": t.class_name,
            "first_seen_timestamp": t.first_seen_timestamp,
            "last_seen_timestamp": t.last_seen_timestamp,
            "duration": t.duration,
            "observation_count": t.observation_count
        }
        for t in tracks
    ]

    stats = (await compute_analysis_statistics(db, job)).to_dict()
    stats.pop("definitions", None)

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
        keyframes=kf_dicts,
        detections=det_dicts,
        tracks=trk_dicts
    )

    # Rebuilt from current DB rows: its hash cannot match the one recorded at
    # completion, so it must never be presented as verified.
    return {
        "manifest_data": manifest_dict,
        "sha256_hash": manifest_hash,
        "recorded_hash": job.manifest_hash,
        "integrity_verified": False,
        "source": "RECONSTRUCTED",
        "analysis_job_id": job.id,
    }

@router.get("/{evidence_id}/derived/keyframes/{filename}")
async def serve_derived_keyframe(
    evidence_id: int,
    filename: str
):
    """
    Serves derived keyframe thumbnail image files securely.
    """
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(settings.DERIVED_STORAGE_DIR, "keyframes", str(evidence_id), safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Keyframe image file not found")

    return FileResponse(file_path, media_type="image/jpeg")
