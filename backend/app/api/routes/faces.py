"""
Face recognition endpoints. Biometric data: Investigator role or above, every
search audited, embeddings never returned, query photos never stored.
"""
import asyncio
import hashlib
import json
import os
import re
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.permissions import RequireInvestigator
from app.database.session import get_db
from app.models.analysis import AnalysisJob, Track
from app.models.audit import AuditLog
from app.models.face import FaceObservation, FaceQuality
from app.models.user import User
from app.services.analysis_jobs import AnalysisJobNotFound, get_active_analysis_job
from app.services.face.engine import FaceEngine
from app.services.face.search_service import (
    FaceSearchError, embed_query_photo, face_search_audit_entry, search_faces,
)

router = APIRouter(prefix="/evidence", tags=["Face Recognition"])


async def _job_or_404(db: AsyncSession, evidence_id: int, job_id: Optional[int]) -> AnalysisJob:
    try:
        job = await get_active_analysis_job(db, evidence_id, job_id, strict=job_id is not None)
    except AnalysisJobNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    if job is None:
        raise HTTPException(status_code=404, detail="No completed analysis job for this evidence.")
    return job


def _crop_url(evidence_id: int, filename: str) -> str:
    return f"/api/v1/evidence/{evidence_id}/faces/crops/{filename}"


@router.post("/{evidence_id}/faces/search")
async def search_by_face(
    evidence_id: int,
    photo: UploadFile = File(..., description="Photo of the person to look for (one clear face)."),
    threshold: Optional[float] = Form(None, ge=0.30, le=0.90),
    job_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(RequireInvestigator),
):
    """
    Compare a photo against every usable face extracted from one analysis run.
    Returns POSSIBLE_MATCH candidates with timestamps and face crops, plus how
    many people in the run could not be compared at all.
    """
    job = await _job_or_404(db, evidence_id, job_id)
    image_bytes = await photo.read(settings.FACE_QUERY_MAX_BYTES + 1)
    photo_sha = hashlib.sha256(image_bytes).hexdigest()

    if job.face_stage == "COMPLETED" and not await asyncio.to_thread(FaceEngine.load):
        raise HTTPException(status_code=503, detail=f"Face model unavailable: {FaceEngine.load_error()}")

    try:
        query = (
            await asyncio.to_thread(embed_query_photo, image_bytes)
            if job.face_stage == "COMPLETED" else None
        )
    except FaceSearchError as e:
        db.add(AuditLog(
            user_id=current_user.id, action="FACE_SEARCH_REJECTED", resource_type="EVIDENCE",
            resource_id=str(evidence_id),
            metadata_json=json.dumps({"analysis_job_id": job.id, "query_photo_sha256": photo_sha,
                                      "reason": str(e)}),
        ))
        await db.commit()
        raise HTTPException(status_code=422, detail=str(e))

    result = await search_faces(db, job, query["embedding"] if query else None, threshold)
    del query  # the query embedding is not retained

    for m in result["matches"]:
        for o in m["observations"]:
            o["crop_url"] = _crop_url(evidence_id, o.pop("crop_filename"))

    db.add(face_search_audit_entry(current_user.id, result, photo_sha))
    await db.commit()
    return result


@router.get("/{evidence_id}/faces")
async def list_faces(
    evidence_id: int,
    job_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(RequireInvestigator),
):
    """Faces extracted per person track (no embeddings), with quality and crops."""
    job = await _job_or_404(db, evidence_id, job_id)
    observations = (await db.execute(
        select(FaceObservation)
        .where(FaceObservation.analysis_job_id == job.id)
        .order_by(FaceObservation.track_number, FaceObservation.timestamp)
    )).scalars().all()
    person_tracks = (await db.execute(
        select(Track).where(Track.analysis_job_id == job.id, Track.class_name == "person")
        .order_by(Track.track_number)
    )).scalars().all()

    faces_by_track = {}
    for o in observations:
        faces_by_track.setdefault(o.track_number, []).append({
            "frame_number": o.frame_number,
            "timestamp": o.timestamp,
            "quality_status": o.quality_status,
            "quality_reasons": o.quality_reasons,
            "face_size_px": [o.face_width_px, o.face_height_px],
            "det_score": o.det_score,
            "sharpness": o.sharpness,
            "crop_url": _crop_url(evidence_id, o.crop_filename),
            "crop_sha256": o.crop_sha256,
        })

    tracks = []
    for t in person_tracks:
        faces = faces_by_track.get(t.track_number, [])
        if any(f["quality_status"] == FaceQuality.USABLE for f in faces):
            status = "USABLE_FACE"
        elif faces:
            status = "LOW_QUALITY"
        else:
            status = "NO_FACE_VISIBLE"
        tracks.append({
            "track_id": t.track_number,
            "first_seen": t.first_seen_timestamp,
            "last_seen": t.last_seen_timestamp,
            "face_status": status,
            "faces": faces,
        })

    db.add(AuditLog(
        user_id=current_user.id, action="FACE_LIST_VIEWED", resource_type="EVIDENCE",
        resource_id=str(evidence_id), metadata_json=json.dumps({"analysis_job_id": job.id}),
    ))
    await db.commit()

    return {
        "evidence_id": evidence_id,
        "analysis_job_id": job.id,
        "face_stage": job.face_stage,
        "face_stage_detail": job.face_stage_detail,
        "match_threshold_default": settings.FACE_MATCH_THRESHOLD,
        "tracks": tracks,
    }


@router.get("/{evidence_id}/faces/crops/{filename}")
async def get_face_crop(
    evidence_id: int,
    filename: str,
    current_user: User = Depends(RequireInvestigator),
):
    """Authenticated face-crop download (biometric data — never served publicly)."""
    if not re.fullmatch(rf"face_ev{evidence_id}_job\d+_trk\d+_fn\d+\.jpg", filename):
        raise HTTPException(status_code=404, detail="Face crop not found.")
    path = os.path.join(settings.DERIVED_STORAGE_DIR, "faces", str(evidence_id), filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Face crop not found.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})
