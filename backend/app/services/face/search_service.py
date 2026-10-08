"""
Search an analysis run for a person, given a photo.

The photo is embedded in memory and compared with every USABLE face stored for
the run. It is never written to disk or the database — only its SHA-256 goes
to the audit log. Results are similarity candidates (POSSIBLE_MATCH), and the
response always states how many people could not be compared at all, so
"no match" is never mistaken for "not present".
"""
import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.analysis import AnalysisJob, Track
from app.models.audit import AuditLog
from app.models.face import FaceObservation, FaceQuality
from app.services.face.crypto import decrypt_embedding
from app.services.face.engine import DetectedFace, FaceEngine, assess_quality

logger = logging.getLogger(__name__)


class FaceSearchError(Exception):
    """A problem with the query photo; message is safe to show the user."""


class FaceStage:
    NOT_RUN = "NOT_RUN"          # run predates face extraction
    COMPLETED = "COMPLETED"
    UNAVAILABLE = "UNAVAILABLE"  # model not installed / disabled
    FAILED = "FAILED"


def embed_query_photo(
    image_bytes: bytes,
    detect_fn: Callable[[np.ndarray], List[DetectedFace]] = FaceEngine.detect,
) -> Dict[str, Any]:
    if not image_bytes:
        raise FaceSearchError("The uploaded photo is empty.")
    if len(image_bytes) > settings.FACE_QUERY_MAX_BYTES:
        raise FaceSearchError("The uploaded photo is too large.")
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FaceSearchError("The uploaded file is not a readable image.")

    faces = detect_fn(img)
    if not faces:
        raise FaceSearchError("No face was found in the uploaded photo.")

    assessed = []
    for f in faces:
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        q = assess_quality(crop, x2 - x1, y2 - y1, f.det_score)
        assessed.append((f, q))

    usable = [(f, q) for f, q in assessed if q.status == FaceQuality.USABLE]
    if not usable:
        raise FaceSearchError(
            "The face in the photo is not clear enough to search with: "
            + "; ".join(assessed[0][1].reasons)
        )
    if len(usable) > 1:
        # Silently picking one could search for the wrong person.
        raise FaceSearchError(
            f"The photo contains {len(usable)} clear faces. Crop it to the one person you are looking for."
        )
    face, _ = usable[0]
    return {"embedding": face.embedding, "photo_sha256": hashlib.sha256(image_bytes).hexdigest()}


async def search_faces(
    db: AsyncSession,
    job: AnalysisJob,
    query_embedding: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    threshold = settings.FACE_MATCH_THRESHOLD if threshold is None else threshold
    base = {
        "evidence_id": job.evidence_id,
        "analysis_job_id": job.id,
        "threshold": threshold,
        "face_stage": job.face_stage,
        "matches": [],
    }

    if job.face_stage != FaceStage.COMPLETED:
        reason = {
            FaceStage.NOT_RUN: "Faces were not extracted for this analysis run (it predates face recognition). Re-run analysis.",
            FaceStage.UNAVAILABLE: f"Face recognition was unavailable during this run: {job.face_stage_detail or ''}".strip(),
            FaceStage.FAILED: f"Face extraction failed during this run: {job.face_stage_detail or ''}".strip(),
        }.get(job.face_stage, "Face extraction status unknown.")
        return {**base, "status": "FACE_EXTRACTION_NOT_AVAILABLE", "message": reason, "coverage": None}

    person_tracks = set((await db.execute(
        select(Track.track_number).where(Track.analysis_job_id == job.id, Track.class_name == "person")
    )).scalars().all())
    observations = (await db.execute(
        select(FaceObservation).where(FaceObservation.analysis_job_id == job.id)
    )).scalars().all()

    usable_tracks = {o.track_number for o in observations if o.quality_status == FaceQuality.USABLE}
    low_only = {o.track_number for o in observations} - usable_tracks
    coverage = {
        "person_tracks": len(person_tracks),
        "tracks_with_usable_face": len(usable_tracks),
        "tracks_with_low_quality_face_only": len(low_only),
        "tracks_with_no_face_visible": len(person_tracks - usable_tracks - low_only),
    }

    q = np.asarray(query_embedding, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)

    per_track: Dict[int, List[Dict[str, Any]]] = {}
    undecryptable = 0
    for o in observations:
        if o.quality_status != FaceQuality.USABLE or o.embedding_encrypted is None:
            continue
        emb = decrypt_embedding(o.embedding_encrypted)
        if emb is None:
            undecryptable += 1
            continue
        sim = float(np.dot(q, emb / (np.linalg.norm(emb) or 1.0)))
        per_track.setdefault(o.track_number, []).append({
            "frame_number": o.frame_number,
            "timestamp": o.timestamp,
            "similarity": round(sim, 4),
            "crop_filename": o.crop_filename,
            "face_width_px": o.face_width_px,
        })

    matches = []
    for track_number, obs in per_track.items():
        obs.sort(key=lambda x: x["similarity"], reverse=True)
        best = obs[0]["similarity"]
        if best < threshold:
            continue
        times = [x["timestamp"] for x in obs]
        matches.append({
            "track_id": track_number,
            "status": "POSSIBLE_MATCH",
            "best_similarity": best,
            "margin_over_threshold": round(best - threshold, 4),
            "supporting_observations": sum(1 for x in obs if x["similarity"] >= threshold),
            "compared_observations": len(obs),
            "first_seen": min(times),
            "last_seen": max(times),
            "observations": obs,
        })
    matches.sort(key=lambda m: m["best_similarity"], reverse=True)

    if undecryptable:
        logger.error(f"[FACE] job={job.id}: {undecryptable} embeddings could not be decrypted (key changed?)")

    if matches:
        status = "POSSIBLE_MATCH_FOUND"
        message = (
            f"{len(matches)} person track(s) resemble the photo above the {threshold:.2f} similarity "
            "threshold. These are candidates for human review, not identifications."
        )
    elif not usable_tracks:
        status = "NO_USABLE_FACES"
        message = (
            "No person in this run had a face clear enough to compare, so presence can be "
            "neither confirmed nor ruled out."
        )
    else:
        unchecked = coverage["person_tracks"] - coverage["tracks_with_usable_face"]
        status = "NO_MATCH_AMONG_USABLE_FACES"
        message = (
            f"None of the {len(usable_tracks)} comparable face(s) resemble the photo."
            + (f" {unchecked} person track(s) had no usable face, so the person may still be present." if unchecked else "")
        )

    return {
        **base,
        "status": status,
        "message": message,
        "coverage": coverage,
        "undecryptable_embeddings": undecryptable,
        "matches": matches,
    }


def face_search_audit_entry(user_id: int, result: Dict[str, Any], photo_sha256: str) -> AuditLog:
    return AuditLog(
        user_id=user_id,
        action="FACE_SEARCH",
        resource_type="EVIDENCE",
        resource_id=str(result["evidence_id"]),
        metadata_json=json.dumps({
            "analysis_job_id": result["analysis_job_id"],
            "query_photo_sha256": photo_sha256,   # the photo itself is never stored
            "threshold": result["threshold"],
            "status": result["status"],
            "matched_track_ids": [m["track_id"] for m in result["matches"]],
            "best_similarities": [m["best_similarity"] for m in result["matches"]],
        }),
    )
