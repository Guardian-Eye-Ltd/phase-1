"""
Canonical analysis statistics.

There is exactly one definition of each counter, documented in DEFINITIONS.
Counters that describe stored data are recomputed from the database for the
specific analysis job, so they can never drift from what actually exists.
Counters that only exist at run time (how many boxes the model returned before
filtering) come from the AnalysisJob row; for jobs that ran before they were
recorded they are reported as None — never as a misleading 0.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import (
    ActivityInterval, AnalysisJob, Detection, FrameObservation, Keyframe,
    PossibleInteraction, Track,
)
from app.models.face import FaceObservation
from app.models.observation import EntityType, VisualAttributeObservation
from app.models.semantic import ForensicDocument, VLMObservation

# Bump when the meaning of a run-time counter changes.
STATS_SCHEMA_VERSION = 2

DEFINITIONS: Dict[str, str] = {
    "source_frames": "Frames in the source video according to its container metadata.",
    "sampled_frames": "Frames selected by the frame sampler at the job's sampling_fps.",
    "processed_frames": "Sampled frames the object detector actually ran on.",
    "frames_with_detections": "Processed frames with at least one stored detection.",
    "raw_model_detections": "Boxes returned by the detector before GuardianEye's confidence filter.",
    "confidence_filtered_detections": "Boxes remaining after the job's confidence threshold.",
    "class_filtered_detections": "Boxes remaining after restricting to supported classes.",
    "stored_detections": "Detection rows persisted for this job.",
    "unique_tracks": "Distinct tracked entities (track_number) in this job.",
    "keyframes": "Keyframe rows persisted for this job.",
    "activity_intervals": "Motion activity intervals persisted for this job.",
    "events": "Event-engine candidates produced during this job's run.",
    "interactions": "Possible spatial interactions persisted for this job.",
    "visual_attributes": "Per-frame visual attribute observations for person tracks.",
    "vehicle_attributes": "Per-frame visual attribute observations for vehicle tracks, excluding plates.",
    "license_plate_observations": "Per-frame licence-plate OCR reads.",
    "vlm_observations": "Keyframe descriptions produced by a real vision-language model.",
    "semantic_documents": "Searchable forensic documents indexed for this job.",
    "face_observations": "Faces extracted from person tracks (usable + low quality). None = face extraction did not run for this job, not 'zero faces'.",
    "face_stage": "Outcome of face extraction for this job: NOT_RUN, COMPLETED, UNAVAILABLE or FAILED.",
}


@dataclass
class AnalysisStatistics:
    analysis_job_id: int
    evidence_id: int
    status: str

    source_frames: Optional[int] = None
    sampled_frames: int = 0
    processed_frames: Optional[int] = None
    frames_with_detections: int = 0
    raw_model_detections: Optional[int] = None
    confidence_filtered_detections: Optional[int] = None
    class_filtered_detections: Optional[int] = None
    stored_detections: int = 0
    unique_tracks: int = 0
    keyframes: int = 0
    activity_intervals: int = 0
    events: Optional[int] = None
    interactions: int = 0
    visual_attributes: int = 0
    vehicle_attributes: int = 0
    license_plate_observations: int = 0
    vlm_observations: int = 0
    semantic_documents: int = 0
    face_observations: Optional[int] = None
    face_stage: str = "NOT_RUN"

    # False for jobs that ran before run-time counters were recorded; the
    # run-time-only fields above are None for those jobs.
    pipeline_counters_recorded: bool = False
    definitions: Dict[str, str] = field(default_factory=lambda: dict(DEFINITIONS))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def _count(db: AsyncSession, stmt) -> int:
    return (await db.execute(stmt)).scalar() or 0


async def compute_analysis_statistics(db: AsyncSession, job: AnalysisJob) -> AnalysisStatistics:
    jid = job.id

    def count_rows(model):
        return select(func.count()).select_from(model).where(model.analysis_job_id == jid)

    vao = VisualAttributeObservation
    stats = AnalysisStatistics(
        analysis_job_id=jid,
        evidence_id=job.evidence_id,
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        sampled_frames=await _count(db, count_rows(FrameObservation)),
        frames_with_detections=await _count(
            db,
            select(func.count(distinct(Detection.frame_number)))
            .where(Detection.analysis_job_id == jid),
        ),
        stored_detections=await _count(db, count_rows(Detection)),
        unique_tracks=await _count(
            db,
            select(func.count(distinct(Track.track_number))).where(Track.analysis_job_id == jid),
        ),
        keyframes=await _count(db, count_rows(Keyframe)),
        activity_intervals=await _count(db, count_rows(ActivityInterval)),
        interactions=await _count(db, count_rows(PossibleInteraction)),
        visual_attributes=await _count(
            db, count_rows(vao).where(vao.entity_type == EntityType.PERSON)
        ),
        vehicle_attributes=await _count(
            db,
            count_rows(vao).where(
                vao.entity_type == EntityType.VEHICLE,
                vao.attribute != "license_plate_text",
            ),
        ),
        license_plate_observations=await _count(
            db, count_rows(vao).where(vao.attribute == "license_plate_text")
        ),
        # Legacy rows written by the old template fallback are not VLM output.
        vlm_observations=await _count(
            db, count_rows(VLMObservation).where(VLMObservation.model_name != "VLM_Heuristic_Fallback")
        ),
        semantic_documents=await _count(db, count_rows(ForensicDocument)),
    )

    # Earlier runs either didn't record these counters or recorded them with
    # different meanings (e.g. raw_detections used to be the post-filter count).
    recorded = (job.stats_schema_version or 0) >= STATS_SCHEMA_VERSION
    stats.pipeline_counters_recorded = recorded
    if job.face_stage == "COMPLETED":
        stats.face_observations = await _count(db, count_rows(FaceObservation))
    stats.face_stage = job.face_stage

    if recorded:
        stats.source_frames = job.source_frames
        stats.processed_frames = job.processed_frames
        stats.raw_model_detections = job.raw_detections
        stats.confidence_filtered_detections = job.confidence_filtered_detections
        stats.class_filtered_detections = job.class_filtered_detections
        stats.events = job.event_candidates

    return stats
