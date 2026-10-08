"""
Canonical evidence-observation model.

Separates the four evidence tiers the investigation system must never conflate:

    RAW OBSERVATION      -> VisualAttributeObservation (status=OBSERVED)
    MODEL INTERPRETATION -> TrackAttributeAggregate   (temporally aggregated)
    EVENT CANDIDATE      -> status=CANDIDATE
    VERIFIED EVENT       -> status=SUPPORTED / VERIFIED

Every row carries full provenance: evidence_id, analysis_job_id, timestamp,
frame_number, track_id, source model + version, and confidence. An observation
without provenance is not admissible evidence.
"""
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, JSON, func, Enum, Index
from app.database.base import Base
from datetime import datetime
import enum
from typing import Optional


class ObservationStatus(str, enum.Enum):
    """
    Evidence status ladder (Phase 34). Deliberately has no "CONFIRMED" member —
    the system never declares criminal responsibility.

    NOTE: distinct from app.models.evidence.EvidenceStatus, which tracks the
    upload/processing lifecycle of a video file, not the strength of a claim.
    """
    OBSERVED = "OBSERVED"            # Raw detector/model output, no correlation yet
    CANDIDATE = "CANDIDATE"          # Event engine proposed it; unverified
    SUPPORTED = "SUPPORTED"          # Correlated with independent evidence
    VERIFIED = "VERIFIED"            # Verification agent confirmed against source records
    WITHHELD = "WITHHELD"            # Insufficient evidence to assert
    CONTRADICTED = "CONTRADICTED"    # Contradicted by other evidence


class ObservationSource(str, enum.Enum):
    """Which model/method produced the value. Never present a heuristic as truth."""
    YOLO_DETECTOR = "YOLO_DETECTOR"
    CLIP_ZERO_SHOT = "CLIP_ZERO_SHOT"
    POSE_CROP_COLOR_MODEL = "POSE_CROP_COLOR_MODEL"
    COLOR_HEURISTIC = "COLOR_HEURISTIC"
    ALPR_OCR = "ALPR_OCR"
    VLM = "VLM"
    COMBINED = "COMBINED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class EntityType(str, enum.Enum):
    PERSON = "PERSON"
    VEHICLE = "VEHICLE"
    OBJECT = "OBJECT"
    UNKNOWN = "UNKNOWN"


class VisualAttributeObservation(Base):
    """
    One row per (frame, track, attribute). This is a RAW OBSERVATION — a single
    model's opinion at a single instant. Never query this directly to answer an
    investigator question; aggregate it first (see TrackAttributeAggregate).
    """
    __tablename__ = "visual_attribute_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- Provenance (Phase 1) ---
    evidence_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    track_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    keyframe_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # --- Claim ---
    entity_type: Mapped[EntityType] = mapped_column(
        Enum(EntityType), default=EntityType.UNKNOWN, nullable=False, index=True
    )
    attribute: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    value: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # --- Model identity ---
    source: Mapped[ObservationSource] = mapped_column(
        Enum(ObservationSource), default=ObservationSource.CLIP_ZERO_SHOT, nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(100), default="unknown", nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), default="1.0", nullable=False)

    status: Mapped[ObservationStatus] = mapped_column(
        Enum(ObservationStatus), default=ObservationStatus.OBSERVED, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_vao_job_attr_value", "analysis_job_id", "attribute", "value"),
        Index("ix_vao_job_track", "analysis_job_id", "track_number"),
    )


class TrackAttributeAggregate(Base):
    """
    One row per (track, attribute): the temporally aggregated consensus across
    every VisualAttributeObservation for that track (Phase 10).

    This is the table structured attribute search should query — a single frame
    is never authoritative.
    """
    __tablename__ = "track_attribute_aggregates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    evidence_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    track_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    entity_type: Mapped[EntityType] = mapped_column(
        Enum(EntityType), default=EntityType.UNKNOWN, nullable=False, index=True
    )

    attribute: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    value: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Aggregated confidence, computed by AttributeAggregator with a documented
    # deterministic formula — never LLM-assigned.
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    observation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    supporting_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    dissenting_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    first_observed_at: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_observed_at: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    source: Mapped[ObservationSource] = mapped_column(
        Enum(ObservationSource), default=ObservationSource.COMBINED, nullable=False
    )
    status: Mapped[ObservationStatus] = mapped_column(
        Enum(ObservationStatus), default=ObservationStatus.OBSERVED, nullable=False
    )

    # Phase 20: per-axis confidence decomposition, not a single opaque number.
    confidence_breakdown: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_taa_job_attr_value", "analysis_job_id", "attribute", "value"),
        Index("ix_taa_job_entity", "analysis_job_id", "entity_type"),
    )
