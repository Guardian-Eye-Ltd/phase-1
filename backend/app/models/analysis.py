from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, BigInteger, Float, ForeignKey, DateTime, Text, Boolean, JSON, func, Enum
from app.database.base import Base
from datetime import datetime
import enum
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from app.models.evidence import Evidence

class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class JobStage(str, enum.Enum):
    VALIDATING = "VALIDATING"
    EXTRACTING_METADATA = "EXTRACTING_METADATA"
    SAMPLING_FRAMES = "SAMPLING_FRAMES"
    MOTION_ANALYSIS = "MOTION_ANALYSIS"
    OBJECT_DETECTION = "OBJECT_DETECTION"
    TRACKING = "TRACKING"
    KEYFRAME_EXTRACTION = "KEYFRAME_EXTRACTION"
    FINALIZING = "FINALIZING"

class ActivityLevel(str, enum.Enum):
    LOW = "LOW ACTIVITY"
    HIGH = "HIGH ACTIVITY"

class KeyframeReason(str, enum.Enum):
    PERSON_APPEARANCE = "PERSON_APPEARANCE"
    OBJECT_DETECTION = "OBJECT_DETECTION"
    MOTION_CHANGE = "MOTION_CHANGE"
    ENTRY_EXIT = "ENTRY_EXIT"

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evidence_id: Mapped[int] = mapped_column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True)
    
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.QUEUED, nullable=False, index=True)
    current_stage: Mapped[JobStage] = mapped_column(Enum(JobStage), default=JobStage.VALIDATING, nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Configuration parameters
    sampling_fps: Mapped[float] = mapped_column(Float, default=2.0, nullable=False)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.4, nullable=False)
    tracking_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Execution metrics & metadata
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Model provenance metadata
    model_name: Mapped[str] = mapped_column(String(100), default="YOLOv8n", nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), default="8.2.0", nullable=False)
    tracker_algorithm: Mapped[str] = mapped_column(String(50), default="ByteTrack", nullable=False)
    manifest_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    evidence: Mapped["Evidence"] = relationship(back_populates="analysis_jobs")
    activity_intervals: Mapped[List["ActivityInterval"]] = relationship(back_populates="analysis_job", cascade="all, delete-orphan")
    frame_observations: Mapped[List["FrameObservation"]] = relationship(back_populates="analysis_job", cascade="all, delete-orphan")
    detections: Mapped[List["Detection"]] = relationship(back_populates="analysis_job", cascade="all, delete-orphan")
    tracks: Mapped[List["Track"]] = relationship(back_populates="analysis_job", cascade="all, delete-orphan")
    keyframes: Mapped[List["Keyframe"]] = relationship(back_populates="analysis_job", cascade="all, delete-orphan")
    interactions: Mapped[List["PossibleInteraction"]] = relationship(back_populates="analysis_job", cascade="all, delete-orphan")

class ActivityInterval(Base):
    __tablename__ = "activity_intervals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_id: Mapped[int] = mapped_column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True)
    
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    activity_level: Mapped[ActivityLevel] = mapped_column(Enum(ActivityLevel), default=ActivityLevel.LOW, nullable=False)
    motion_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    analysis_job: Mapped["AnalysisJob"] = relationship(back_populates="activity_intervals")

class FrameObservation(Base):
    __tablename__ = "frame_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_id: Mapped[int] = mapped_column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True)
    
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    frame_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    sha256_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    analysis_job: Mapped["AnalysisJob"] = relationship(back_populates="frame_observations")

class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_id: Mapped[int] = mapped_column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True)
    
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    class_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Normalized bounding box [0.0 - 1.0] or pixel values
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)

    analysis_job: Mapped["AnalysisJob"] = relationship(back_populates="detections")

class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_id: Mapped[int] = mapped_column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True)
    
    track_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    class_name: Mapped[str] = mapped_column(String(100), default="person", nullable=False)
    
    first_seen_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    last_seen_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    first_seen_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    last_seen_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    keyframe_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    analysis_job: Mapped["AnalysisJob"] = relationship(back_populates="tracks")

class Keyframe(Base):
    __tablename__ = "keyframes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_id: Mapped[int] = mapped_column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True)
    
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    selection_reason: Mapped[KeyframeReason] = mapped_column(Enum(KeyframeReason), default=KeyframeReason.PERSON_APPEARANCE, nullable=False)
    
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    
    track_ids: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    detection_ids: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    analysis_job: Mapped["AnalysisJob"] = relationship(back_populates="keyframes")

class PossibleInteraction(Base):
    __tablename__ = "possible_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_id: Mapped[int] = mapped_column(Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True)
    
    entity_a_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entity_b_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    min_distance_or_overlap: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.8, nullable=False)
    label: Mapped[str] = mapped_column(String(100), default="POSSIBLE_INTERACTION", nullable=False)

    analysis_job: Mapped["AnalysisJob"] = relationship(back_populates="interactions")
