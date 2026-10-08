"""
Face observations extracted from person tracks.

Biometric data. Embeddings are stored encrypted and are never returned by any
API; a match is only ever reported as a similarity candidate.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, Integer, JSON, LargeBinary, String, UniqueConstraint, func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class FaceQuality:
    USABLE = "USABLE"            # embedded and searchable
    LOW_QUALITY = "LOW_QUALITY"  # face seen but too small / uncertain / blurry to compare


class FaceObservation(Base):
    __tablename__ = "face_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evidence_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evidence.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    track_number: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)

    # Face box in normalised source-frame coordinates.
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    face_width_px: Mapped[int] = mapped_column(Integer, nullable=False)
    face_height_px: Mapped[int] = mapped_column(Integer, nullable=False)

    det_score: Mapped[float] = mapped_column(Float, nullable=False)
    sharpness: Mapped[float] = mapped_column(Float, nullable=False)
    quality_status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    quality_reasons: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Fernet ciphertext of a float32 L2-normalised embedding. NULL for
    # LOW_QUALITY faces: they are never compared.
    embedding_encrypted: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False, default=512)

    crop_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    crop_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("analysis_job_id", "track_number", "frame_number", name="uq_face_job_track_frame"),
        Index("ix_face_job_quality", "analysis_job_id", "quality_status"),
    )
