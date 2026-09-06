from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, BigInteger, Float, ForeignKey, DateTime, func, Enum
from app.database.base import Base
from datetime import datetime
import enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.analysis import AnalysisJob

class EvidenceStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Evidence(Base):
    """
    SQLAlchemy model representing uploaded CCTV video evidence and digital forensic metadata.
    """
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Video technical metadata
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(50), nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Analysis & forensic state
    status: Mapped[EvidenceStatus] = mapped_column(
        Enum(EvidenceStatus), 
        default=EvidenceStatus.UPLOADED, 
        nullable=False,
        index=True
    )
    
    # Forensic Integrity Hash
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Ownership & timestamps
    uploaded_by: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("users.id", ondelete="RESTRICT"), 
        nullable=False
    )
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    uploader: Mapped["User"] = relationship(back_populates="evidence_files")
    analysis_jobs: Mapped[List["AnalysisJob"]] = relationship(back_populates="evidence", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Evidence id={self.id}, filename='{self.original_filename}', status='{self.status}', hash='{self.sha256_hash[:8]}...'>"
