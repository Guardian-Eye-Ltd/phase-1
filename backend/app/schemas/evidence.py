from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from app.models.evidence import EvidenceStatus

class EvidenceBase(BaseModel):
    original_filename: str = Field(..., description="Original client filename")
    stored_filename: str = Field(..., description="Unique internal storage filename")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME content type")
    duration: float | None = Field(None, description="Video duration in seconds")
    resolution: str | None = Field(None, description="Video resolution string, e.g. 1920x1080")
    fps: float | None = Field(None, description="Frames per second")
    status: EvidenceStatus = Field(..., description="Analysis pipeline status")
    sha256_hash: str = Field(..., description="Forensic integrity SHA-256 hash")

class EvidenceResponse(EvidenceBase):
    id: int
    file_path: str
    uploaded_by: int
    uploaded_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class EvidenceListResponse(BaseModel):
    items: list[EvidenceResponse]
    total: int
    page: int
    size: int

class EvidenceUploadResponse(BaseModel):
    evidence: EvidenceResponse
    is_duplicate: bool = Field(False, description="True if identical SHA-256 evidence was previously ingested")
    message: str
