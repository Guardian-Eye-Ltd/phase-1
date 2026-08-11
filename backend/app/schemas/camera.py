from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime

class CameraBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Human-readable camera designation")
    camera_code: str = Field(
        ..., 
        min_length=2, 
        max_length=50, 
        pattern="^[a-zA-Z0-9_-]+$", 
        description="Unique identifier code for camera device"
    )
    location: str = Field(..., min_length=2, max_length=100, description="Physical location or zone")
    description: str | None = Field(None, max_length=255, description="Optional description details")
    protocol: str = Field("RTSP", description="Streaming protocol (e.g., RTSP, RTSPS)")
    stream_url: str = Field(..., min_length=5, max_length=500, description="Network stream URI")
    username: str | None = Field(None, max_length=100, description="RTSP authentication username")

    @field_validator("protocol")
    @classmethod
    def validate_protocol(cls, v: str) -> str:
        upper = v.upper()
        if upper not in ("RTSP", "RTSPS", "HTTP", "HTTPS"):
            raise ValueError("Protocol must be one of: RTSP, RTSPS, HTTP, HTTPS")
        return upper

class CameraCreate(CameraBase):
    """Schema input for adding a new camera device."""
    password: str | None = Field(None, max_length=255, description="RTSP authentication password")

class CameraUpdate(BaseModel):
    """Schema input for updating camera properties."""
    name: str | None = Field(None, min_length=2, max_length=100)
    camera_code: str | None = Field(None, min_length=2, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    location: str | None = Field(None, min_length=2, max_length=100)
    description: str | None = Field(None, max_length=255)
    protocol: str | None = Field(None)
    stream_url: str | None = Field(None, min_length=5, max_length=500)
    username: str | None = Field(None, max_length=100)
    password: str | None = Field(None, max_length=255)
    is_active: bool | None = Field(None)

    @field_validator("protocol")
    @classmethod
    def validate_protocol(cls, v: str | None) -> str | None:
        if v is None:
            return None
        upper = v.upper()
        if upper not in ("RTSP", "RTSPS", "HTTP", "HTTPS"):
            raise ValueError("Protocol must be one of: RTSP, RTSPS, HTTP, HTTPS")
        return upper

class CameraResponse(CameraBase):
    """
    Public response schema for camera entities.
    EXCLUDES password and sanitizes credentials embedded in stream URLs.
    """
    id: int
    is_active: bool
    status: str
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime | None = None

    @field_validator("stream_url", mode="before")
    @classmethod
    def sanitize_stream_url(cls, v: str) -> str:
        """Sanitize embedded password in stream URL strings."""
        if not v or "@" not in v or "://" not in v:
            return v
        try:
            proto_part, rest = v.split("://", 1)
            if "@" in rest:
                auth_part, host_part = rest.split("@", 1)
                if ":" in auth_part:
                    usr, _ = auth_part.split(":", 1)
                    return f"{proto_part}://{usr}:***@{host_part}"
                return f"{proto_part}://***@{host_part}"
        except Exception:
            pass
        return v

    model_config = ConfigDict(from_attributes=True)

class CameraStatusResponse(BaseModel):
    """Detailed operational health status metrics for a camera stream."""
    id: int
    camera_code: str
    status: str
    is_active: bool
    last_seen_at: datetime | None = None
    fps: float = Field(0.0, description="Current frame rendering rate")
    frame_count: int = Field(0, description="Total processed frames since connect")
    last_error: str | None = Field(None, description="Recent error string if offline or failed")
