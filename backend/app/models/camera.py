from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Boolean, DateTime, Text, func
from datetime import datetime
from app.database.base import Base

class Camera(Base):
    """
    SQLAlchemy model representing a surveillance camera device and its RTSP stream configuration.
    """
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    camera_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    location: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Network stream properties
    protocol: Mapped[str] = mapped_column(String(20), default="RTSP", nullable=False)
    stream_url: Mapped[str] = mapped_column(String(500), nullable=False)
    username: Mapped[str | None] = mapped_column(String(100), nullable=True)
    password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Operational flags and health status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="OFFLINE", nullable=False)
    
    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Camera id={self.id}, code='{self.camera_code}', name='{self.name}', status='{self.status}'>"
