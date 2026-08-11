from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from app.database.base import Base

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    alert_type = Column(String, nullable=False, index=True) # e.g. "MOTION_DETECTED", "UNAUTHORIZED_ACCESS"
    severity = Column(String, nullable=False) # e.g. "LOW", "MEDIUM", "HIGH", "CRITICAL"
    description = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    status = Column(String, default="NEW") # e.g. "NEW", "INVESTIGATING", "RESOLVED"
