from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, ForeignKey, Boolean, DateTime, func
from app.database.base import Base
from datetime import datetime
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.role import Role
    from app.models.evidence import Evidence
    from app.models.audit import AuditLog

class User(Base):
    """
    SQLAlchemy model representing a platform user (investigator, admin, operator).
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Foreign Key linking user to role
    role_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("roles.id", ondelete="RESTRICT"), 
        nullable=False
    )
    
    # Status flags
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    role: Mapped["Role"] = relationship(back_populates="users")
    evidence_files: Mapped[List["Evidence"]] = relationship(back_populates="uploader", cascade="all, delete-orphan")
    audit_logs: Mapped[List["AuditLog"]] = relationship(back_populates="user")

    def __repr__(self) -> str:
        return f"<User id={self.id}, username='{self.username}', role_id={self.role_id}>"
