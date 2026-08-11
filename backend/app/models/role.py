from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer
from app.database.base import Base
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular imports in type checking layer
    from app.models.user import User

class Role(Base):
    """
    SQLAlchemy model representing authorization roles (e.g. Admin, Operator, Investigator).
    """
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)

    # One-to-Many: A single Role can authorize multiple Users
    users: Mapped[List["User"]] = relationship(back_populates="role")

    def __repr__(self) -> str:
        return f"<Role id={self.id}, name='{self.role_name}'>"
