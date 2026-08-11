from pydantic import BaseModel, Field, ConfigDict

class RoleBase(BaseModel):
    role_name: str = Field(
        ..., 
        min_length=2, 
        max_length=50, 
        description="Name of the security access authorization role"
    )

class RoleCreate(RoleBase):
    """
    Schema for creating a new role.
    """
    pass

class RoleResponse(RoleBase):
    """
    Schema for database role responses, matching database mapping relationships.
    """
    id: int

    # Pydantic v2 Config to read properties directly from SQLAlchemy ORM models
    model_config = ConfigDict(from_attributes=True)
