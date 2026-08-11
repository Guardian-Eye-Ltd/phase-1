from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator
from datetime import datetime
from app.schemas.role import RoleResponse

class UserBase(BaseModel):
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=50, 
        pattern="^[a-zA-Z0-9_-]+$", 
        description="Unique credentials identifier, containing only alphanumeric, hyphen or underscore chars"
    )
    email: EmailStr = Field(..., description="Valid operator email address")
    full_name: str = Field(
        ..., 
        min_length=2, 
        max_length=100, 
        description="Full legal name of the operator"
    )

class UserCreate(UserBase):
    """
    Schema representing signup details. Enforces complex passwords and links user to a role.
    """
    password: str = Field(
        ..., 
        min_length=8, 
        max_length=128, 
        description="Secure operator password"
    )
    role_id: int = Field(..., description="ID references the Role database model mapping")

    @field_validator("password")
    @classmethod
    def check_password_complexity(cls, v: str) -> str:
        """
        Enforce credential complexity to guarantee security standards are met.
        """
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one numerical digit.")
        if not any(char.isalpha() for char in v):
            raise ValueError("Password must contain at least one alphabetical letter.")
        return v

class UserLogin(BaseModel):
    """
    Schema input specifying user login query body.
    """
    username: str = Field(..., description="Registered credential handle")
    password: str = Field(..., description="Plaintext login password")

class UserResponse(UserBase):
    """
    Schema representation for serializing User data, securely removing private passwords or hashes.
    """
    id: int
    role_id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    role: RoleResponse | None = None

    # ORM config enables data loading directly from SQLAlchemy base classes
    model_config = ConfigDict(from_attributes=True)

class TokenResponse(BaseModel):
    """
    Schema container holding JWT Token bindings issued upon authorization success.
    """
    access_token: str = Field(..., description="Short-lived authentication token")
    refresh_token: str = Field(..., description="Long-lived session refresh token")
    token_type: str = Field("bearer", description="Token authentication specification style")
    user: UserResponse = Field(..., description="Public profile properties of the authenticated operator")

class RefreshTokenResponse(BaseModel):
    """
    Schema output returning new short-lived session access token.
    """
    access_token: str = Field(..., description="Newly generated short-lived authentication token")
    token_type: str = Field("bearer", description="Token authentication specification style")
