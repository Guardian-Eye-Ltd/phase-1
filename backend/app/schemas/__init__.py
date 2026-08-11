from app.schemas.common import BaseResponse, ErrorResponse
from app.schemas.role import RoleBase, RoleCreate, RoleResponse
from app.schemas.user import UserBase, UserCreate, UserLogin, UserResponse, TokenResponse, RefreshTokenResponse
from app.schemas.camera import CameraBase, CameraCreate, CameraUpdate, CameraResponse, CameraStatusResponse
from app.schemas.alert import AlertCreate, AlertResponse
from app.schemas.incident import IncidentCreate, IncidentResponse

__all__ = [
    "BaseResponse",
    "ErrorResponse",
    "RoleBase",
    "RoleCreate",
    "RoleResponse",
    "UserBase",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "TokenResponse",
    "RefreshTokenResponse",
    "CameraBase",
    "CameraCreate",
    "CameraUpdate",
    "CameraResponse",
    "CameraStatusResponse",
    "AlertCreate",
    "AlertResponse",
    "IncidentCreate",
    "IncidentResponse",
]
