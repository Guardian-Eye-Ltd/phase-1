from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.database.session import get_db
from app.schemas.user import UserCreate, UserLogin, UserResponse, TokenResponse, RefreshTokenResponse
from app.services.auth_service import AuthService
from app.api.dependencies.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., description="Long-lived JWT refresh token")

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new platform user account"
)
async def register(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new operator/investigator/admin account with encrypted credentials.
    """
    try:
        user = await AuthService.register_user(db, user_in)
        return UserResponse.model_validate(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate credentials and issue JWT Access and Refresh tokens"
)
async def login(
    user_in: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user via JSON credentials payload.
    """
    try:
        return await AuthService.login_user(db, user_in)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )

@router.post(
    "/login/oauth",
    response_model=TokenResponse,
    summary="OAuth2 compatible login endpoint (Swagger UI support)"
)
async def login_oauth(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user via standard OAuth2 form data (for Swagger UI Authorize button).
    """
    user_in = UserLogin(username=form_data.username, password=form_data.password)
    try:
        return await AuthService.login_user(db, user_in)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )

@router.post(
    "/refresh",
    response_model=RefreshTokenResponse,
    summary="Exchange a valid refresh token for a new access token"
)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Issue a new short-lived access token using a valid refresh token.
    """
    try:
        return await AuthService.refresh_access_token(db, request.refresh_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile"
)
async def get_me(
    current_user: User = Depends(get_current_user)
):
    """
    Fetch public profile properties of the currently authenticated user session.
    """
    return UserResponse.model_validate(current_user)
