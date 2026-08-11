from fastapi import Query, WebSocketException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User
from app.services.auth_service import AuthService
from app.authentication.jwt import verify_token, TokenExpiredError, TokenInvalidError

async def get_current_user_ws(
    token: str = Query(...),
    db: AsyncSession = None
) -> User:
    """
    Authenticate WebSocket connection using token from query parameters.
    Since db is usually provided via Depends(get_db) in the router, we can't inject it here as a default easily 
    without Depends, so we rely on the route to pass db, or we make this a proper FastAPI dependency.
    """
    from app.database.session import get_db
    from fastapi import Depends
    
    # We actually need this to be a FastAPI dependency, so let's adjust it:
    pass

from fastapi import Depends
from app.database.session import SessionLocal

async def get_ws_current_user(
    token: str = Query(...)
) -> User:
    try:
        payload = verify_token(token, expected_type="access")
    except TokenExpiredError:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Token expired")
    except TokenInvalidError:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Token invalid")

    user_id = payload.get("user_id")
    if not user_id:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token payload")

    async with SessionLocal() as db:
        user = await AuthService.get_user_by_id(db, user_id)
        
    if not user or not user.is_active:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="User inactive or deleted")

    return user
