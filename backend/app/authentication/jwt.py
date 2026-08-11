import jwt
from datetime import datetime, timedelta, timezone
from typing import Any
from app.core.config import settings

# Define custom exception classes for granular, production-ready auth errors
class JWTError(Exception):
    """Base exception class for all JWT related failures."""
    pass

class TokenExpiredError(JWTError):
    """Raised when token has exceeded its expiration timestamp."""
    pass

class TokenInvalidError(JWTError):
    """Raised when a token signature is invalid, claims are missing, or format is malformed."""
    pass

def create_access_token(user_id: int, username: str, role: str, expires_delta: timedelta | None = None) -> str:
    """
    Generate a signed, short-lived JWT Access Token.
    
    Args:
        user_id: Unique database ID of the user.
        username: Registered credentials login handle.
        role: Security context level (e.g. Admin, Operator).
        expires_delta: Optional override for token duration.
        
    Returns:
        Encoded HS256 JWT string.
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": str(username),
        "user_id": user_id,
        "role": role,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp())
    }
    
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def create_refresh_token(user_id: int, expires_delta: timedelta | None = None) -> str:
    """
    Generate a signed, long-lived JWT Refresh Token.
    
    Args:
        user_id: Unique database ID of the user.
        expires_delta: Optional override for token duration.
        
    Returns:
        Encoded HS256 JWT string.
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        # Default refresh tokens to 7 days
        expire = now + timedelta(days=7)
        
    payload = {
        "user_id": user_id,
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp())
    }
    
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and verify a raw JWT string's signature and expiration using SECRET_KEY.
    
    Args:
        token: Raw JWT string.
        
    Returns:
        The deserialized claims dictionary.
        
    Raises:
        TokenExpiredError: If current time exceeds expiration claim.
        TokenInvalidError: If signature fails or formatting is invalid.
    """
    try:
        return jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Token has expired")
    except jwt.InvalidSignatureError:
        raise TokenInvalidError("Invalid token signature")
    except jwt.InvalidTokenError:
        raise TokenInvalidError("Malformed token or invalid format")

def verify_token(token: str, expected_type: str = "access") -> dict[str, Any]:
    """
    Decode token and validate token-specific metadata types and key claims.
    
    Args:
        token: Raw JWT string.
        expected_type: Target token classification ('access' or 'refresh').
        
    Returns:
        Claims payload mapping dictionary.
        
    Raises:
        TokenInvalidError: If token structure does not match specifications.
    """
    payload = decode_token(token)
    
    # 1. Verify token classification claim
    if payload.get("type") != expected_type:
        raise TokenInvalidError(f"Invalid token type: expected '{expected_type}'")
        
    # 2. Check existence of mandatory validation claims
    if expected_type == "access":
        if not all(claim in payload for claim in ("sub", "user_id", "role")):
            raise TokenInvalidError("Missing core access token claims (sub, user_id, role)")
    elif expected_type == "refresh":
        if "user_id" not in payload:
            raise TokenInvalidError("Missing core refresh token claims (user_id)")
            
    return payload

def get_current_user_payload(token: str) -> dict[str, Any]:
    """
    Convenience wrapper to load and verify the current request's user access token payload.
    """
    return verify_token(token, expected_type="access")
