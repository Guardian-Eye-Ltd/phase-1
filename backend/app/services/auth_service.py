from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from app.models.user import User
from app.models.role import Role
from app.schemas.user import UserCreate, UserLogin, UserResponse, TokenResponse, RefreshTokenResponse
from app.authentication.password import hash_password, verify_password
from app.authentication.jwt import create_access_token, create_refresh_token, verify_token, JWTError

class AuthService:
    """
    Service layer providing authentication logic, user provisioning, credential validation,
    and JWT lifecycle management.
    """

    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
        """Fetch user by primary key ID with eagerly loaded role relationship."""
        statement = select(User).where(User.id == user_id).options(selectinload(User.role))
        result = await db.execute(statement)
        return result.scalars().first()

    @staticmethod
    async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
        """Fetch user by unique username handle."""
        statement = select(User).where(User.username == username).options(selectinload(User.role))
        result = await db.execute(statement)
        return result.scalars().first()

    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
        """Fetch user by unique email address."""
        statement = select(User).where(User.email == email).options(selectinload(User.role))
        result = await db.execute(statement)
        return result.scalars().first()

    @staticmethod
    async def register_user(db: AsyncSession, user_in: UserCreate) -> User:
        """
        Register a new platform user.
        
        Raises:
            ValueError: If username/email already exists or role_id is invalid.
        """
        # 1. Check for duplicate username
        existing_username = await AuthService.get_user_by_username(db, user_in.username)
        if existing_username:
            raise ValueError(f"Username '{user_in.username}' is already registered.")

        # 2. Check for duplicate email
        existing_email = await AuthService.get_user_by_email(db, user_in.email)
        if existing_email:
            raise ValueError(f"Email '{user_in.email}' is already registered.")

        # 3. Verify target role exists
        role_stmt = select(Role).where(Role.id == user_in.role_id)
        role_res = await db.execute(role_stmt)
        role = role_res.scalars().first()
        if not role:
            raise ValueError(f"Role ID {user_in.role_id} does not exist.")

        # 4. Hash plain password with salt
        hashed_pw = hash_password(user_in.password)

        # 5. Persist user entity
        db_user = User(
            full_name=user_in.full_name,
            username=user_in.username,
            email=user_in.email,
            password_hash=hashed_pw,
            role_id=user_in.role_id,
            is_active=True
        )
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)

        # 6. Re-fetch user with selectinload for role
        user_with_role = await AuthService.get_user_by_id(db, db_user.id)
        return user_with_role or db_user

    @staticmethod
    async def login_user(db: AsyncSession, user_in: UserLogin) -> TokenResponse:
        """
        Authenticate user credentials and issue Access & Refresh tokens.
        
        Raises:
            ValueError: If credentials are invalid or user is deactivated.
        """
        # Accept either username or email for flexible operator login
        user = await AuthService.get_user_by_username(db, user_in.username)
        if not user:
            user = await AuthService.get_user_by_email(db, user_in.username)

        if not user or not verify_password(user_in.password, user.password_hash):
            raise ValueError("Invalid username or password.")

        if not user.is_active:
            raise ValueError("User account is currently disabled.")

        role_name = user.role.role_name if user.role else "Viewer"

        # Generate JWT token pair
        access_token = create_access_token(
            user_id=user.id,
            username=user.username,
            role=role_name
        )
        refresh_token = create_refresh_token(user_id=user.id)

        user_response = UserResponse.model_validate(user)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user=user_response
        )

    @staticmethod
    async def refresh_access_token(db: AsyncSession, refresh_token: str) -> RefreshTokenResponse:
        """
        Issue a new short-lived access token using a valid refresh token.
        
        Raises:
            ValueError: If refresh token is invalid, expired, or user is inactive.
        """
        try:
            payload = verify_token(refresh_token, expected_type="refresh")
        except JWTError as e:
            raise ValueError(f"Invalid refresh token: {str(e)}")

        user_id = payload.get("user_id")
        if not user_id:
            raise ValueError("Invalid refresh token payload.")

        user = await AuthService.get_user_by_id(db, user_id)
        if not user or not user.is_active:
            raise ValueError("User associated with refresh token is invalid or inactive.")

        role_name = user.role.role_name if user.role else "Viewer"
        new_access_token = create_access_token(
            user_id=user.id,
            username=user.username,
            role=role_name
        )

        return RefreshTokenResponse(
            access_token=new_access_token,
            token_type="bearer"
        )
