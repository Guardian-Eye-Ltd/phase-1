import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.role import Role
from app.authentication.password import hash_password
from app.authentication.jwt import create_access_token
from app.core.permissions import PermissionChecker, SystemRole
from fastapi import HTTPException

@pytest.mark.asyncio
async def test_user_registration_success(async_client: AsyncClient):
    """Test registering a new user with valid details."""
    payload = {
        "full_name": "Test Operator",
        "username": "test_operator",
        "email": "operator@test.com",
        "password": "Password123",
        "role_id": 2
    }
    response = await async_client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "test_operator"
    assert data["email"] == "operator@test.com"
    assert "password" not in data
    assert "password_hash" not in data

@pytest.mark.asyncio
async def test_user_registration_duplicate_username(async_client: AsyncClient):
    """Test duplicate username registration prevention."""
    payload = {
        "full_name": "Duplicate User",
        "username": "test_operator",
        "email": "another@test.com",
        "password": "Password123",
        "role_id": 2
    }
    response = await async_client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"]

@pytest.mark.asyncio
async def test_user_login_success(async_client: AsyncClient):
    """Test authenticating registered credentials."""
    login_payload = {
        "username": "test_operator",
        "password": "Password123"
    }
    response = await async_client.post("/api/v1/auth/login", json=login_payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == "test_operator"

@pytest.mark.asyncio
async def test_user_login_invalid_password(async_client: AsyncClient):
    """Test authentication failure with incorrect password."""
    login_payload = {
        "username": "test_operator",
        "password": "WrongPassword123"
    }
    response = await async_client.post("/api/v1/auth/login", json=login_payload)
    assert response.status_code == 401
    assert "Invalid username or password" in response.json()["detail"]

@pytest.mark.asyncio
async def test_token_refresh_flow(async_client: AsyncClient):
    """Test obtaining new access token via refresh token."""
    # First login to get a refresh token
    login_payload = {
        "username": "test_operator",
        "password": "Password123"
    }
    login_res = await async_client.post("/api/v1/auth/login", json=login_payload)
    refresh_token = login_res.json()["refresh_token"]

    # Exchange refresh token
    refresh_payload = {"refresh_token": refresh_token}
    response = await async_client.post("/api/v1/auth/refresh", json=refresh_payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_get_me_profile_success(async_client: AsyncClient):
    """Test profile endpoint with valid access token."""
    # Login
    login_res = await async_client.post("/api/v1/auth/login", json={"username": "test_operator", "password": "Password123"})
    access_token = login_res.json()["access_token"]

    headers = {"Authorization": f"Bearer {access_token}"}
    response = await async_client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["username"] == "test_operator"

@pytest.mark.asyncio
async def test_get_me_unauthorized(async_client: AsyncClient):
    """Test profile endpoint without authorization header."""
    response = await async_client.get("/api/v1/auth/me")
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_rbac_permission_checker():
    """Test RBAC PermissionChecker logic directly."""
    admin_role = Role(id=1, role_name="Admin")
    operator_role = Role(id=2, role_name="Operator")

    admin_user = User(id=1, username="admin_usr", role=admin_role)
    operator_user = User(id=2, username="op_usr", role=operator_role)

    admin_checker = PermissionChecker([SystemRole.ADMIN])

    # Admin user should pass Admin checker
    assert admin_checker(admin_user) == admin_user

    # Operator user should be rejected by Admin checker
    with pytest.raises(HTTPException) as exc_info:
        admin_checker(operator_user)
    assert exc_info.value.status_code == 403
