import pytest
import io
import hashlib
from httpx import AsyncClient, ASGITransport
from main import app
from app.database.init_db import init_db

@pytest.fixture(autouse=True)
async def setup_db():
    """Ensure DB and default roles/users are seeded before tests run."""
    await init_db()

@pytest.mark.asyncio
async def test_auth_login_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/v1/auth/login", json={
            "username": "investigator",
            "password": "Investigator123!"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user"]["username"] == "investigator"

@pytest.mark.asyncio
async def test_auth_login_invalid():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/v1/auth/login", json={
            "username": "investigator",
            "password": "WrongPassword123!"
        })
        assert response.status_code == 401

@pytest.mark.asyncio
async def test_evidence_upload_and_sha256():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Login
        login_res = await ac.post("/api/v1/auth/login", json={
            "username": "investigator",
            "password": "Investigator123!"
        })
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Prepare mock MP4 video content
        import uuid
        video_content = f"FAKEMP4VIDEOCONTENT_{uuid.uuid4().hex}".encode('utf-8')
        expected_sha256 = hashlib.sha256(video_content).hexdigest()

        files = {"file": ("cctv_test_feed.mp4", io.BytesIO(video_content), "video/mp4")}

        # 3. Upload evidence
        upload_res = await ac.post("/api/v1/evidence/upload", headers=headers, files=files)
        assert upload_res.status_code == 201
        upload_data = upload_res.json()
        assert upload_data["is_duplicate"] is False
        evidence = upload_data["evidence"]
        assert evidence["sha256_hash"] == expected_sha256
        assert evidence["original_filename"] == "cctv_test_feed.mp4"

        evidence_id = evidence["id"]

        # 4. Duplicate upload test
        files_dup = {"file": ("cctv_test_feed_copy.mp4", io.BytesIO(video_content), "video/mp4")}
        dup_res = await ac.post("/api/v1/evidence/upload", headers=headers, files=files_dup)
        assert dup_res.status_code == 201
        assert dup_res.json()["is_duplicate"] is True

        # 5. Fetch evidence details & verify hash
        get_res = await ac.get(f"/api/v1/evidence/{evidence_id}", headers=headers)
        assert get_res.status_code == 200
        assert get_res.json()["sha256_hash"] == expected_sha256

        # 6. Fetch audit log & check UPLOAD_EVIDENCE recorded
        audit_res = await ac.get("/api/v1/audit", headers=headers)
        assert audit_res.status_code == 200
        logs = audit_res.json()["items"]
        actions = [log["action"] for log in logs]
        assert "UPLOAD_EVIDENCE" in actions

@pytest.mark.asyncio
async def test_invalid_file_extension_rejected():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        login_res = await ac.post("/api/v1/auth/login", json={
            "username": "investigator",
            "password": "Investigator123!"
        })
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        bad_files = {"file": ("malicious_script.exe", io.BytesIO(b"malicious_code"), "application/x-msdownload")}
        res = await ac.post("/api/v1/evidence/upload", headers=headers, files=bad_files)
        assert res.status_code == 400
        assert "Unsupported file format" in res.json()["detail"]
