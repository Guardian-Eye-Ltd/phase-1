"""
System reset tests.

SAFETY: every filesystem and vector-store hook in system_reset_service is
monkeypatched to temp directories / fakes, and the destructive test runs on its
own isolated in-memory database. These tests must never be able to touch the
real storage/ folders, ChromaDB, or the shared test database.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.system_reset_service as reset_mod
from app.authentication.jwt import create_access_token
from app.authentication.password import hash_password
from app.database.init_db import create_tables
from app.models.analysis import AnalysisJob, Detection, JobStage, JobStatus, Keyframe, Track
from app.models.audit import AuditLog
from app.models.camera import Camera
from app.models.evidence import Evidence, EvidenceStatus
from app.models.observation import (
    EntityType, ObservationSource, ObservationStatus, TrackAttributeAggregate,
    VisualAttributeObservation,
)
from app.models.role import Role
from app.models.semantic import DocumentType, ForensicDocument
from app.models.user import User
from app.services.system_reset_service import (
    RESET_CONFIRMATION_PHRASE, ResetBlockedError, SystemResetService,
)


@pytest.fixture
def sandboxed_storage(tmp_path, monkeypatch):
    """Point every destructive hook at a temp sandbox."""
    evidence_dir = tmp_path / "storage" / "evidence"
    derived_dir = tmp_path / "storage" / "derived"
    (derived_dir / "13" / "keyframes").mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "abc_video.mp4").write_bytes(b"x" * 1024)
    (derived_dir / "13" / "keyframes" / "kf1.jpg").write_bytes(b"y" * 256)
    (derived_dir / "manifest.json").write_text("{}")

    fake_collections = {"ev13_job42", "ev12_job40"}
    wiped = {"called": False}

    def fake_wipe():
        wiped["called"] = True
        n = len(fake_collections)
        fake_collections.clear()
        return {"collections_deleted": n, "errors": []}

    monkeypatch.setattr(reset_mod, "_storage_dirs", lambda: [evidence_dir, derived_dir])
    monkeypatch.setattr(reset_mod, "_list_vector_collections", lambda: sorted(fake_collections))
    monkeypatch.setattr(reset_mod, "_wipe_vector_store", fake_wipe)
    monkeypatch.setattr(reset_mod, "_running_job_ids", lambda: [])
    return {"evidence_dir": evidence_dir, "derived_dir": derived_dir, "wiped": wiped}


@pytest_asyncio.fixture
async def isolated_db():
    """A private in-memory database so the reset cannot wipe other tests' data."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    await create_tables(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as s:
        s.add_all([Role(id=1, role_name="Admin"), Role(id=3, role_name="Investigator")])
        await s.commit()
    async with maker() as session:
        yield session
    await engine.dispose()


async def _seed_evidence_graph(db: AsyncSession) -> dict:
    admin = User(full_name="Admin", username="admin_r", email="admin_r@t.local",
                 password_hash=hash_password("Admin123!"), role_id=1, is_active=True)
    db.add(admin)
    await db.commit()
    await db.refresh(admin)

    ev = Evidence(original_filename="v.mp4", stored_filename="abc_video.mp4",
                  file_path="/tmp/abc_video.mp4", file_size=1024, mime_type="video/mp4",
                  sha256_hash="a" * 64, status=EvidenceStatus.COMPLETED, uploaded_by=admin.id)
    db.add(ev)
    db.add(Camera(name="Gate", camera_code="CAM-R1", location="Front",
                  stream_url="rtsp://x", is_active=True, status="ONLINE"))
    db.add(AuditLog(user_id=admin.id, action="UPLOAD_EVIDENCE", resource_type="evidence",
                    resource_id="1", metadata_json=None))
    await db.commit()
    await db.refresh(ev)

    job = AnalysisJob(evidence_id=ev.id, status=JobStatus.COMPLETED,
                      current_stage=JobStage.FINALIZING, progress=100.0,
                      completed_at=datetime(2026, 5, 1))
    db.add(job)
    await db.commit()
    await db.refresh(job)

    db.add_all([
        Track(analysis_job_id=job.id, evidence_id=ev.id, track_number=1, class_name="car",
              first_seen_timestamp=1.0, last_seen_timestamp=5.0, first_seen_frame=1,
              last_seen_frame=50, duration=4.0, observation_count=5, keyframe_count=1),
        Detection(analysis_job_id=job.id, evidence_id=ev.id, frame_number=10, timestamp=1.0,
                  class_name="car", confidence=0.9, bbox_x1=0.1, bbox_y1=0.1,
                  bbox_x2=0.2, bbox_y2=0.2, track_id=1),
        Keyframe(analysis_job_id=job.id, evidence_id=ev.id, frame_number=10, timestamp=1.0,
                 image_path="kf1.jpg", sha256_hash="b" * 64),
        VisualAttributeObservation(
            evidence_id=ev.id, analysis_job_id=job.id, track_number=1, frame_number=10,
            timestamp=1.0, entity_type=EntityType.VEHICLE, attribute="vehicle_color",
            value="red", confidence=0.8, source=ObservationSource.CLIP_ZERO_SHOT,
            model_name="clip", model_version="1.0", status=ObservationStatus.OBSERVED),
        TrackAttributeAggregate(
            evidence_id=ev.id, analysis_job_id=job.id, track_number=1,
            entity_type=EntityType.VEHICLE, attribute="vehicle_color", value="red",
            confidence=0.7, observation_count=5, supporting_count=5, dissenting_count=0,
            first_observed_at=1.0, last_observed_at=5.0, source=ObservationSource.COMBINED,
            status=ObservationStatus.OBSERVED, confidence_breakdown={}),
        ForensicDocument(evidence_id=ev.id, analysis_job_id=job.id, track_id=1,
                         document_type=DocumentType.TRACK, title="t", content="c",
                         metadata_json={}),
    ])
    await db.commit()
    return {"admin_id": admin.id}


async def _count(db, model) -> int:
    return (await db.execute(select(func.count()).select_from(model))).scalar()


# ======================================================================
# Service-level: the actual wipe (isolated DB + sandboxed storage)
# ======================================================================

@pytest.mark.asyncio
async def test_reset_wipes_all_evidence_derived_data(isolated_db, sandboxed_storage):
    db = isolated_db
    ids = await _seed_evidence_graph(db)

    result = await SystemResetService.reset(db, user_id=ids["admin_id"])

    assert result["status"] == "COMPLETED"
    for model in (Evidence, AnalysisJob, Track, Detection, Keyframe,
                  VisualAttributeObservation, TrackAttributeAggregate, ForensicDocument):
        assert await _count(db, model) == 0, f"{model.__tablename__} not wiped"


@pytest.mark.asyncio
async def test_reset_preserves_users_roles_cameras(isolated_db, sandboxed_storage):
    db = isolated_db
    ids = await _seed_evidence_graph(db)
    await SystemResetService.reset(db, user_id=ids["admin_id"])

    assert await _count(db, User) == 1
    assert await _count(db, Role) == 2
    assert await _count(db, Camera) == 1


@pytest.mark.asyncio
async def test_reset_preserves_audit_log_and_records_itself(isolated_db, sandboxed_storage):
    """Chain of custody: prior audit entries survive and the reset is logged."""
    db = isolated_db
    ids = await _seed_evidence_graph(db)
    await SystemResetService.reset(db, user_id=ids["admin_id"])

    actions = (await db.execute(select(AuditLog.action))).scalars().all()
    assert "UPLOAD_EVIDENCE" in actions
    assert "SYSTEM_RESET" in actions

    entry = (await db.execute(
        select(AuditLog).where(AuditLog.action == "SYSTEM_RESET")
    )).scalars().first()
    assert entry.user_id == ids["admin_id"]
    meta = json.loads(entry.metadata_json)
    assert meta["deleted_rows"]["evidence"] == 1
    assert meta["deleted_rows"]["detections"] == 1


@pytest.mark.asyncio
async def test_reset_empties_storage_but_keeps_directories(isolated_db, sandboxed_storage):
    db = isolated_db
    ids = await _seed_evidence_graph(db)
    result = await SystemResetService.reset(db, user_id=ids["admin_id"])

    ev_dir, der_dir = sandboxed_storage["evidence_dir"], sandboxed_storage["derived_dir"]
    assert ev_dir.exists() and der_dir.exists()
    assert list(ev_dir.iterdir()) == []
    assert list(der_dir.iterdir()) == []
    assert result["files_deleted"] == 3


@pytest.mark.asyncio
async def test_reset_wipes_vector_store(isolated_db, sandboxed_storage):
    db = isolated_db
    ids = await _seed_evidence_graph(db)
    result = await SystemResetService.reset(db, user_id=ids["admin_id"])
    assert sandboxed_storage["wiped"]["called"]
    assert result["vector_collections_deleted"] == 2


@pytest.mark.asyncio
async def test_reset_blocked_while_analysis_running(isolated_db, sandboxed_storage, monkeypatch):
    db = isolated_db
    ids = await _seed_evidence_graph(db)
    monkeypatch.setattr(reset_mod, "_running_job_ids", lambda: [42])

    with pytest.raises(ResetBlockedError):
        await SystemResetService.reset(db, user_id=ids["admin_id"])

    # Nothing may be touched when blocked.
    assert await _count(db, Evidence) == 1
    assert list(sandboxed_storage["evidence_dir"].iterdir()) != []
    assert not sandboxed_storage["wiped"]["called"]


@pytest.mark.asyncio
async def test_preview_reports_counts_without_deleting(isolated_db, sandboxed_storage):
    db = isolated_db
    await _seed_evidence_graph(db)
    p = await SystemResetService.preview(db)

    assert p["tables"]["evidence"] == 1
    assert p["tables"]["detections"] == 1
    assert p["vector_collections"] == 2
    assert p["can_reset"] is True
    assert "audit_logs" in p["preserved"]
    assert sum(s["files"] for s in p["storage"]) == 3
    assert await _count(db, Evidence) == 1  # preview is read-only


@pytest.mark.parametrize("bad", ["/", "C:\\"])
def test_unsafe_paths_are_refused(bad):
    """A misconfigured STORAGE_DIR must never rmtree a filesystem root."""
    assert not reset_mod._is_safe_to_wipe(Path(bad))


def test_working_directory_is_refused():
    assert not reset_mod._is_safe_to_wipe(Path.cwd())


# ======================================================================
# Route-level guards (no successful wipe is executed through the shared DB)
# ======================================================================

async def _make_user(db, role_id: int, role_name: str):
    tag = uuid.uuid4().hex[:8]
    u = User(full_name=role_name, username=f"{role_name.lower()}_{tag}",
             email=f"{tag}@t.local", password_hash=hash_password("Pw123456!"),
             role_id=role_id, is_active=True)
    db.add(u)
    await db.commit()
    await db.refresh(u)
    return {"Authorization": f"Bearer {create_access_token(u.id, u.username, role_name)}"}


@pytest.mark.asyncio
async def test_reset_requires_authentication(async_client):
    r = await async_client.post("/api/v1/admin/reset",
                                json={"confirmation": RESET_CONFIRMATION_PHRASE})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_reset_forbidden_for_non_admin(async_client, db_session, sandboxed_storage):
    headers = await _make_user(db_session, 3, "Investigator")
    r = await async_client.post("/api/v1/admin/reset", headers=headers,
                                json={"confirmation": RESET_CONFIRMATION_PHRASE})
    assert r.status_code == 403
    r = await async_client.get("/api/v1/admin/reset/preview", headers=headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_reset_rejects_wrong_confirmation_phrase(async_client, db_session, sandboxed_storage):
    headers = await _make_user(db_session, 1, "Admin")
    r = await async_client.post("/api/v1/admin/reset", headers=headers,
                                json={"confirmation": "delete all evidence"})
    assert r.status_code == 400
    assert not sandboxed_storage["wiped"]["called"]


@pytest.mark.asyncio
async def test_reset_returns_409_while_job_running(async_client, db_session,
                                                    sandboxed_storage, monkeypatch):
    headers = await _make_user(db_session, 1, "Admin")
    monkeypatch.setattr(reset_mod, "_running_job_ids", lambda: [7])
    r = await async_client.post("/api/v1/admin/reset", headers=headers,
                                json={"confirmation": RESET_CONFIRMATION_PHRASE})
    assert r.status_code == 409
    assert not sandboxed_storage["wiped"]["called"]


@pytest.mark.asyncio
async def test_admin_can_preview(async_client, db_session, sandboxed_storage):
    headers = await _make_user(db_session, 1, "Admin")
    r = await async_client.get("/api/v1/admin/reset/preview", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["confirmation_phrase"] == RESET_CONFIRMATION_PHRASE
    assert "evidence" in body["tables"]
