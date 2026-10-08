"""
Phase 1 — analysis-job isolation, canonical statistics, manifest integrity,
detector diagnostics, VLM provenance, and integrity migrations.
"""
import json
import uuid
from datetime import datetime, timedelta

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.system_reset_service as reset_mod
from app.authentication.jwt import create_access_token
from app.authentication.password import hash_password
from app.core.config import settings
from app.database.base import Base
from app.models.analysis import (
    AnalysisJob, Detection, FrameObservation, JobStage, JobStatus, Keyframe, Track,
)
from app.models.evidence import Evidence, EvidenceStatus
from app.models.semantic import DocumentSourceType, DocumentType, ForensicDocument, VLMObservation
from app.models.user import User
from app.services.agents.evidence_verifier import EvidenceVerifier
from app.services.agents.investigation_tools import InvestigationToolSystem, _scope_filter
from app.services.analysis_jobs import AnalysisJobNotFound, get_active_analysis_job
from app.services.analysis_statistics import STATS_SCHEMA_VERSION, compute_analysis_statistics
from app.services.pipeline.keyframe_extractor import KeyframeExtractor
from app.services.pipeline.manifest_generator import ManifestGenerator
from app.services.pipeline.object_detector import ObjectDetector


# ----------------------------------------------------------------------
# Seed helpers
# ----------------------------------------------------------------------

async def _user(db, role_id=3):
    tag = uuid.uuid4().hex[:8]
    u = User(full_name="U", username=f"u_{tag}", email=f"{tag}@t.local",
             password_hash=hash_password("Pw123456!"), role_id=role_id, is_active=True)
    db.add(u)
    await db.commit()
    await db.refresh(u)
    return u


async def _evidence(db, user):
    tag = uuid.uuid4().hex[:8]
    ev = Evidence(original_filename=f"{tag}.mp4", stored_filename=f"{tag}_s.mp4",
                  file_path=f"/tmp/{tag}.mp4", file_size=1, mime_type="video/mp4",
                  sha256_hash=tag.ljust(64, "d"), status=EvidenceStatus.COMPLETED,
                  uploaded_by=user.id)
    db.add(ev)
    await db.commit()
    await db.refresh(ev)
    return ev


async def _job(db, ev, status=JobStatus.COMPLETED, completed_at=None, **kw):
    job = AnalysisJob(evidence_id=ev.id, status=status, current_stage=JobStage.FINALIZING,
                      progress=100.0, completed_at=completed_at, **kw)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


def _track(job, n, cls):
    return Track(analysis_job_id=job.id, evidence_id=job.evidence_id, track_number=n,
                 class_name=cls, first_seen_timestamp=1.0, last_seen_timestamp=5.0,
                 first_seen_frame=10, last_seen_frame=50, duration=4.0,
                 observation_count=8, keyframe_count=1)


def _det(job, frame, cls="car", track=1):
    return Detection(analysis_job_id=job.id, evidence_id=job.evidence_id, frame_number=frame,
                     timestamp=frame / 10.0, class_name=cls, confidence=0.9,
                     bbox_x1=0.1, bbox_y1=0.1, bbox_x2=0.2, bbox_y2=0.2, track_id=track)


T0 = datetime(2026, 5, 1, 12, 0, 0)


# ======================================================================
# Canonical job resolver
# ======================================================================

@pytest.mark.asyncio
async def test_resolver_returns_latest_completed_and_ignores_others(db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    old = await _job(db_session, ev, completed_at=T0)
    new = await _job(db_session, ev, completed_at=T0 + timedelta(hours=1))
    await _job(db_session, ev, status=JobStatus.PROCESSING)
    await _job(db_session, ev, status=JobStatus.FAILED, completed_at=T0 + timedelta(hours=2))

    job = await get_active_analysis_job(db_session, ev.id)
    assert job.id == new.id != old.id


@pytest.mark.asyncio
async def test_resolver_breaks_completed_at_ties_by_id(db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    await _job(db_session, ev, completed_at=T0)
    second = await _job(db_session, ev, completed_at=T0)
    assert (await get_active_analysis_job(db_session, ev.id)).id == second.id


@pytest.mark.asyncio
async def test_resolver_rejects_job_of_other_evidence_when_strict(db_session):
    u = await _user(db_session)
    ev_a, ev_b = await _evidence(db_session, u), await _evidence(db_session, u)
    job_b = await _job(db_session, ev_b, completed_at=T0)
    job_a = await _job(db_session, ev_a, completed_at=T0)

    with pytest.raises(AnalysisJobNotFound):
        await get_active_analysis_job(db_session, ev_a.id, job_b.id, strict=True)
    # Non-strict falls back to evidence A's own latest job — never evidence B's.
    assert (await get_active_analysis_job(db_session, ev_a.id, job_b.id)).id == job_a.id


@pytest.mark.asyncio
async def test_resolver_rejects_unfinished_job_when_strict(db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    running = await _job(db_session, ev, status=JobStatus.PROCESSING)
    with pytest.raises(AnalysisJobNotFound):
        await get_active_analysis_job(db_session, ev.id, running.id, strict=True)


# ======================================================================
# Cross-run contamination
# ======================================================================

def test_scope_filter_with_missing_job_matches_nothing():
    conds = _scope_filter(Track, 1, None)
    rendered = " AND ".join(str(c) for c in conds)
    assert "analysis_job_id IS NULL" in rendered  # column is NOT NULL -> no rows


@pytest.mark.asyncio
async def test_tools_without_job_id_use_only_the_active_run(db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    job_a = await _job(db_session, ev, completed_at=T0)
    job_b = await _job(db_session, ev, completed_at=T0 + timedelta(hours=1))
    db_session.add_all([_track(job_a, 1, "car"), _track(job_a, 2, "car"), _track(job_b, 1, "car")])
    await db_session.commit()

    res = await InvestigationToolSystem.find_vehicles(db=db_session, evidence_id=ev.id)
    assert len(res) == 1  # job B only; previously 3 (both runs merged)


@pytest.mark.asyncio
async def test_tools_return_nothing_when_no_completed_run(db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    job = await _job(db_session, ev, status=JobStatus.FAILED)
    db_session.add(_track(job, 1, "car"))
    await db_session.commit()
    assert await InvestigationToolSystem.find_vehicles(db=db_session, evidence_id=ev.id) == []


@pytest.mark.asyncio
async def test_verifier_resolves_track_number_within_the_right_run(db_session):
    """Track #1 is a car in run A and a person in run B — they are different entities."""
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    job_a = await _job(db_session, ev, completed_at=T0)
    job_b = await _job(db_session, ev, completed_at=T0 + timedelta(hours=1))
    db_session.add_all([_track(job_a, 1, "car"), _track(job_b, 1, "person")])
    await db_session.commit()

    claim = "A vehicle was observed"
    in_b = await EvidenceVerifier.verify_finding(db_session, ev.id, claim, track_id=1,
                                                 relevance_score=0.8, analysis_job_id=job_b.id)
    in_a = await EvidenceVerifier.verify_finding(db_session, ev.id, claim, track_id=1,
                                                 relevance_score=0.8, analysis_job_id=job_a.id)
    default = await EvidenceVerifier.verify_finding(db_session, ev.id, claim, track_id=1,
                                                    relevance_score=0.8)
    assert in_b["status"] == "CONTRADICTED"
    assert in_a["status"] != "CONTRADICTED"
    assert default["status"] == "CONTRADICTED"  # defaults to the active run (B)


# ======================================================================
# Canonical statistics
# ======================================================================

@pytest.mark.asyncio
async def test_statistics_are_counted_per_job_from_the_database(db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    job_a = await _job(db_session, ev, completed_at=T0)
    job_b = await _job(db_session, ev, completed_at=T0)
    db_session.add_all([
        _track(job_a, 1, "car"), _track(job_a, 2, "person"), _track(job_b, 1, "car"),
        _det(job_a, 10), _det(job_a, 10, "person", 2), _det(job_a, 20), _det(job_b, 10),
        FrameObservation(analysis_job_id=job_a.id, evidence_id=ev.id, frame_number=10, timestamp=1.0),
        FrameObservation(analysis_job_id=job_a.id, evidence_id=ev.id, frame_number=20, timestamp=2.0),
        FrameObservation(analysis_job_id=job_a.id, evidence_id=ev.id, frame_number=30, timestamp=3.0),
    ])
    await db_session.commit()

    s = await compute_analysis_statistics(db_session, job_a)
    assert s.unique_tracks == 2
    assert s.stored_detections == 3
    assert s.frames_with_detections == 2  # distinct frames 10 and 20
    assert s.sampled_frames == 3
    assert (await compute_analysis_statistics(db_session, job_b)).stored_detections == 1


@pytest.mark.asyncio
async def test_legacy_runs_report_unknown_runtime_counters_not_zero(db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    legacy = await _job(db_session, ev, completed_at=T0, raw_detections=196)  # old meaning
    s = await compute_analysis_statistics(db_session, legacy)
    assert s.pipeline_counters_recorded is False
    assert s.raw_model_detections is None
    assert s.processed_frames is None
    assert s.confidence_filtered_detections is None


@pytest.mark.asyncio
async def test_current_runs_report_runtime_counters(db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    job = await _job(db_session, ev, completed_at=T0, stats_schema_version=STATS_SCHEMA_VERSION,
                     source_frames=302, processed_frames=21, raw_detections=400,
                     confidence_filtered_detections=250, class_filtered_detections=234,
                     event_candidates=3)
    s = await compute_analysis_statistics(db_session, job)
    assert s.pipeline_counters_recorded is True
    assert (s.raw_model_detections, s.confidence_filtered_detections,
            s.class_filtered_detections) == (400, 250, 234)
    assert s.face_observations is None  # no face model: unknown, not zero
    assert "processed_frames" in s.definitions


@pytest.mark.asyncio
async def test_vlm_statistic_excludes_legacy_template_rows(db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    job = await _job(db_session, ev, completed_at=T0)
    for i, model in enumerate(["VLM_Heuristic_Fallback", "Salesforce/blip-image-captioning-base"]):
        db_session.add(VLMObservation(evidence_id=ev.id, analysis_job_id=job.id, keyframe_id=1000 + i,
                                      model_name=model, model_version="1.0", description="x"))
    await db_session.commit()
    assert (await compute_analysis_statistics(db_session, job)).vlm_observations == 1


# ======================================================================
# Manifest integrity
# ======================================================================

def _manifest(**over):
    kw = dict(evidence_id=1, analysis_job_id=7, source_sha256="a" * 64, started_at=T0,
              completed_at=T0, model_name="yolo11n.pt", model_version="11", tracker_algorithm="t",
              sampling_fps=2.0, confidence_threshold=0.4, stats={"stored_detections": 3},
              keyframes=[])
    kw.update(over)
    return ManifestGenerator.generate_manifest(**kw)


def test_manifest_is_deterministic():
    assert _manifest()[1] == _manifest()[1]
    assert _manifest()[1] != _manifest(stats={"stored_detections": 4})[1]


def test_sealed_manifest_round_trip_verifies(tmp_path):
    data, digest = _manifest()
    path = tmp_path / "m.json"
    ManifestGenerator.save_manifest(path, data)
    loaded, actual, ok = ManifestGenerator.load_and_verify(path, digest)
    assert ok and actual == digest and loaded == json.loads(json.dumps(data, default=str))


def test_tampered_manifest_fails_verification(tmp_path):
    data, digest = _manifest()
    path = tmp_path / "m.json"
    ManifestGenerator.save_manifest(path, data)
    path.write_bytes(path.read_bytes().replace(b'"stored_detections": 3', b'"stored_detections": 9'))
    _, _, ok = ManifestGenerator.load_and_verify(path, digest)
    assert ok is False


def test_manifest_path_is_job_scoped():
    assert ManifestGenerator.manifest_path(1, 7) != ManifestGenerator.manifest_path(1, 8)


# ======================================================================
# Job-scoped keyframe files
# ======================================================================

def test_keyframes_from_two_runs_do_not_overwrite_each_other(tmp_path):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    a = KeyframeExtractor._save_keyframe(frame, 10, 1.0, "MOTION_CHANGE", 5, str(tmp_path), [],
                                         analysis_job_id=41)
    b = KeyframeExtractor._save_keyframe(frame, 10, 1.0, "MOTION_CHANGE", 5, str(tmp_path), [],
                                         analysis_job_id=42)
    assert a["image_path"] != b["image_path"]
    assert "_job41_" in a["image_path"] and "_job42_" in b["image_path"]
    assert len(list(tmp_path.iterdir())) == 2


# ======================================================================
# Detector diagnostics & no fabricated attributes
# ======================================================================

class _Box:
    def __init__(self, cls_id, conf):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array([10.0, 10.0, 40.0, 60.0])]


class _Result:
    names = {0: "person", 2: "car", 56: "chair"}

    def __init__(self, boxes):
        self.boxes = boxes
        self.orig_shape = (100, 100)


class _FakeYolo:
    def __init__(self):
        self.seen_conf = None

    def predict(self, frames, conf, iou, device, verbose):
        self.seen_conf = conf
        # sorted by confidence like ultralytics: 2 pass threshold+class, 1 passes
        # threshold but is an unsupported class, 2 are below threshold.
        return [_Result([_Box(2, 0.9), _Box(0, 0.7), _Box(56, 0.6), _Box(2, 0.3), _Box(0, 0.1)])
                for _ in frames]


def _detector(fake):
    d = ObjectDetector.__new__(ObjectDetector)
    d.model_name, d.confidence_threshold, d.iou_threshold = "fake", 0.4, 0.45
    d.max_processing_dim, d.yolo_model, d.pose_model, d.last_diagnostics = 1280, fake, None, {}
    return d


def test_detector_reports_every_filtering_stage(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_ATTRIBUTE_CLASSIFICATION", False)
    monkeypatch.setattr(settings, "ENABLE_ALPR", False)
    fake = _FakeYolo()
    det = _detector(fake)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = det.detect_batch([(0, 0.0, frame), (15, 0.5, frame)])

    assert fake.seen_conf == settings.DETECTOR_RAW_CONFIDENCE_FLOOR
    assert det.last_diagnostics == {
        "processed_frames": 2,
        "raw_model_detections": 10,
        "confidence_filtered_detections": 6,
        "class_filtered_detections": 4,
    }
    assert len(out) == 4
    assert all(d["confidence"] >= 0.4 for d in out)


def test_detector_does_not_fabricate_no_bag_observation(monkeypatch):
    """With attribute classification off, carries_bag is unknown, not False."""
    monkeypatch.setattr(settings, "ENABLE_ATTRIBUTE_CLASSIFICATION", False)
    monkeypatch.setattr(settings, "ENABLE_ALPR", False)
    out = _detector(_FakeYolo()).detect_batch([(0, 0.0, np.zeros((100, 100, 3), dtype=np.uint8))])
    person = [d for d in out if d["class_name"] == "person"][0]
    assert person["carries_bag"] is None

    from app.services.pipeline.attribute_aggregator import AttributeAggregator
    person["track_id"] = 1
    obs = AttributeAggregator.extract_observations([person])
    assert not any(o["attribute"] == "carries_bag" for o in obs)


# ======================================================================
# VLM provenance
# ======================================================================

@pytest.mark.asyncio
async def test_without_vlm_no_vlm_observation_is_recorded(db_session, monkeypatch):
    from app.services.semantic.vlm_service import VisionLanguageService as VLS
    monkeypatch.setattr(VLS, "_load_attempted", True)
    monkeypatch.setattr(VLS, "_vlm_model", None)
    monkeypatch.setattr(VLS, "_unavailable_reason", "test: no model")

    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    job = await _job(db_session, ev, completed_at=T0)
    db_session.add_all([
        Keyframe(analysis_job_id=job.id, evidence_id=ev.id, frame_number=10, timestamp=1.0,
                 image_path="/nope.jpg", sha256_hash="c" * 64),
        _det(job, 10, "car", 1), _det(job, 10, "car", 2),
    ])
    await db_session.commit()

    created = await VLS.analyze_keyframes_for_job(db_session, ev.id, job.id)
    assert created == []
    vlm_rows = (await db_session.execute(
        select(VLMObservation).where(VLMObservation.analysis_job_id == job.id))).scalars().all()
    assert vlm_rows == []

    doc = (await db_session.execute(
        select(ForensicDocument).where(ForensicDocument.analysis_job_id == job.id))).scalars().one()
    assert doc.source_type == DocumentSourceType.SYSTEM_GENERATED
    assert doc.content.startswith("[Detection Summary — not a VLM description]")
    assert doc.metadata_json["is_vlm"] is False
    assert "2 car(s)" in doc.content


# ======================================================================
# Integrity migrations & interrupted-job recovery
# ======================================================================

@pytest.mark.asyncio
async def test_integrity_migrations_dedupe_and_enforce_uniqueness():
    from app.database.init_db import apply_integrity_migrations

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    def vlm(model="m"):
        return VLMObservation(evidence_id=1, analysis_job_id=1, keyframe_id=5,
                              model_name=model, model_version="1", description="d")

    async with maker() as s:
        s.add_all([vlm("VLM_Heuristic_Fallback"), vlm("VLM_Heuristic_Fallback")])
        s.add(ForensicDocument(evidence_id=1, analysis_job_id=1, keyframe_id=5,
                               document_type=DocumentType.KEYFRAME,
                               source_type=DocumentSourceType.VLM_OBSERVATION,
                               title="t", content="c", metadata_json={}))
        await s.commit()

    await apply_integrity_migrations(engine)
    await apply_integrity_migrations(engine)  # idempotent

    async with maker() as s:
        assert len((await s.execute(select(VLMObservation))).scalars().all()) == 1
        doc = (await s.execute(select(ForensicDocument))).scalars().one()
        assert doc.source_type == DocumentSourceType.SYSTEM_GENERATED
        s.add(vlm())
        with pytest.raises(IntegrityError):
            await s.commit()
    await engine.dispose()


@pytest.mark.asyncio
async def test_interrupted_jobs_are_marked_failed_on_startup():
    from app.database.init_db import create_tables, recover_interrupted_jobs
    from app.database.session import SessionLocal, engine as app_engine

    assert "guardianeye_test_" in str(app_engine.url)  # never the real database
    await create_tables(app_engine)
    async with SessionLocal() as s:
        user = User(full_name="R", username=f"r_{uuid.uuid4().hex[:8]}", email=f"{uuid.uuid4().hex[:8]}@t",
                    password_hash="x", role_id=1, is_active=True)
        s.add(user)
        await s.commit()
        await s.refresh(user)
        ev = await _evidence(s, user)
        ev.status = EvidenceStatus.PROCESSING
        done = await _job(s, ev, completed_at=T0)
        stuck = await _job(s, ev, status=JobStatus.PROCESSING)
        stuck_id, done_id, ev_id = stuck.id, done.id, ev.id

    await recover_interrupted_jobs()

    async with SessionLocal() as s:
        stuck = (await s.execute(select(AnalysisJob).where(AnalysisJob.id == stuck_id))).scalars().one()
        done = (await s.execute(select(AnalysisJob).where(AnalysisJob.id == done_id))).scalars().one()
        ev = (await s.execute(select(Evidence).where(Evidence.id == ev_id))).scalars().one()
        assert stuck.status == JobStatus.FAILED and "Interrupted" in stuck.error_message
        assert done.status == JobStatus.COMPLETED
        assert ev.status == EvidenceStatus.COMPLETED  # it still has a good run


# ======================================================================
# Routes
# ======================================================================

def _auth(user, role):
    return {"Authorization": f"Bearer {create_access_token(user.id, user.username, role)}"}


@pytest.mark.asyncio
async def test_reprocess_creates_a_real_job_instead_of_faking_completion(
        async_client, db_session, monkeypatch):
    import app.services.analysis_runner as runner
    enqueued = []

    async def fake_run(job_id):
        enqueued.append(job_id)

    monkeypatch.setattr(runner, "run_analysis_job_async", fake_run)
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    ev.status = EvidenceStatus.UPLOADED
    await db_session.commit()

    r = await async_client.post(f"/api/v1/evidence/{ev.id}/reprocess", headers=_auth(u, "Investigator"))
    assert r.status_code == 200
    assert r.json()["status"] == "UPLOADED"  # not faked to COMPLETED
    jobs = (await db_session.execute(
        select(AnalysisJob).where(AnalysisJob.evidence_id == ev.id))).scalars().all()
    assert len(jobs) == 1 and enqueued == [jobs[0].id]


@pytest.mark.asyncio
async def test_statistics_route_rejects_job_of_other_evidence(async_client, db_session):
    u = await _user(db_session)
    ev_a, ev_b = await _evidence(db_session, u), await _evidence(db_session, u)
    await _job(db_session, ev_a, completed_at=T0)
    job_b = await _job(db_session, ev_b, completed_at=T0)

    h = _auth(u, "Investigator")
    ok = await async_client.get(f"/api/v1/evidence/{ev_a.id}/statistics", headers=h)
    assert ok.status_code == 200 and "definitions" in ok.json()
    bad = await async_client.get(f"/api/v1/evidence/{ev_a.id}/statistics?job_id={job_b.id}", headers=h)
    assert bad.status_code == 404


@pytest.mark.asyncio
async def test_legacy_manifest_is_flagged_unverified(async_client, db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    await _job(db_session, ev, completed_at=T0, manifest_hash="f" * 64)
    r = await async_client.get(f"/api/v1/evidence/{ev.id}/manifest", headers=_auth(u, "Investigator"))
    body = r.json()
    assert body["source"] == "RECONSTRUCTED" and body["integrity_verified"] is False


@pytest.mark.asyncio
async def test_analysis_jobs_listing_marks_active_run(async_client, db_session):
    u = await _user(db_session)
    ev = await _evidence(db_session, u)
    await _job(db_session, ev, completed_at=T0)
    newest = await _job(db_session, ev, completed_at=T0 + timedelta(hours=1))
    await _job(db_session, ev, status=JobStatus.FAILED)
    r = await async_client.get(f"/api/v1/evidence/{ev.id}/analysis-jobs", headers=_auth(u, "Investigator"))
    active = [j["job_id"] for j in r.json() if j["is_active"]]
    assert active == [newest.id]


@pytest.fixture
def sandboxed_evidence_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(reset_mod, "_derived_dir", lambda: tmp_path / "derived")
    monkeypatch.setattr(reset_mod, "_wipe_vector_collections_with_prefix",
                        lambda prefix: {"collections_deleted": 0, "errors": []})
    monkeypatch.setattr(reset_mod, "_running_job_ids", lambda: [])
    kf = tmp_path / "derived" / "keyframes"
    return kf


@pytest.mark.asyncio
async def test_reset_analysis_requires_admin(async_client, db_session, sandboxed_evidence_reset):
    u = await _user(db_session, role_id=3)
    ev = await _evidence(db_session, u)
    r = await async_client.post(f"/api/v1/evidence/{ev.id}/reset-analysis",
                                headers=_auth(u, "Investigator"),
                                json={"confirmation": f"RESET EVIDENCE {ev.id}"})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_reset_analysis_wipes_only_that_evidence(async_client, db_session, sandboxed_evidence_reset):
    admin = await _user(db_session, role_id=1)
    ev, other = await _evidence(db_session, admin), await _evidence(db_session, admin)
    job, other_job = await _job(db_session, ev, completed_at=T0), await _job(db_session, other, completed_at=T0)
    db_session.add_all([_track(job, 1, "car"), _det(job, 10), _track(other_job, 1, "car")])
    await db_session.commit()
    kf_dir = sandboxed_evidence_reset / str(ev.id)
    kf_dir.mkdir(parents=True)
    (kf_dir / "k.jpg").write_bytes(b"x")

    h = _auth(admin, "Admin")
    wrong = await async_client.post(f"/api/v1/evidence/{ev.id}/reset-analysis", headers=h,
                                    json={"confirmation": f"RESET EVIDENCE {other.id}"})
    assert wrong.status_code == 400

    r = await async_client.post(f"/api/v1/evidence/{ev.id}/reset-analysis", headers=h,
                                json={"confirmation": f"RESET EVIDENCE {ev.id}"})
    assert r.status_code == 200, r.text
    assert r.json()["deleted_rows"]["tracks"] == 1
    assert not kf_dir.exists()

    remaining = (await db_session.execute(select(Track).where(Track.evidence_id == other.id))).scalars().all()
    assert len(remaining) == 1  # other evidence untouched
    assert (await db_session.execute(select(Evidence).where(Evidence.id == ev.id))).scalars().one()
