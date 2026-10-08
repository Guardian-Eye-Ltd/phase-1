"""
Analysis runner orchestration: stage timing, event-loop responsiveness,
one-run-at-a-time scheduling and cancel-while-queued.

Model stages are replaced with fakes so these tests exercise the runner, not
YOLO/CLIP/OCR. The runner opens its own sessions, so it is pointed at an
isolated SQLite file for the duration of each test.
"""
import asyncio
import os
import threading
import time
import uuid

import cv2
import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.analysis_runner as runner
from app.database.init_db import create_tables
from app.models.analysis import AnalysisJob, JobStage, JobStatus, Track
from app.models.evidence import Evidence, EvidenceStatus
from app.models.role import Role
from app.models.user import User

STAGES = {"validate", "sample_frames", "motion", "detection", "tracking",
          "keyframes", "interactions", "events", "manifest", "semantic_index"}


class FakeDetector:
    """One person walking left to right; optionally blocks like real inference."""
    block_seconds = 0.0
    active = 0
    max_active = 0
    running_snapshots = []
    _lock = threading.Lock()

    def __init__(self, **kwargs):
        self.last_diagnostics = {}

    def detect_batch(self, frames, batch_size=16):
        cls = FakeDetector
        with cls._lock:
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
            cls.running_snapshots.append(sorted(runner.running_analysis_jobs))
        try:
            time.sleep(cls.block_seconds)   # blocking, like real model inference
        finally:
            with cls._lock:
                cls.active -= 1
        dets = []
        for i, (fn, ts, _) in enumerate(frames):
            x = 0.1 + 0.05 * i
            dets.append({"frame_number": fn, "timestamp": ts, "class_name": "person",
                         "confidence": 0.9, "bbox_x1": x, "bbox_y1": 0.3,
                         "bbox_x2": x + 0.1, "bbox_y2": 0.7, "carries_bag": None})
        self.last_diagnostics = {"processed_frames": len(frames), "raw_model_detections": len(dets),
                                 "confidence_filtered_detections": 0, "class_filtered_detections": 0}
        return dets


@pytest_asyncio.fixture
async def runner_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path.as_posix()}/runner.db")
    await create_tables(engine)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as s:
        s.add_all([Role(id=1, role_name="Admin"), Role(id=3, role_name="Investigator")])
        await s.commit()

    async def no_index(evidence_id, job_id):
        return None

    monkeypatch.setattr(runner, "SessionLocal", maker)
    monkeypatch.setattr(runner, "ObjectDetector", FakeDetector)
    monkeypatch.setattr(runner, "read_vehicle_plates", lambda *a, **k: [])
    monkeypatch.setattr(runner.FaceEngine, "load", staticmethod(lambda: False))
    monkeypatch.setattr(runner.SemanticIndexer, "index_evidence_async", staticmethod(no_index))
    monkeypatch.setattr(runner, "_analysis_slot", asyncio.Semaphore(1))  # bound to this test's loop
    FakeDetector.block_seconds, FakeDetector.active, FakeDetector.max_active = 0.0, 0, 0
    FakeDetector.running_snapshots = []
    yield maker
    await engine.dispose()


def _video(tmp_path):
    path = os.path.join(tmp_path, f"{uuid.uuid4().hex[:8]}.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for i in range(30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20 + i * 8, 80), (60 + i * 8, 160), (255, 255, 255), -1)
        out.write(frame)
    out.release()
    return path


async def _queued_job(maker, video_path, status=JobStatus.QUEUED):
    tag = uuid.uuid4().hex[:8]
    async with maker() as s:
        u = User(full_name="R", username=f"r_{tag}", email=f"{tag}@t.local",
                 password_hash="x", role_id=3, is_active=True)
        s.add(u)
        await s.commit()
        ev = Evidence(original_filename=f"{tag}.mp4", stored_filename=f"{tag}.mp4", file_path=video_path,
                      file_size=os.path.getsize(video_path), mime_type="video/mp4",
                      sha256_hash=tag.ljust(64, "e"), status=EvidenceStatus.UPLOADED, uploaded_by=u.id)
        s.add(ev)
        await s.commit()
        job = AnalysisJob(evidence_id=ev.id, status=status, current_stage=JobStage.VALIDATING)
        s.add(job)
        await s.commit()
        return job.id


async def _load(maker, job_id):
    async with maker() as s:
        return (await s.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))).scalar_one()


@pytest.mark.asyncio
async def test_completed_run_records_per_stage_timings(runner_db, tmp_path):
    job_id = await _queued_job(runner_db, _video(tmp_path))
    await runner.run_analysis_job_async(job_id)

    job = await _load(runner_db, job_id)
    assert job.status == JobStatus.COMPLETED, job.error_message
    assert STAGES <= set(job.stage_timings), job.stage_timings
    assert all(isinstance(v, float) and v >= 0 for v in job.stage_timings.values())
    async with runner_db() as s:
        tracks = (await s.execute(select(Track).where(Track.analysis_job_id == job_id))).scalars().all()
    assert len(tracks) == 1   # tracking ran on the worker thread and was persisted
    assert job_id not in runner.running_analysis_jobs


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_while_a_stage_blocks(runner_db, tmp_path):
    """Previously every stage ran on the event loop: the API froze for 37-65 s per analysis."""
    FakeDetector.block_seconds = 1.0
    job_id = await _queued_job(runner_db, _video(tmp_path))

    gaps, stop = [], asyncio.Event()

    async def heartbeat():
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.05)
            now = time.perf_counter()
            gaps.append(now - last - 0.05)
            last = now

    hb = asyncio.create_task(heartbeat())
    await runner.run_analysis_job_async(job_id)
    stop.set()
    await hb

    assert (await _load(runner_db, job_id)).status == JobStatus.COMPLETED
    assert max(gaps) < 0.5, f"event loop stalled for {max(gaps):.2f}s"


@pytest.mark.asyncio
async def test_runs_are_serialised_and_queued_runs_count_as_running(runner_db, tmp_path):
    FakeDetector.block_seconds = 0.3
    video = _video(tmp_path)
    a = await _queued_job(runner_db, video)
    b = await _queued_job(runner_db, video)

    await asyncio.gather(runner.run_analysis_job_async(a), runner.run_analysis_job_async(b))

    assert FakeDetector.max_active == 1
    # While the first run was detecting, the queued one was already registered,
    # so a system reset would have been refused.
    assert FakeDetector.running_snapshots[0] == sorted([a, b])
    for job_id in (a, b):
        assert (await _load(runner_db, job_id)).status == JobStatus.COMPLETED
    assert not ({a, b} & runner.running_analysis_jobs)


@pytest.mark.asyncio
async def test_job_cancelled_while_queued_is_not_resurrected(runner_db, tmp_path):
    job_id = await _queued_job(runner_db, _video(tmp_path), status=JobStatus.CANCELLED)
    await runner.run_analysis_job_async(job_id)

    job = await _load(runner_db, job_id)
    assert job.status == JobStatus.CANCELLED
    assert job.started_at is None
    assert FakeDetector.running_snapshots == []   # no stage ran
    assert job_id not in runner.running_analysis_jobs
