import os
import pytest
import cv2
import numpy as np
import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.database.base import Base
from app.models.user import User
from app.models.evidence import Evidence, EvidenceStatus
from app.models.analysis import (
    AnalysisJob, JobStatus, JobStage, Track, Keyframe, ActivityInterval, ActivityLevel, PossibleInteraction, Detection
)
from app.models.semantic import ForensicDocument, DocumentType, DocumentSourceType, ConfidenceLevel
from app.services.semantic.document_generator import ForensicDocumentGenerator
from app.services.semantic.vlm_service import VisionLanguageService
from app.services.semantic.vector_service import VectorService
from app.services.semantic.query_parser import QueryIntentParser
from app.services.semantic.hybrid_search_engine import HybridSearchEngine
from app.services.pipeline.video_validator import VideoValidator
from app.services.pipeline.frame_sampler import FrameSampler
from app.services.pipeline.motion_filter import MotionFilter
from app.services.pipeline.object_detector import ObjectDetector
from app.services.pipeline.multi_object_tracker import MultiObjectTracker
from app.services.pipeline.keyframe_extractor import KeyframeExtractor
from app.services.pipeline.interaction_detector import InteractionDetector
from app.services.pipeline.manifest_generator import ManifestGenerator

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture
async def async_db_session():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session

    await engine.dispose()

@pytest.fixture
async def seeded_test_environment(async_db_session: AsyncSession):
    user = User(full_name="Investigator Lead", username="lead", email="lead@guardianeye.io", password_hash="hash", role_id=1)
    async_db_session.add(user)
    await async_db_session.commit()

    evidence = Evidence(
        original_filename="cctv_audit_feed.mp4",
        stored_filename="cctv_audit_feed_uuid.mp4",
        file_path="storage/evidence/cctv_audit_feed_uuid.mp4",
        file_size=2048500,
        mime_type="video/mp4",
        duration=120.0,
        fps=30.0,
        resolution="1920x1080",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        status=EvidenceStatus.COMPLETED,
        uploaded_by=user.id
    )
    async_db_session.add(evidence)
    await async_db_session.commit()

    job = AnalysisJob(
        evidence_id=evidence.id,
        status=JobStatus.COMPLETED,
        current_stage=JobStage.FINALIZING,
        progress=100.0,
        model_name="yolov8n.pt",
        sampling_fps=2.0,
        confidence_threshold=0.4
    )
    async_db_session.add(job)
    await async_db_session.commit()

    # Seed Person Track 17
    track17 = Track(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        track_number=17,
        class_name="person",
        first_seen_timestamp=40.0,
        last_seen_timestamp=100.0,
        first_seen_frame=80,
        last_seen_frame=200,
        duration=60.0,
        observation_count=20
    )
    async_db_session.add(track17)

    # Seed Car Track 22
    track22 = Track(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        track_number=22,
        class_name="car",
        first_seen_timestamp=10.0,
        last_seen_timestamp=110.0,
        first_seen_frame=20,
        last_seen_frame=220,
        duration=100.0,
        observation_count=35
    )
    async_db_session.add(track22)

    # Seed Keyframe 42
    kf42 = Keyframe(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        frame_number=180,
        timestamp=90.0,
        image_path="storage/derived/keyframes/1/kf_42.jpg",
        sha256_hash="kfhash42",
        track_ids={"tracks": [17, 22]},
        detection_ids={"classes": ["person", "car", "backpack"]}
    )
    async_db_session.add(kf42)

    # Seed Person Detection
    det_person = Detection(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        frame_number=180,
        timestamp=90.0,
        class_name="person",
        confidence=0.94,
        bbox_x1=0.1, bbox_y1=0.1, bbox_x2=0.4, bbox_y2=0.9,
        track_id=17
    )
    async_db_session.add(det_person)

    # Seed Backpack Detection
    det_backpack = Detection(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        frame_number=180,
        timestamp=90.0,
        class_name="backpack",
        confidence=0.88,
        bbox_x1=0.15, bbox_y1=0.2, bbox_x2=0.35, bbox_y2=0.5,
        track_id=17
    )
    async_db_session.add(det_backpack)

    # Seed Spatial Interaction
    inter = PossibleInteraction(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        entity_a_track_id=17,
        entity_b_track_id=22,
        start_time=45.0,
        end_time=95.0,
        min_distance_or_overlap=0.12,
        confidence_score=0.89,
        label="Near Track 22 (car)"
    )
    async_db_session.add(inter)

    # Seed Activity Interval
    activity = ActivityInterval(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        start_time=1.0,
        end_time=15.0,
        start_frame=2,
        end_frame=30,
        motion_score=0.10,
        activity_level=ActivityLevel.LOW
    )
    async_db_session.add(activity)

    await async_db_session.commit()

    # Generate documents
    docs = await ForensicDocumentGenerator.generate_documents_for_job(
        db=async_db_session,
        evidence_id=evidence.id,
        analysis_job_id=job.id
    )

    return {"user": user, "evidence": evidence, "job": job, "docs": docs}

@pytest.mark.asyncio
async def test_scenario_1_person_query_returns_person_tracks(async_db_session: AsyncSession, seeded_test_environment):
    """Scenario 1: 'Find people' query returns person tracks/keyframes, excluding activity intervals."""
    evidence = seeded_test_environment["evidence"]
    user = seeded_test_environment["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_db_session,
        evidence_id=evidence.id,
        query_text="Find people",
        user_id=user.id
    )

    assert search_res["extracted_intent"]["query_type"] == "OBJECT_SEARCH"
    assert "ACTIVITY_INTERVAL" not in search_res["extracted_intent"]["candidate_document_types"]
    assert search_res["total_results"] > 0
    top_result = search_res["results"][0]
    assert top_result["why_explanation"]["document_type"] in ["TRACK", "KEYFRAME"]

@pytest.mark.asyncio
async def test_scenario_2_vehicle_query_returns_cars(async_db_session: AsyncSession, seeded_test_environment):
    """Scenario 2: 'Find cars' query returns vehicle tracks."""
    evidence = seeded_test_environment["evidence"]
    user = seeded_test_environment["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_db_session,
        evidence_id=evidence.id,
        query_text="Find cars",
        user_id=user.id
    )

    assert search_res["total_results"] > 0
    top_result = search_res["results"][0]
    assert top_result["track_id"] == 22 or "Car" in top_result["title"]

@pytest.mark.asyncio
async def test_scenario_3_clothing_attribute_query_no_hallucination(async_db_session: AsyncSession, seeded_test_environment):
    """Scenario 3: 'Find a person wearing a white shirt' must not hallucinate activity intervals."""
    evidence = seeded_test_environment["evidence"]
    user = seeded_test_environment["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_db_session,
        evidence_id=evidence.id,
        query_text="Find a person wearing a white shirt",
        user_id=user.id
    )

    assert search_res["extracted_intent"]["query_type"] == "ATTRIBUTE_SEARCH"
    if search_res["total_results"] == 0:
        assert "NO SUPPORTED EVIDENCE" in search_res["answer"]
    else:
        top_res = search_res["results"][0]
        assert top_res["why_explanation"]["document_type"] != "ACTIVITY_INTERVAL"

@pytest.mark.asyncio
async def test_scenario_4_carrying_object_query(async_db_session: AsyncSession, seeded_test_environment):
    """Scenario 4: 'Find a person carrying a backpack' returns PERSON + BACKPACK evidence."""
    evidence = seeded_test_environment["evidence"]
    user = seeded_test_environment["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_db_session,
        evidence_id=evidence.id,
        query_text="Find a person carrying a backpack",
        user_id=user.id
    )

    assert search_res["total_results"] > 0
    top_res = search_res["results"][0]
    assert top_res["track_id"] == 17
    assert top_res["evidence_support"] == "HIGH"

@pytest.mark.asyncio
async def test_scenario_5_relationship_query(async_db_session: AsyncSession, seeded_test_environment):
    """Scenario 5: 'Find people near cars' returns spatial interaction or proximity evidence."""
    evidence = seeded_test_environment["evidence"]
    user = seeded_test_environment["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_db_session,
        evidence_id=evidence.id,
        query_text="Find people near cars",
        user_id=user.id
    )

    assert search_res["extracted_intent"]["query_type"] == "RELATIONSHIP_SEARCH"
    assert search_res["total_results"] > 0

@pytest.mark.asyncio
async def test_scenario_6_track_lookup_query(async_db_session: AsyncSession, seeded_test_environment):
    """Scenario 6: 'Show Track 17' directly retrieves Track 17 with 100% relevance."""
    evidence = seeded_test_environment["evidence"]
    user = seeded_test_environment["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_db_session,
        evidence_id=evidence.id,
        query_text="Show Track 17",
        user_id=user.id
    )

    assert search_res["total_results"] == 1
    top_res = search_res["results"][0]
    assert top_res["track_id"] == 17
    assert top_res["query_relevance"] == 100.0

@pytest.mark.asyncio
async def test_scenario_7_activity_query(async_db_session: AsyncSession, seeded_test_environment):
    """Scenario 7: 'What happened around 00:05?' returns activity interval timeline events."""
    evidence = seeded_test_environment["evidence"]
    user = seeded_test_environment["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_db_session,
        evidence_id=evidence.id,
        query_text="What happened around 00:05?",
        user_id=user.id
    )

    assert search_res["extracted_intent"]["query_type"] in ["ACTIVITY_SEARCH", "TEMPORAL_SEARCH"]
    assert "ACTIVITY_INTERVAL" in search_res["extracted_intent"]["candidate_document_types"]

@pytest.mark.asyncio
async def test_scenario_8_unsupported_query_no_evidence(async_db_session: AsyncSession, seeded_test_environment):
    """Scenario 8: Completely unsupported query returns NO_SUPPORTED_EVIDENCE answer state."""
    evidence = seeded_test_environment["evidence"]
    user = seeded_test_environment["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_db_session,
        evidence_id=evidence.id,
        query_text="Find a flying dragon carrying a golden treasure",
        user_id=user.id
    )

    assert search_res["total_results"] == 0
    assert "NO SUPPORTED EVIDENCE" in search_res["answer"]

@pytest.mark.asyncio
async def test_scenario_9_separate_confidence_metrics(async_db_session: AsyncSession, seeded_test_environment):
    """Scenario 9: Search results expose separate query_relevance, detection_confidence, evidence_support, and verification_status."""
    evidence = seeded_test_environment["evidence"]
    user = seeded_test_environment["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_db_session,
        evidence_id=evidence.id,
        query_text="Find a person carrying a backpack",
        user_id=user.id
    )

    assert search_res["total_results"] > 0
    top_res = search_res["results"][0]
    assert "query_relevance" in top_res
    assert "evidence_support" in top_res
    assert "model_confidence" in top_res
    assert "verification_status" in top_res
    assert top_res["verification_status"] == "VERIFIED"

def test_scenario_10_end_to_end_pipeline_performance(tmp_path):
    """Scenario 10: Validates full video processing pipeline components on synthetic CCTV video."""
    video_file = os.path.join(tmp_path, "cctv_synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, 10.0, (640, 480))

    for i in range(20):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x_pos = int(50 + i * 15)
        cv2.rectangle(frame, (x_pos, 100), (x_pos + 60, 300), (255, 255, 255), -1)
        out.write(frame)

    out.release()

    # 1. Validation
    info = VideoValidator.validate_video(video_file)
    assert info["valid"] is True
    assert info["width"] == 640
    assert info["height"] == 480

    # 2. Sampling
    sampled = list(FrameSampler.sample_frames(video_file, target_fps=2.0))
    assert len(sampled) > 0

    # 3. Motion
    motion = MotionFilter.analyze_motion(sampled)
    assert isinstance(motion, list)

    # 4. Detector & Tracker
    detector = ObjectDetector(confidence_threshold=0.3)
    tracker = MultiObjectTracker()

    raw_dets = detector.detect_batch(sampled)
    dets_by_frame = {}
    for d in raw_dets:
        dets_by_frame.setdefault(d["frame_number"], []).append(d)

    all_tracked = []
    for fn, ts, _ in sampled:
        frame_dets = dets_by_frame.get(fn, [])
        if frame_dets:
            tracked_frame_dets = tracker.process_frame_detections(frame_dets)
            all_tracked.extend(tracked_frame_dets)

    summaries = tracker.generate_track_summaries(all_tracked)
    
    # 5. Keyframes
    derived_dir = os.path.join(tmp_path, "derived")
    keyframes = KeyframeExtractor.extract_keyframes(sampled, all_tracked, summaries, 1, derived_dir)
    assert isinstance(keyframes, list)

    # 6. Manifest
    manifest_dict, manifest_hash = ManifestGenerator.generate_manifest(
        evidence_id=1,
        analysis_job_id=10,
        source_sha256="synthhash123",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        model_name="yolov8n.pt",
        model_version="8.2.0",
        tracker_algorithm="Hybrid-IoU-Centroid",
        sampling_fps=2.0,
        confidence_threshold=0.4,
        stats={"total_frames": len(sampled)},
        keyframes=keyframes
    )
    assert len(manifest_hash) == 64
