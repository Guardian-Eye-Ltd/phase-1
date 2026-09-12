import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.database.base import Base
from app.models.user import User
from app.models.role import Role
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

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture
async def async_session():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session

    await engine.dispose()

@pytest.fixture
async def seeded_evidence(async_session: AsyncSession):
    user = User(full_name="Investigator", username="inv", email="inv@test.io", password_hash="hash", role_id=1)
    async_session.add(user)
    await async_session.commit()

    evidence = Evidence(
        original_filename="cctv_test.mp4",
        stored_filename="cctv_test_uuid.mp4",
        file_path="storage/evidence/cctv_test_uuid.mp4",
        file_size=1024,
        mime_type="video/mp4",
        duration=180.0,
        fps=30.0,
        resolution="1920x1080",
        sha256_hash="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        status=EvidenceStatus.COMPLETED,
        uploaded_by=user.id
    )
    async_session.add(evidence)
    await async_session.commit()

    job = AnalysisJob(
        evidence_id=evidence.id,
        status=JobStatus.COMPLETED,
        current_stage=JobStage.FINALIZING,
        progress=100.0
    )
    async_session.add(job)
    await async_session.commit()

    # Seed Person Track 17
    track17 = Track(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        track_number=17,
        class_name="person",
        first_seen_timestamp=41.0,
        last_seen_timestamp=133.0,
        first_seen_frame=80,
        last_seen_frame=260,
        duration=92.0,
        observation_count=15
    )
    async_session.add(track17)

    # Seed Keyframe 42
    kf42 = Keyframe(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        frame_number=180,
        timestamp=92.0,
        image_path="storage/derived/kf_42.jpg",
        sha256_hash="kfhash42"
    )
    async_session.add(kf42)

    # Seed Backpack Detection linked to Track 17
    det_backpack = Detection(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        frame_number=180,
        timestamp=92.0,
        class_name="backpack",
        confidence=0.91,
        bbox_x1=0.1, bbox_y1=0.1, bbox_x2=0.3, bbox_y2=0.3,
        track_id=17
    )
    async_session.add(det_backpack)

    # Seed Person Detection linked to Track 17
    det_person = Detection(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        frame_number=180,
        timestamp=92.0,
        class_name="person",
        confidence=0.94,
        bbox_x1=0.1, bbox_y1=0.1, bbox_x2=0.5, bbox_y2=0.9,
        track_id=17
    )
    async_session.add(det_person)

    # Seed Activity Interval
    activity = ActivityInterval(
        analysis_job_id=job.id,
        evidence_id=evidence.id,
        start_time=1.0,
        end_time=10.0,
        start_frame=30,
        end_frame=300,
        motion_score=0.15,
        activity_level=ActivityLevel.LOW
    )
    async_session.add(activity)

    await async_session.commit()

    # Generate documents
    docs = await ForensicDocumentGenerator.generate_documents_for_job(
        db=async_session,
        evidence_id=evidence.id,
        analysis_job_id=job.id
    )

    return {"user": user, "evidence": evidence, "job": job, "docs": docs}

@pytest.mark.asyncio
async def test_document_generation_and_intent_parsing(async_session: AsyncSession, seeded_evidence):
    evidence = seeded_evidence["evidence"]
    user = seeded_evidence["user"]
    docs = seeded_evidence["docs"]

    assert len(docs) >= 2
    track_doc = next((d for d in docs if d.track_id == 17), None)
    assert track_doc is not None
    assert "Person Track 17" in track_doc.content
    assert "backpack" in track_doc.metadata_json["associated_objects"]

    # Test Query Intent Parser
    intent = QueryIntentParser.parse_query("Find a person with a backpack near Track 17 between 00:40 and 02:00")
    assert intent["track_id"] == 17
    assert "person" in intent["entities"]
    assert "backpack" in intent["objects"]
    assert intent["start_time"] == 40.0
    assert intent["end_time"] == 120.0

@pytest.mark.asyncio
async def test_1_person_clothing_query_excludes_activity_interval(async_session: AsyncSession, seeded_evidence):
    """Test 1: Query 'Find a person wearing a white shirt' must NOT return ACTIVITY_INTERVAL as top result."""
    evidence = seeded_evidence["evidence"]
    user = seeded_evidence["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_session,
        evidence_id=evidence.id,
        query_text="Find a person wearing a white shirt",
        user_id=user.id
    )

    # Either no results or top result is a person track/keyframe, NEVER an ACTIVITY_INTERVAL
    if search_res["total_results"] > 0:
        top_res = search_res["results"][0]
        assert top_res["why_explanation"]["document_type"] != "ACTIVITY_INTERVAL"
    else:
        assert search_res["total_results"] == 0
        assert "NO SUPPORTED EVIDENCE" in search_res["answer"] or "no evidence meeting" in search_res["answer"]

@pytest.mark.asyncio
async def test_2_activity_query_allows_activity_interval(async_session: AsyncSession, seeded_evidence):
    """Test 2: Query 'What happened around 01:30?' allows ACTIVITY_INTERVAL or timeline events."""
    evidence = seeded_evidence["evidence"]
    user = seeded_evidence["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_session,
        evidence_id=evidence.id,
        query_text="What happened around 00:05?",
        user_id=user.id
    )

    assert search_res["extracted_intent"]["is_activity_query"] is True

@pytest.mark.asyncio
async def test_3_track_query_direct_lookup(async_session: AsyncSession, seeded_evidence):
    """Test 3: Query 'Show Track 17' must directly retrieve Track 17."""
    evidence = seeded_evidence["evidence"]
    user = seeded_evidence["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_session,
        evidence_id=evidence.id,
        query_text="Show Track 17",
        user_id=user.id
    )

    assert search_res["total_results"] >= 1
    top_res = search_res["results"][0]
    assert top_res["track_id"] == 17
    assert top_res["query_relevance"] == 100.0

@pytest.mark.asyncio
async def test_4_person_backpack_query(async_session: AsyncSession, seeded_evidence):
    """Test 4: Query 'Find a person carrying a backpack' top results contain PERSON + BACKPACK evidence."""
    evidence = seeded_evidence["evidence"]
    user = seeded_evidence["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_session,
        evidence_id=evidence.id,
        query_text="Find a person carrying a backpack",
        user_id=user.id
    )

    assert search_res["total_results"] > 0
    top_res = search_res["results"][0]
    assert top_res["track_id"] == 17
    assert top_res["confidence_level"] == "HIGH"
    assert top_res["evidence_support"] == "HIGH"

@pytest.mark.asyncio
async def test_5_unsupported_low_relevance_query(async_session: AsyncSession, seeded_evidence):
    """Test 5: Very low semantic match returns NO_SUPPORTED_EVIDENCE and excludes weak matches."""
    evidence = seeded_evidence["evidence"]
    user = seeded_evidence["user"]

    search_res = await HybridSearchEngine.execute_search(
        db=async_session,
        evidence_id=evidence.id,
        query_text="Find a flying dragon carrying a golden treasure",
        user_id=user.id
    )

    assert search_res["total_results"] == 0
    assert "NO SUPPORTED EVIDENCE" in search_res["answer"]
