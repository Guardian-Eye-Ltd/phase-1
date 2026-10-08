"""
Tests for the intent-first investigation architecture.

Covers:
    - QueryIntentParser produces correct structured_intent for COUNT, FRAME_COUNT,
      ATTRIBUTE_SEARCH, EVENT_SEARCH, SUMMARY, SEARCH.
    - count_entities deduplicates track ids, respects analysis_job_id isolation,
      and never sums per-detection rows.
    - count_entities_at_time only includes tracks active at the given timestamp.
    - EvidenceVerifier.verify_count_result status reflects deterministic checks.
"""
import pytest
import pytest_asyncio
import uuid
from datetime import datetime

from app.services.semantic.query_parser import (
    QueryIntentParser,
    INTENT_COUNT, INTENT_FRAME_COUNT, INTENT_ATTRIBUTE_SEARCH,
    INTENT_EVENT_SEARCH, INTENT_SUMMARY, INTENT_SEARCH,
)
from app.services.agents.investigation_tools import InvestigationToolSystem
from app.services.agents.evidence_verifier import EvidenceVerifier
from app.services.agents.taxonomy import normalize_entity_class
from app.models.evidence import Evidence, EvidenceStatus
from app.models.analysis import AnalysisJob, JobStatus, Track, Detection, JobStage
from app.models.user import User
from app.authentication.password import hash_password


# ======================================================================
# Parser tests — pure, no DB
# ======================================================================

def _si(q):
    return QueryIntentParser.parse_query(q)["structured_intent"]


def test_total_number_of_vehicles_is_count():
    r = _si("total number of vehicles")
    assert r["intent"] == INTENT_COUNT
    assert r["entity"] == "vehicle"
    assert r["aggregation"] == "DISTINCT_TRACK_COUNT"
    assert r["scope"] == "ENTIRE_VIDEO"


def test_how_many_cars_is_count_car():
    r = _si("how many cars are there?")
    assert r["intent"] == INTENT_COUNT
    assert r["entity"] == "car"


def test_how_many_buses_is_count_bus():
    r = _si("how many buses")
    assert r["intent"] == INTENT_COUNT
    assert r["entity"] == "bus"


def test_frame_count_at_timestamp():
    r = _si("how many vehicles are visible at 9 seconds?")
    assert r["intent"] == INTENT_FRAME_COUNT
    assert r["entity"] == "vehicle"
    assert r["timestamp"] == 9.0
    assert r["aggregation"] == "DISTINCT_ACTIVE_TRACK_COUNT"


def test_frame_count_mmss():
    r = _si("how many cars at 01:20")
    assert r["intent"] == INTENT_FRAME_COUNT
    assert r["timestamp"] == 80.0


def test_attribute_search_red_car():
    r = _si("find the red car")
    assert r["intent"] == INTENT_ATTRIBUTE_SEARCH
    assert r["entity"] == "car"
    # Key matches the TrackAttributeAggregate column so tools need no translation.
    assert r["attributes"].get("vehicle_color") == "red"


def test_find_people_wearing_black_maps_to_upper_garment():
    r = _si("find people wearing black")
    assert r["intent"] == INTENT_ATTRIBUTE_SEARCH
    assert r["entity"] == "person"
    assert r["attributes"].get("upper_garment_color") == "black"


def test_lower_garment_named_explicitly_maps_to_lower():
    r = _si("find a person wearing blue jeans")
    assert r["intent"] == INTENT_ATTRIBUTE_SEARCH
    assert r["attributes"].get("lower_garment_color") == "blue"


def test_plate_query_extracts_plate():
    r = _si("find vehicle KL01AB1234")
    assert r["intent"] == INTENT_ATTRIBUTE_SEARCH
    assert r["attributes"].get("license_plate_text") == "KL01AB1234"


def test_plate_query_with_explicit_cue():
    r = _si("find the car with plate number MH12XY9876")
    assert r["attributes"].get("license_plate_text") == "MH12XY9876"


def test_ordinary_words_are_not_mistaken_for_plates():
    """A plate must mix letters and digits — 'black' and '123456' are not plates."""
    for q in ("find people wearing black", "how many vehicles", "find a red car"):
        assert "license_plate_text" not in (_si(q).get("attributes") or {})


def test_backpack_query_targets_person_carrying():
    r = _si("find the person carrying a backpack")
    assert r["intent"] == INTENT_ATTRIBUTE_SEARCH
    assert r["entity"] == "person"
    assert r["attributes"].get("carries_bag") == "true"


def test_attribute_search_white_shirt_person():
    r = _si("find a person wearing a white shirt")
    assert r["intent"] == INTENT_ATTRIBUTE_SEARCH
    assert r["entity"] == "person"
    assert r["attributes"].get("upper_garment_color") == "white"


def test_event_search_sudden_acceleration():
    r = _si("find sudden acceleration")
    assert r["intent"] == INTENT_EVENT_SEARCH
    assert r["event_type"] == "SUDDEN_ACCELERATION"


def test_summary_intent():
    r = _si("summarize the video")
    assert r["intent"] == INTENT_SUMMARY


def test_default_search_intent():
    r = _si("show me interesting things")
    assert r["intent"] in (INTENT_SEARCH, INTENT_SUMMARY)


def test_normalize_entity_class_maps_to_canonical():
    assert normalize_entity_class("car") == "car"
    assert normalize_entity_class("CAR") == "car"
    assert normalize_entity_class("vehicle") == "vehicle"
    assert normalize_entity_class("pedestrian") == "person"


# ======================================================================
# DB-backed tests for count_entities / count_entities_at_time
# ======================================================================

@pytest_asyncio.fixture
async def seeded_investigation_env(db_session):
    """
    Seeds:
        user
        evidence #1
        analysis_job #15 (completed): 2 car tracks
        analysis_job #16 (completed, newest): 2 cars + 1 bus
        detections for each track (multiple per track, to test de-dup)
    """
    tag = uuid.uuid4().hex[:8]
    user = User(
        full_name="Investigator",
        username=f"inv_{tag}",
        email=f"inv_{tag}@test.local",
        password_hash=hash_password("Testing123!"),
        role_id=3,
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    evidence = Evidence(
        original_filename=f"test_{tag}.mp4",
        stored_filename=f"test_stored_{tag}.mp4",
        file_path=f"/tmp/test_{tag}.mp4",
        file_size=1024,
        mime_type="video/mp4",
        sha256_hash=tag.ljust(64, "a"),
        status=EvidenceStatus.COMPLETED,
        uploaded_by=user.id,
    )
    db_session.add(evidence)
    await db_session.commit()
    await db_session.refresh(evidence)

    job_a = AnalysisJob(
        evidence_id=evidence.id,
        status=JobStatus.COMPLETED,
        current_stage=JobStage.FINALIZING,
        progress=100.0,
        completed_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    job_b = AnalysisJob(
        evidence_id=evidence.id,
        status=JobStatus.COMPLETED,
        current_stage=JobStage.FINALIZING,
        progress=100.0,
        completed_at=datetime(2026, 1, 2, 12, 0, 0),
    )
    db_session.add_all([job_a, job_b])
    await db_session.commit()
    await db_session.refresh(job_a)
    await db_session.refresh(job_b)

    # Job A: track 1 (car), track 2 (car)
    tracks_a = [
        Track(
            analysis_job_id=job_a.id, evidence_id=evidence.id,
            track_number=1, class_name="car",
            first_seen_timestamp=1.0, last_seen_timestamp=5.0,
            first_seen_frame=1, last_seen_frame=50, duration=4.0,
            observation_count=10, keyframe_count=1,
        ),
        Track(
            analysis_job_id=job_a.id, evidence_id=evidence.id,
            track_number=2, class_name="car",
            first_seen_timestamp=6.0, last_seen_timestamp=12.0,
            first_seen_frame=60, last_seen_frame=120, duration=6.0,
            observation_count=8, keyframe_count=1,
        ),
    ]
    # Job B: track 1 (car, 8-15s), track 2 (car, 2-6s), track 3 (bus, 10-20s)
    tracks_b = [
        Track(
            analysis_job_id=job_b.id, evidence_id=evidence.id,
            track_number=1, class_name="car",
            first_seen_timestamp=8.0, last_seen_timestamp=15.0,
            first_seen_frame=80, last_seen_frame=150, duration=7.0,
            observation_count=12, keyframe_count=2,
        ),
        Track(
            analysis_job_id=job_b.id, evidence_id=evidence.id,
            track_number=2, class_name="car",
            first_seen_timestamp=2.0, last_seen_timestamp=6.0,
            first_seen_frame=20, last_seen_frame=60, duration=4.0,
            observation_count=5, keyframe_count=1,
        ),
        Track(
            analysis_job_id=job_b.id, evidence_id=evidence.id,
            track_number=3, class_name="bus",
            first_seen_timestamp=10.0, last_seen_timestamp=20.0,
            first_seen_frame=100, last_seen_frame=200, duration=10.0,
            observation_count=20, keyframe_count=3,
        ),
    ]
    db_session.add_all(tracks_a + tracks_b)
    await db_session.commit()

    # Many detections for job_b track 1 — must NOT inflate the count
    detections = [
        Detection(
            analysis_job_id=job_b.id, evidence_id=evidence.id,
            frame_number=f, timestamp=float(f) * 0.1 + 8.0,
            class_name="car", confidence=0.9,
            bbox_x1=0.1, bbox_y1=0.1, bbox_x2=0.2, bbox_y2=0.2,
            track_id=1,
        )
        for f in range(80, 150, 5)
    ]
    db_session.add_all(detections)
    await db_session.commit()

    return {
        "evidence_id": evidence.id,
        "job_a_id": job_a.id,
        "job_b_id": job_b.id,
    }


@pytest.mark.asyncio
async def test_count_vehicles_scoped_to_latest_job(db_session, seeded_investigation_env):
    """TEST 7 — latest-job isolation. Job B is newest, has 3 vehicle tracks."""
    env = seeded_investigation_env
    result = await InvestigationToolSystem.count_entities(
        db=db_session,
        evidence_id=env["evidence_id"],
        analysis_job_id=env["job_b_id"],
        entity="vehicle",
    )
    assert result["total"] == 3
    assert result["breakdown"].get("car", 0) == 2
    assert result["breakdown"].get("bus", 0) == 1
    # Must NOT merge job A's 2 extra tracks.
    assert result["total"] != 5


@pytest.mark.asyncio
async def test_count_cars_only(db_session, seeded_investigation_env):
    env = seeded_investigation_env
    result = await InvestigationToolSystem.count_entities(
        db=db_session, evidence_id=env["evidence_id"],
        analysis_job_id=env["job_b_id"], entity="car",
    )
    assert result["total"] == 2
    assert "bus" not in result["breakdown"]


@pytest.mark.asyncio
async def test_count_not_inflated_by_detections(db_session, seeded_investigation_env):
    """TEST 8 — many Detection rows must not inflate the count."""
    env = seeded_investigation_env
    result = await InvestigationToolSystem.count_entities(
        db=db_session, evidence_id=env["evidence_id"],
        analysis_job_id=env["job_b_id"], entity="car",
    )
    assert result["total"] == 2  # not len(detections) == 14


@pytest.mark.asyncio
async def test_frame_count_at_timestamp_active_only(db_session, seeded_investigation_env):
    """TEST 3 — only tracks active at t=9s should be counted."""
    env = seeded_investigation_env
    # Active at t=9 in job B: track 1 (8-15s) only; bus (10-20) and car2 (2-6) not active.
    result = await InvestigationToolSystem.count_entities_at_time(
        db=db_session, evidence_id=env["evidence_id"],
        analysis_job_id=env["job_b_id"], entity="vehicle", timestamp=9.0,
    )
    assert result["count"] == 1
    assert result["track_ids"] == [1]


@pytest.mark.asyncio
async def test_verify_count_result_supported(db_session, seeded_investigation_env):
    env = seeded_investigation_env
    result = await InvestigationToolSystem.count_entities(
        db=db_session, evidence_id=env["evidence_id"],
        analysis_job_id=env["job_b_id"], entity="vehicle",
    )
    verification = await EvidenceVerifier.verify_count_result(
        db=db_session, evidence_id=env["evidence_id"], count_result=result,
    )
    assert verification["status"] == "SUPPORTED"
    assert verification["evidence_basis"] == "TRACKS"
    assert verification["checks"]["no_duplicate_track_ids"]
