"""
Slice B — structured attribute search over temporally aggregated observations.

Verifies that attribute queries hit TrackAttributeAggregate (consensus), respect
analysis-job isolation, exclude WITHHELD aggregates, and never fuzzy-match a
licence plate unless explicitly asked.
"""
import pytest
import pytest_asyncio
import uuid
from datetime import datetime

from app.services.agents.investigation_tools import InvestigationToolSystem as Tools
from app.models.observation import (
    VisualAttributeObservation, TrackAttributeAggregate,
    ObservationStatus, ObservationSource, EntityType,
)
from app.models.evidence import Evidence, EvidenceStatus
from app.models.analysis import AnalysisJob, JobStatus, JobStage, Track
from app.models.user import User
from app.authentication.password import hash_password


def _track(job_id, ev_id, n, cls, t0=1.0, t1=5.0):
    return Track(
        analysis_job_id=job_id, evidence_id=ev_id, track_number=n, class_name=cls,
        first_seen_timestamp=t0, last_seen_timestamp=t1,
        first_seen_frame=int(t0 * 10), last_seen_frame=int(t1 * 10),
        duration=t1 - t0, observation_count=10, keyframe_count=1,
    )


def _agg(job_id, ev_id, n, entity, attr, value, conf,
         status=ObservationStatus.OBSERVED, obs=5, sup=5):
    return TrackAttributeAggregate(
        analysis_job_id=job_id, evidence_id=ev_id, track_number=n,
        entity_type=entity, attribute=attr, value=value, confidence=conf,
        observation_count=obs, supporting_count=sup, dissenting_count=obs - sup,
        first_observed_at=1.0, last_observed_at=5.0,
        source=ObservationSource.COMBINED, status=status,
        confidence_breakdown={"model": conf, "temporal": 1.0, "coverage": 1.0,
                              "consensus_ratio": sup / obs, "support": 1.0,
                              "capability": 0.8},
    )


@pytest_asyncio.fixture
async def search_env(db_session):
    """
    Job A (older) : trk1 person black shirt, trk2 car red
    Job B (newer) : trk1 person white shirt, trk2 car blue,
                    trk3 car red but WITHHELD, trk4 car plate KL01AB1234
    """
    tag = uuid.uuid4().hex[:8]
    user = User(full_name="S", username=f"srch_{tag}", email=f"srch_{tag}@t.local",
                password_hash=hash_password("Testing123!"), role_id=3, is_active=True)
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    ev = Evidence(original_filename=f"s_{tag}.mp4", stored_filename=f"s_{tag}_x.mp4",
                  file_path=f"/tmp/s_{tag}.mp4", file_size=1, mime_type="video/mp4",
                  sha256_hash=tag.ljust(64, "c"), status=EvidenceStatus.COMPLETED,
                  uploaded_by=user.id)
    db_session.add(ev)
    await db_session.commit()
    await db_session.refresh(ev)

    ja = AnalysisJob(evidence_id=ev.id, status=JobStatus.COMPLETED,
                     current_stage=JobStage.FINALIZING, progress=100.0,
                     completed_at=datetime(2026, 5, 1, 9, 0, 0))
    jb = AnalysisJob(evidence_id=ev.id, status=JobStatus.COMPLETED,
                     current_stage=JobStage.FINALIZING, progress=100.0,
                     completed_at=datetime(2026, 5, 2, 9, 0, 0))
    db_session.add_all([ja, jb])
    await db_session.commit()
    await db_session.refresh(ja)
    await db_session.refresh(jb)

    db_session.add_all([
        _track(ja.id, ev.id, 1, "person"), _track(ja.id, ev.id, 2, "car"),
        _track(jb.id, ev.id, 1, "person"), _track(jb.id, ev.id, 2, "car"),
        _track(jb.id, ev.id, 3, "car"), _track(jb.id, ev.id, 4, "car"),
    ])
    db_session.add_all([
        _agg(ja.id, ev.id, 1, EntityType.PERSON, "upper_garment_color", "black", 0.80),
        _agg(ja.id, ev.id, 2, EntityType.VEHICLE, "vehicle_color", "red", 0.80),

        _agg(jb.id, ev.id, 1, EntityType.PERSON, "upper_garment_color", "white", 0.75),
        _agg(jb.id, ev.id, 1, EntityType.PERSON, "carries_bag", "true", 0.70),
        _agg(jb.id, ev.id, 2, EntityType.VEHICLE, "vehicle_color", "blue", 0.72),
        _agg(jb.id, ev.id, 3, EntityType.VEHICLE, "vehicle_color", "red", 0.30,
             status=ObservationStatus.WITHHELD, obs=4, sup=2),
        _agg(jb.id, ev.id, 4, EntityType.VEHICLE, "license_plate_text",
             "KL01AB1234", 0.68),
    ])
    db_session.add_all([
        VisualAttributeObservation(
            evidence_id=ev.id, analysis_job_id=jb.id, track_number=1,
            frame_number=f, timestamp=float(f) / 10.0,
            entity_type=EntityType.PERSON, attribute="upper_garment_color",
            value="white", confidence=0.75, source=ObservationSource.CLIP_ZERO_SHOT,
            model_name="clip", model_version="1.0",
            status=ObservationStatus.OBSERVED,
        )
        for f in (10, 20, 30)
    ])
    await db_session.commit()
    return {"evidence_id": ev.id, "job_a": ja.id, "job_b": jb.id}


# ======================================================================
# Structured attribute matching
# ======================================================================

@pytest.mark.asyncio
async def test_find_person_by_garment_colour(db_session, search_env):
    e = search_env
    res = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"upper_garment_color": "white"}, entity="person",
    )
    assert len(res) == 1
    assert res[0]["track_id"] == 1
    assert res[0]["attributes"]["upper_garment_color"]["value"] == "white"


@pytest.mark.asyncio
async def test_attribute_search_is_job_scoped(db_session, search_env):
    """Job A's black shirt must not leak into a Job B query."""
    e = search_env
    res = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"upper_garment_color": "black"}, entity="person",
    )
    assert res == []

    res_a = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_a"],
        attributes={"upper_garment_color": "black"}, entity="person",
    )
    assert len(res_a) == 1


@pytest.mark.asyncio
async def test_withheld_aggregates_are_excluded_by_default(db_session, search_env):
    """Track 3's red is WITHHELD for contested consensus — must not match."""
    e = search_env
    res = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"vehicle_color": "red"},
    )
    assert res == []


@pytest.mark.asyncio
async def test_specific_vehicle_class_is_respected(db_session, search_env):
    """A bus (e.g. a train the detector called 'bus') must not answer 'find the blue car'."""
    e = search_env
    db_session.add_all([
        _track(e["job_b"], e["evidence_id"], 7, "bus"),
        _agg(e["job_b"], e["evidence_id"], 7, EntityType.VEHICLE, "vehicle_color", "blue", 0.9),
    ])
    await db_session.commit()
    cars = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"vehicle_color": "blue"}, entity="car")
    vehicles = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"vehicle_color": "blue"}, entity="vehicle")
    assert [m["track_id"] for m in cars] == [2]
    assert sorted(m["track_id"] for m in vehicles) == [2, 7]


@pytest.mark.asyncio
async def test_contradicted_aggregates_never_match(db_session, search_env):
    e = search_env
    db_session.add(_agg(e["job_b"], e["evidence_id"], 9, EntityType.PERSON, "upper_garment_color",
                        "green", 0.9, status=ObservationStatus.CONTRADICTED))
    await db_session.commit()
    res = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"upper_garment_color": "green"}, entity="person",
    )
    assert res == []


@pytest.mark.asyncio
async def test_withheld_can_be_included_explicitly(db_session, search_env):
    e = search_env
    res = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"vehicle_color": "red"}, include_withheld=True,
    )
    assert len(res) == 1
    assert res[0]["attributes"]["vehicle_color"]["status"] == "WITHHELD"


@pytest.mark.asyncio
async def test_multiple_attributes_require_all_to_match(db_session, search_env):
    e = search_env
    both = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"upper_garment_color": "white", "carries_bag": "true"},
        entity="person",
    )
    assert len(both) == 1 and both[0]["track_id"] == 1

    impossible = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"upper_garment_color": "black", "carries_bag": "true"},
        entity="person",
    )
    assert impossible == []


@pytest.mark.asyncio
async def test_entity_filter_prevents_cross_type_matches(db_session, search_env):
    """A person query must never return a vehicle track."""
    e = search_env
    res = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"vehicle_color": "blue"}, entity="person",
    )
    assert res == []


@pytest.mark.asyncio
async def test_results_include_supporting_frames_for_why_explanation(db_session, search_env):
    e = search_env
    res = await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        attributes={"upper_garment_color": "white"}, entity="person",
    )
    frames = res[0]["attributes"]["upper_garment_color"]["supporting_frames"]
    assert frames == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_empty_attributes_returns_nothing(db_session, search_env):
    e = search_env
    assert await Tools.find_tracks_by_attributes(
        db=db_session, evidence_id=e["evidence_id"],
        analysis_job_id=e["job_b"], attributes={},
    ) == []


# ======================================================================
# Licence plate lookup
# ======================================================================

@pytest.mark.asyncio
async def test_exact_plate_match(db_session, search_env):
    e = search_env
    res = await Tools.find_vehicle_by_plate(
        db=db_session, evidence_id=e["evidence_id"],
        analysis_job_id=e["job_b"], plate_text="KL01AB1234",
    )
    assert len(res) == 1
    assert res[0]["track_id"] == 4
    assert res[0]["match_type"] == "EXACT_PLATE_MATCH"


@pytest.mark.asyncio
async def test_plate_match_ignores_separators_and_case(db_session, search_env):
    e = search_env
    res = await Tools.find_vehicle_by_plate(
        db=db_session, evidence_id=e["evidence_id"],
        analysis_job_id=e["job_b"], plate_text="kl01-ab 1234",
    )
    assert len(res) == 1


@pytest.mark.asyncio
async def test_near_miss_plate_not_matched_without_fuzzy(db_session, search_env):
    """One wrong character must not silently identify a vehicle."""
    e = search_env
    res = await Tools.find_vehicle_by_plate(
        db=db_session, evidence_id=e["evidence_id"],
        analysis_job_id=e["job_b"], plate_text="KL01A81234",
    )
    assert res == []


@pytest.mark.asyncio
async def test_fuzzy_plate_match_is_labelled_possible(db_session, search_env):
    e = search_env
    res = await Tools.find_vehicle_by_plate(
        db=db_session, evidence_id=e["evidence_id"], analysis_job_id=e["job_b"],
        plate_text="KL01A81234", allow_fuzzy=True,
    )
    assert len(res) == 1
    assert res[0]["match_type"] == "POSSIBLE_PLATE_MATCH"
    assert res[0]["character_differences"] == 1


@pytest.mark.asyncio
async def test_unknown_plate_returns_no_evidence(db_session, search_env):
    e = search_env
    res = await Tools.find_vehicle_by_plate(
        db=db_session, evidence_id=e["evidence_id"],
        analysis_job_id=e["job_b"], plate_text="ZZ99ZZ9999", allow_fuzzy=True,
    )
    assert res == []


# ======================================================================
# Show-evidence accessor
# ======================================================================

@pytest.mark.asyncio
async def test_get_visual_attributes_returns_all_for_track(db_session, search_env):
    e = search_env
    res = await Tools.get_visual_attributes(
        db=db_session, evidence_id=e["evidence_id"],
        analysis_job_id=e["job_b"], track_number=1,
    )
    assert res["track_id"] == 1
    assert set(res["attributes"]) == {"upper_garment_color", "carries_bag"}
    assert res["attributes"]["upper_garment_color"]["confidence_breakdown"]["model"] == 0.75
