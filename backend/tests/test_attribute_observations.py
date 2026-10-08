"""
Tests for Slice A — canonical attribute observations, temporal aggregation,
confidence decomposition, evidence status, and capability honesty.
"""
import pytest
import pytest_asyncio
import uuid
from datetime import datetime

from app.services.pipeline.attribute_aggregator import (
    AttributeAggregator,
    compute_confidence_breakdown,
    overall_confidence,
    MIN_AGGREGATE_CONFIDENCE,
)
from app.services.pipeline.capability_registry import (
    get_capability, is_available, is_usable, unavailable_result,
    capability_report, CapabilityState,
)
from app.models.observation import (
    VisualAttributeObservation, TrackAttributeAggregate,
    ObservationStatus, ObservationSource, EntityType,
)
from app.models.evidence import Evidence, EvidenceStatus
from app.models.analysis import AnalysisJob, JobStatus, JobStage
from app.models.user import User
from app.authentication.password import hash_password


def _det(track_id, cls, ts, frame, **attrs):
    base = {
        "track_id": track_id, "class_name": cls,
        "timestamp": ts, "frame_number": frame,
    }
    base.update(attrs)
    return base


# ======================================================================
# Observation extraction
# ======================================================================

def test_extract_observations_skips_untracked_detections():
    dets = [_det(None, "person", 1.0, 10, upper_garment_color="black",
                 upper_garment_color_confidence=0.9)]
    assert AttributeAggregator.extract_observations(dets) == []


def test_extract_observations_skips_unknown_values():
    dets = [_det(1, "person", 1.0, 10,
                 upper_garment_color="unknown",
                 upper_garment_color_confidence=0.2)]
    obs = AttributeAggregator.extract_observations(dets)
    assert all(o["attribute"] != "upper_garment_color" for o in obs)


def test_extract_observations_captures_confidence():
    dets = [_det(1, "person", 1.0, 10,
                 upper_garment_color="black",
                 upper_garment_color_confidence=0.88)]
    obs = AttributeAggregator.extract_observations(dets)
    shirt = [o for o in obs if o["attribute"] == "upper_garment_color"][0]
    assert shirt["value"] == "black"
    assert shirt["confidence"] == 0.88
    assert shirt["entity_type"] == "PERSON"


def test_missing_confidence_defaults_low_not_certain():
    """A classifier result without a confidence must not be treated as certain."""
    dets = [_det(1, "person", 1.0, 10, upper_garment_color="black")]
    obs = AttributeAggregator.extract_observations(dets)
    shirt = [o for o in obs if o["attribute"] == "upper_garment_color"][0]
    assert shirt["confidence"] == 0.30


def test_vehicle_observations_use_vehicle_entity_type():
    dets = [_det(42, "car", 2.0, 20,
                 vehicle_color="red", vehicle_color_confidence=0.91)]
    obs = AttributeAggregator.extract_observations(dets)
    color = [o for o in obs if o["attribute"] == "vehicle_color"][0]
    assert color["entity_type"] == "VEHICLE"
    assert color["value"] == "red"


def test_plates_are_not_read_per_detection():
    """Plates come from the per-track ALPR stage, never from detection dicts."""
    dets = [_det(42, "car", 2.0, 20, license_plate_number="KL01AB1234",
                 license_plate_confidence=0.9)]
    obs = AttributeAggregator.extract_observations(dets)
    assert not any(o["attribute"] == "license_plate_text" for o in obs)


# ======================================================================
# Temporal aggregation (Phase 10)
# ======================================================================

def test_consistent_observations_produce_high_confidence_consensus():
    dets = [
        _det(17, "person", t, f, upper_garment_color="black",
             upper_garment_color_confidence=c)
        for t, f, c in [(10.0, 100, 0.71), (10.5, 105, 0.83),
                        (11.0, 110, 0.80), (11.5, 115, 0.86)]
    ]
    obs = AttributeAggregator.extract_observations(dets)
    aggs = AttributeAggregator.aggregate(
        obs, track_observation_counts={17: 4},
        capability_states={"person_garment_color": "DEGRADED"},
    )
    shirt = [a for a in aggs if a["attribute"] == "upper_garment_color"][0]
    assert shirt["value"] == "black"
    assert shirt["supporting_count"] == 4
    assert shirt["dissenting_count"] == 0
    assert shirt["status"] == "OBSERVED"
    assert shirt["confidence"] >= MIN_AGGREGATE_CONFIDENCE


def test_majority_vote_beats_minority_outlier():
    dets = [
        _det(17, "person", 10.0, 100, upper_garment_color="black",
             upper_garment_color_confidence=0.85),
        _det(17, "person", 10.5, 105, upper_garment_color="black",
             upper_garment_color_confidence=0.82),
        _det(17, "person", 11.0, 110, upper_garment_color="black",
             upper_garment_color_confidence=0.80),
        _det(17, "person", 11.5, 115, upper_garment_color="blue",
             upper_garment_color_confidence=0.55),
    ]
    obs = AttributeAggregator.extract_observations(dets)
    aggs = AttributeAggregator.aggregate(obs, track_observation_counts={17: 4})
    shirt = [a for a in aggs if a["attribute"] == "upper_garment_color"][0]
    assert shirt["value"] == "black"
    assert shirt["supporting_count"] == 3
    assert shirt["dissenting_count"] == 1


def test_contested_consensus_is_withheld_not_asserted():
    """50/50 split must not be presented as a finding."""
    dets = [
        _det(17, "person", 10.0, 100, upper_garment_color="black",
             upper_garment_color_confidence=0.60),
        _det(17, "person", 10.5, 105, upper_garment_color="blue",
             upper_garment_color_confidence=0.59),
    ]
    obs = AttributeAggregator.extract_observations(dets)
    aggs = AttributeAggregator.aggregate(obs, track_observation_counts={17: 2})
    shirt = [a for a in aggs if a["attribute"] == "upper_garment_color"][0]
    assert shirt["status"] == "WITHHELD"


def test_single_weak_observation_is_withheld():
    dets = [_det(17, "person", 10.0, 100, upper_garment_color="black",
                 upper_garment_color_confidence=0.31)]
    obs = AttributeAggregator.extract_observations(dets)
    aggs = AttributeAggregator.aggregate(obs, track_observation_counts={17: 1})
    shirt = [a for a in aggs if a["attribute"] == "upper_garment_color"][0]
    assert shirt["status"] == "WITHHELD"


def test_aggregation_is_per_track_not_global():
    dets = [
        _det(1, "person", 10.0, 100, upper_garment_color="black",
             upper_garment_color_confidence=0.9),
        _det(2, "person", 10.0, 100, upper_garment_color="white",
             upper_garment_color_confidence=0.9),
    ]
    obs = AttributeAggregator.extract_observations(dets)
    aggs = AttributeAggregator.aggregate(
        obs, track_observation_counts={1: 1, 2: 1})
    by_track = {a["track_number"]: a["value"]
                for a in aggs if a["attribute"] == "upper_garment_color"}
    assert by_track == {1: "black", 2: "white"}


def test_conflicting_ocr_reads_lower_consensus():
    """Disagreeing plate reads must not assert a plate."""
    obs = [
        {"track_number": 42, "entity_type": "VEHICLE", "attribute": "license_plate_text",
         "value": v, "confidence": c, "frame_number": f, "timestamp": f / 10.0, "source": "ALPR_OCR"}
        for v, c, f in (("KL01AB1234", 0.45, 10), ("KL01A81234", 0.44, 15))
    ]
    aggs = AttributeAggregator.aggregate(obs, track_observation_counts={42: 2})
    plate = [a for a in aggs if a["attribute"] == "license_plate_text"][0]
    assert plate["status"] == "WITHHELD"


# ======================================================================
# Confidence decomposition (Phase 20)
# ======================================================================

def test_confidence_breakdown_has_all_axes():
    b = compute_confidence_breakdown(
        mean_model_confidence=0.85, consensus_ratio=1.0,
        observation_count=5, track_observation_count=5,
        capability_state="AVAILABLE",
    )
    assert set(b) == {"model", "temporal", "coverage", "consensus_ratio",
                      "support", "capability"}


def test_thin_evidence_does_not_inherit_full_agreement_credit():
    """One observation trivially agrees with itself; it must not score like five."""
    one = compute_confidence_breakdown(
        mean_model_confidence=0.80, consensus_ratio=1.0,
        observation_count=1, track_observation_count=1,
        capability_state="DEGRADED",
    )
    five = compute_confidence_breakdown(
        mean_model_confidence=0.80, consensus_ratio=1.0,
        observation_count=5, track_observation_count=5,
        capability_state="DEGRADED",
    )
    assert one["support"] < five["support"]
    assert overall_confidence(one) < overall_confidence(five)


def test_degraded_capability_reduces_confidence():
    kwargs = dict(mean_model_confidence=0.85, consensus_ratio=1.0,
                  observation_count=5, track_observation_count=5)
    strong = overall_confidence(compute_confidence_breakdown(
        **kwargs, capability_state="AVAILABLE"))
    degraded = overall_confidence(compute_confidence_breakdown(
        **kwargs, capability_state="DEGRADED"))
    assert degraded < strong


def test_unavailable_capability_zeroes_confidence():
    score = overall_confidence(compute_confidence_breakdown(
        mean_model_confidence=0.99, consensus_ratio=1.0,
        observation_count=10, track_observation_count=10,
        capability_state="NOT_AVAILABLE",
    ))
    assert score == 0.0


def test_more_agreeing_frames_raises_confidence():
    def score(n):
        return overall_confidence(compute_confidence_breakdown(
            mean_model_confidence=0.80, consensus_ratio=1.0,
            observation_count=n, track_observation_count=n,
            capability_state="DEGRADED",
        ))
    assert score(1) < score(3) < score(5)


# ======================================================================
# Capability honesty (anti-fabrication)
# ======================================================================

def test_vehicle_make_is_not_available():
    assert not is_available("vehicle_make")
    assert not is_usable("vehicle_make")
    assert get_capability("vehicle_make")["state"] == CapabilityState.NOT_AVAILABLE


def test_fire_smoke_is_not_available():
    assert get_capability("fire_smoke_detection")["state"] == CapabilityState.NOT_AVAILABLE


def test_calibrated_speed_is_not_available_without_calibration():
    assert get_capability("calibrated_speed_kmh")["state"] == CapabilityState.NOT_AVAILABLE


def test_unavailable_result_returns_null_not_a_guess():
    r = unavailable_result("vehicle_make", track_id=42)
    assert r["value"] is None
    assert r["confidence"] == 0.0
    assert r["status"] == "WITHHELD"
    assert r["reason"]
    assert r["track_id"] == 42


def test_degraded_capabilities_are_usable_but_flagged():
    assert is_usable("person_garment_color")
    assert not is_available("person_garment_color")


def test_capability_report_covers_every_capability():
    rep = capability_report()
    for key in ("person_garment_color", "vehicle_color", "license_plate_text",
                "vehicle_make", "fire_smoke_detection", "calibrated_speed_kmh"):
        assert key in rep
        assert rep[key]["reason"]


def test_plate_capability_tracks_installed_ocr_engines():
    """
    Declared state must be downgraded when no OCR engine is installed —
    a capability is only as good as the packages next to it.
    """
    import importlib.util
    has_ocr = any(
        importlib.util.find_spec(m) is not None
        for m in ("paddleocr", "easyocr")
    )
    state = get_capability("license_plate_text")["state"]
    if has_ocr:
        assert state in (CapabilityState.AVAILABLE, CapabilityState.DEGRADED)
    else:
        assert state == CapabilityState.NOT_AVAILABLE


# ======================================================================
# Evidence status ladder (Phase 34)
# ======================================================================

def test_observation_status_has_no_confirmed_member():
    """The system must never declare a confirmed crime."""
    members = {m.value for m in ObservationStatus}
    assert "CONFIRMED" not in members
    assert members == {
        "OBSERVED", "CANDIDATE", "SUPPORTED",
        "VERIFIED", "WITHHELD", "CONTRADICTED",
    }


# ======================================================================
# DB persistence round-trip
# ======================================================================

@pytest_asyncio.fixture
async def attr_env(db_session):
    tag = uuid.uuid4().hex[:8]
    user = User(full_name="T", username=f"attr_{tag}", email=f"attr_{tag}@t.local",
                password_hash=hash_password("Testing123!"), role_id=3, is_active=True)
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    ev = Evidence(original_filename=f"a_{tag}.mp4", stored_filename=f"a_{tag}_s.mp4",
                  file_path=f"/tmp/a_{tag}.mp4", file_size=1, mime_type="video/mp4",
                  sha256_hash=tag.ljust(64, "b"), status=EvidenceStatus.COMPLETED,
                  uploaded_by=user.id)
    db_session.add(ev)
    await db_session.commit()
    await db_session.refresh(ev)

    job = AnalysisJob(evidence_id=ev.id, status=JobStatus.COMPLETED,
                      current_stage=JobStage.FINALIZING, progress=100.0,
                      completed_at=datetime(2026, 5, 1, 10, 0, 0))
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    return {"evidence_id": ev.id, "job_id": job.id}


@pytest.mark.asyncio
async def test_observation_persists_with_full_provenance(db_session, attr_env):
    obs = VisualAttributeObservation(
        evidence_id=attr_env["evidence_id"], analysis_job_id=attr_env["job_id"],
        track_number=17, frame_number=375, timestamp=12.5,
        entity_type=EntityType.PERSON, attribute="upper_garment_color",
        value="black", confidence=0.84,
        source=ObservationSource.CLIP_ZERO_SHOT,
        model_name="openai/clip-vit-base-patch32", model_version="1.0",
        status=ObservationStatus.OBSERVED,
    )
    db_session.add(obs)
    await db_session.commit()
    await db_session.refresh(obs)

    assert obs.id is not None
    assert obs.analysis_job_id == attr_env["job_id"]
    assert obs.model_name == "openai/clip-vit-base-patch32"
    assert obs.status == ObservationStatus.OBSERVED


@pytest.mark.asyncio
async def test_aggregate_persists_confidence_breakdown(db_session, attr_env):
    agg = TrackAttributeAggregate(
        evidence_id=attr_env["evidence_id"], analysis_job_id=attr_env["job_id"],
        track_number=17, entity_type=EntityType.PERSON,
        attribute="upper_garment_color", value="black", confidence=0.81,
        observation_count=5, supporting_count=4, dissenting_count=1,
        first_observed_at=10.0, last_observed_at=11.5,
        source=ObservationSource.COMBINED, status=ObservationStatus.OBSERVED,
        confidence_breakdown={"model": 0.82, "temporal": 0.8, "coverage": 1.0,
                              "consensus_ratio": 0.8, "capability": 0.8},
    )
    db_session.add(agg)
    await db_session.commit()
    await db_session.refresh(agg)

    assert agg.confidence_breakdown["model"] == 0.82
    assert agg.supporting_count == 4
    assert agg.dissenting_count == 1
