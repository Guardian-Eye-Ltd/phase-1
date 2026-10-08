"""
Plates and visual attributes in semantic search and investigations.

Before this, search documents read colour/plate fields that do not exist on
Detection rows, so no track document ever mentioned a colour or a plate, and
"what is the vehicle number" had no route to the per-track consensus at all.
"""
import uuid
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.authentication.password import hash_password
from app.models.analysis import AnalysisJob, Detection, JobStage, JobStatus, Track
from app.models.evidence import Evidence, EvidenceStatus
from app.models.observation import EntityType, ObservationSource, ObservationStatus, TrackAttributeAggregate
from app.models.semantic import DocumentType, ForensicDocument
from app.models.user import User
from app.services.agents.investigation_orchestrator import InvestigationOrchestrator
from app.services.agents.investigation_tools import InvestigationToolSystem as Tools
from app.services.pipeline.alpr_service import vote_plate
from app.services.semantic.document_generator import ForensicDocumentGenerator
from app.services.semantic.hybrid_search_engine import HybridSearchEngine
from app.services.semantic.query_parser import QueryIntentParser


def _intent(q):
    return QueryIntentParser.parse_query(q)["structured_intent"]


# ======================================================================
# Query parsing
# ======================================================================

@pytest.mark.parametrize("query", [
    "what is the vehicle number", "show number plates", "number plate",
    "licence plate of the car", "registration of the vehicle",
])
def test_plate_requests_route_to_plate_lookup(query):
    assert _intent(query)["intent"] == "PLATE_LOOKUP"


def test_vehicle_number_of_a_track_is_not_a_count():
    parsed = QueryIntentParser.parse_query("vehicle number of track 14")
    assert parsed["structured_intent"]["intent"] == "PLATE_LOOKUP"
    assert parsed["structured_intent"]["track_id"] == 14
    assert parsed["is_count_query"] is False


@pytest.mark.parametrize("query", ["number of cars", "how many cars are there", "total number of people"])
def test_real_counts_still_count(query):
    assert _intent(query)["intent"] == "COUNT"


def test_words_after_plate_are_not_read_as_a_plate():
    si = _intent("license plate of track 14")
    assert si["intent"] == "PLATE_LOOKUP"
    assert "license_plate_text" not in si["attributes"]


def test_spaced_plate_after_cue_is_joined():
    assert _intent("number plate KA 03 K 961")["attributes"] == {"license_plate_text": "KA03K961"}


def test_plate_request_keeps_colour_filter():
    si = _intent("registration of the red car")
    assert si["intent"] == "PLATE_LOOKUP" and si["attributes"] == {"vehicle_color": "red"} and si["entity"] == "car"


# ======================================================================
# Plate vote
# ======================================================================

def test_one_read_cannot_outvote_another_on_a_character():
    """Real reads from job #2: EA82545 vs CA82545 had been asserted as CA82545."""
    v = vote_plate([("EA82545", 0.385), ("82545", 0.344), ("82545", 0.293), ("CA82545", 0.59)])
    assert v["status"] == "WITHHELD"
    assert "character 1" in v["reason"] and "C vs E" in v["reason"]


def test_two_agreeing_reads_per_character_are_asserted():
    v = vote_plate([("KA37961", 0.55), ("KA3K961", 0.465), ("TA3K961", 0.599)])
    assert v["status"] == "OBSERVED" and v["value"] == "KA3K961" and v["format_complete"] is True


def test_consistent_read_with_impossible_state_code_is_not_complete():
    """Real job: JA3K961 was read consistently, but JA is no Indian state (KA3K961 was read once)."""
    v = vote_plate([("JA37961", 0.55), ("JA3K961", 0.465), ("TA3K961", 0.599)])
    assert v["status"] == "OBSERVED" and v["format_complete"] is False
    assert "'JA' is not an Indian state code" in v["format_issue"]


def test_partial_plate_is_flagged_incomplete():
    v = vote_plate([("82545", 0.6), ("82545", 0.5)])
    assert v["status"] == "OBSERVED" and v["format_complete"] is False


# ======================================================================
# Database-backed behaviour
# ======================================================================

def _agg(job, ev, n, entity, attr, value, conf, status=ObservationStatus.OBSERVED, obs=4, sup=3, **bd):
    return TrackAttributeAggregate(
        analysis_job_id=job, evidence_id=ev, track_number=n, entity_type=entity, attribute=attr,
        value=value, confidence=conf, observation_count=obs, supporting_count=sup,
        dissenting_count=obs - sup, first_observed_at=1.0, last_observed_at=5.0,
        source=ObservationSource.COMBINED, status=status, confidence_breakdown=bd,
    )


@pytest_asyncio.fixture
async def plate_env(db_session):
    """
    car #1   white (asserted), plate KL01AB1234 READ
    car #2   red WITHHELD, plate CA82545 UNCONFIRMED (one-vs-one disagreement)
    car #3   no plate read; largest view 96 px wide on a 1920 px frame
    person #4
    """
    tag = uuid.uuid4().hex[:8]
    user = User(full_name="P", username=f"plate_{tag}", email=f"plate_{tag}@t.local",
                password_hash=hash_password("Testing123!"), role_id=3, is_active=True)
    db_session.add(user)
    await db_session.commit()
    ev = Evidence(original_filename=f"p_{tag}.mp4", stored_filename=f"p_{tag}.mp4", file_path=f"/tmp/p_{tag}.mp4",
                  file_size=1, mime_type="video/mp4", sha256_hash=tag.ljust(64, "f"),
                  status=EvidenceStatus.COMPLETED, uploaded_by=user.id, resolution="1920x1080")
    db_session.add(ev)
    await db_session.commit()
    job = AnalysisJob(evidence_id=ev.id, status=JobStatus.COMPLETED, current_stage=JobStage.FINALIZING,
                      progress=100.0, completed_at=datetime(2026, 6, 1, 9, 0, 0))
    db_session.add(job)
    await db_session.commit()

    for n, cls in ((1, "car"), (2, "car"), (3, "car"), (4, "person")):
        db_session.add(Track(analysis_job_id=job.id, evidence_id=ev.id, track_number=n, class_name=cls,
                             first_seen_timestamp=float(n), last_seen_timestamp=n + 4.0,
                             first_seen_frame=n * 10, last_seen_frame=n * 10 + 40, duration=4.0,
                             observation_count=8, keyframe_count=0))
        width = 0.05 if n == 3 else 0.2
        db_session.add(Detection(analysis_job_id=job.id, evidence_id=ev.id, frame_number=n * 10, timestamp=float(n),
                                 class_name=cls, confidence=0.9, bbox_x1=0.1, bbox_y1=0.1,
                                 bbox_x2=0.1 + width, bbox_y2=0.4, track_id=n))
    V, P = EntityType.VEHICLE, EntityType.PERSON
    db_session.add_all([
        _agg(job.id, ev.id, 1, V, "vehicle_color", "white", 0.7),
        _agg(job.id, ev.id, 1, V, "license_plate_text", "KL01AB1234", 0.6, obs=3, sup=2,
             reads_considered=3, format_complete=True),
        _agg(job.id, ev.id, 2, V, "vehicle_color", "red", 0.2, status=ObservationStatus.WITHHELD),
        _agg(job.id, ev.id, 2, V, "license_plate_text", "CA82545", 0.3, status=ObservationStatus.WITHHELD,
             obs=4, sup=1, withheld_reason="reads disagree on character 1: C vs E", format_complete=True),
        _agg(job.id, ev.id, 4, P, "upper_garment_color", "black", 0.8),
    ])
    await db_session.commit()
    return {"ev": ev.id, "job": job.id}


@pytest.mark.asyncio
async def test_plate_listing_gives_every_vehicle_a_status(db_session, plate_env):
    res = await Tools.list_vehicle_plates(db_session, plate_env["ev"], plate_env["job"])
    by = {v["track_id"]: v for v in res["vehicles"]}
    assert set(by) == {1, 2, 3}                       # the person is not a vehicle
    assert by[1]["plate"]["status"] == "READ" and by[1]["plate"]["text"] == "KL01AB1234"
    assert by[2]["plate"]["status"] == "UNCONFIRMED" and "C vs E" in by[2]["plate"]["reason"]
    assert by[3]["plate"]["status"] == "NOT_READ" and "96px" in by[3]["plate"]["reason"]
    assert res["counts"] == {"READ": 1, "UNCONFIRMED": 1, "NOT_READ": 1}


@pytest.mark.asyncio
async def test_plate_listing_colour_filter_uses_asserted_colour_only(db_session, plate_env):
    white = await Tools.list_vehicle_plates(db_session, plate_env["ev"], plate_env["job"], vehicle_color="white")
    red = await Tools.list_vehicle_plates(db_session, plate_env["ev"], plate_env["job"], vehicle_color="red")
    assert [v["track_id"] for v in white["vehicles"]] == [1]
    assert red["vehicles"] == []                       # car #2's red was withheld


@pytest.mark.asyncio
async def test_track_documents_carry_asserted_attributes_only(db_session, plate_env):
    await ForensicDocumentGenerator.generate_documents_for_job(db_session, plate_env["ev"], plate_env["job"])
    docs = (await db_session.execute(select(ForensicDocument).where(
        ForensicDocument.analysis_job_id == plate_env["job"],
        ForensicDocument.document_type == DocumentType.TRACK))).scalars().all()
    by = {d.track_id: d for d in docs}
    assert "white car" in by[1].content and "KL01AB1234" in by[1].content
    assert by[1].metadata_json["license_plate_number"] == "KL01AB1234"
    assert "CA82545" not in by[2].content and "red" not in by[2].content
    assert "CA82545" not in str(by[2].metadata_json)   # metadata is part of the keyword check
    assert "black" in by[4].content


async def _search(db, env, q):
    return await HybridSearchEngine.execute_search(db, env["ev"], q)


@pytest.mark.asyncio
async def test_semantic_search_answers_what_is_the_vehicle_number(db_session, plate_env):
    r = await _search(db_session, plate_env, "what is the vehicle number")
    assert "KL01AB1234 (Track #1)" in r["answer"]
    titles = [x["title"] for x in r["results"]]
    assert titles[0].endswith("plate KL01AB1234")      # plate text keeps its case
    unconfirmed = next(x for x in r["results"] if x["track_id"] == 2)
    assert unconfirmed["verification_status"] == "UNVERIFIED" and "not confirmed" in unconfirmed["summary"]


@pytest.mark.asyncio
async def test_semantic_search_plate_of_a_track(db_session, plate_env):
    r = await _search(db_session, plate_env, "vehicle number of track 3")
    assert len(r["results"]) == 1 and "too small" in r["results"][0]["summary"]
    r = await _search(db_session, plate_env, "license plate of track 4")
    assert r["results"] == [] and "not a vehicle track" in r["answer"]


@pytest.mark.asyncio
async def test_semantic_search_exact_plate(db_session, plate_env):
    r = await _search(db_session, plate_env, "find KL01AB1234")
    assert r["answer"].startswith("Plate KL01AB1234 was read on car Track #1")


@pytest.mark.asyncio
async def test_unconfirmed_read_is_never_an_identification(db_session, plate_env):
    r = await _search(db_session, plate_env, "find CA82545")
    assert "No vehicle has a plate read exactly as CA82545" in r["answer"]
    assert [x["track_id"] for x in r["results"]] == [2]
    assert r["results"][0]["verification_status"] == "UNVERIFIED"


@pytest.mark.asyncio
async def test_near_miss_plate_is_not_badged_verified(db_session, plate_env):
    r = await _search(db_session, plate_env, "find KL01AB1284")
    assert "No vehicle has a plate read exactly as KL01AB1284" in r["answer"]
    near = r["results"][0]
    assert near["track_id"] == 1 and near["title"].startswith("Possible match")
    assert near["verification_status"] == "PARTIALLY_SUPPORTED"


@pytest.mark.asyncio
async def test_track_question_returns_the_track_document(db_session, plate_env):
    # Indexing writes VLM captions before track documents, so a caption that
    # mentions track 1 has the lower id — it used to be returned instead.
    db_session.add(ForensicDocument(evidence_id=plate_env["ev"], analysis_job_id=plate_env["job"], track_id=1,
                                    document_type=DocumentType.KEYFRAME, title="VLM KF", content="a car on a road",
                                    metadata_json={}))
    await db_session.commit()
    await ForensicDocumentGenerator.generate_documents_for_job(db_session, plate_env["ev"], plate_env["job"])
    r = await _search(db_session, plate_env, "what colour is track 1")
    assert "white car" in r["results"][0]["summary"]


class _FakeCollection:
    def __init__(self, ids):
        self.ids = ids

    def count(self):
        return len(self.ids)

    def query(self, **kw):
        return {"ids": [[f"doc_{i}" for i in self.ids]], "distances": [[0.3] * len(self.ids)]}


@pytest.mark.asyncio
async def test_attribute_miss_falls_back_to_scene_descriptions_only(db_session, plate_env, monkeypatch):
    import numpy as np
    import app.services.semantic.hybrid_search_engine as H

    await ForensicDocumentGenerator.generate_documents_for_job(db_session, plate_env["ev"], plate_env["job"])
    caption = ForensicDocument(evidence_id=plate_env["ev"], analysis_job_id=plate_env["job"],
                               document_type=DocumentType.KEYFRAME, title="VLM KF-9",
                               content="[VLM Visual Observation] a white van next to a red car", metadata_json={})
    db_session.add(caption)
    await db_session.commit()
    track_doc = (await db_session.execute(select(ForensicDocument).where(
        ForensicDocument.analysis_job_id == plate_env["job"], ForensicDocument.track_id == 1,
        ForensicDocument.document_type == DocumentType.TRACK))).scalars().first()

    class _Client:
        def get_collection(self, name):
            return _FakeCollection([track_doc.id, caption.id])

    class _Model:
        def encode(self, text, normalize_embeddings=True):
            return np.zeros(3)

    monkeypatch.setattr(H, "get_chroma_client", lambda: _Client())
    monkeypatch.setattr(H, "get_embedding_model", lambda: _Model())

    r = await _search(db_session, plate_env, "red car")
    assert r["answer"].startswith("No tracked car has a confirmed consensus for colour red")
    assert "scene descriptions, not verified attributes" in r["answer"]
    assert [x["doc_id"] for x in r["results"]] == [caption.id]      # the white car's track doc is gone
    assert r["results"][0]["verification_status"] == "UNVERIFIED"

    # The white *car*'s track document mentions "white", but no bus is white:
    # a track document must not be offered as a "scene description" fallback.
    r = await _search(db_session, plate_env, "find white buses")
    assert r["answer"].startswith("No tracked bus has a confirmed consensus for colour white")
    assert track_doc.id not in [x["doc_id"] for x in r["results"]]


@pytest.mark.asyncio
async def test_semantic_search_finds_white_cars_from_consensus(db_session, plate_env):
    r = await _search(db_session, plate_env, "find white cars")
    assert [x["track_id"] for x in r["results"]] == [1]
    assert r["results"][0]["why_explanation"]["visual_attribute_matches"][0]["formatted"].startswith("✓ WHITE")


@pytest.mark.asyncio
async def test_investigation_answers_plate_questions(db_session, plate_env):
    r = await InvestigationOrchestrator.run_investigation(db_session, plate_env["ev"], "what is the vehicle number")
    assert r["intent"]["type"] == "PLATE_LOOKUP"
    assert r["findings"] and r["findings"][0]["title"].endswith("plate KL01AB1234")
    assert "| #1 | car | white | KL01AB1234 | `READ`" in r["report_markdown"]


@pytest.mark.asyncio
async def test_investigation_attribute_findings_render_as_cards(db_session, plate_env):
    r = await InvestigationOrchestrator.run_investigation(db_session, plate_env["ev"], "find white cars")
    card = r["findings"][0]
    assert card["track_id"] == 1 and card["title"] and card["summary"] and card["verification_status"] == "VERIFIED"
    assert r["result"]["matches"][0]["track_id"] == 1   # raw matches still available
