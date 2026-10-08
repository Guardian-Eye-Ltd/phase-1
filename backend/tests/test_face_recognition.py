"""
Face recognition: encryption, quality gating, frame selection, extraction,
search semantics, job isolation, privacy and access control.

Uses synthetic embeddings and a fake detector; the real model is exercised by
the separate end-to-end verification on sample images.
"""
import json
import uuid
from datetime import datetime

import numpy as np
import pytest
from sqlalchemy import select

import app.api.routes.faces as faces_routes
from app.authentication.jwt import create_access_token
from app.authentication.password import hash_password
from app.core.config import settings
from app.models.analysis import AnalysisJob, JobStage, JobStatus, Track
from app.models.audit import AuditLog
from app.models.evidence import Evidence, EvidenceStatus
from app.models.face import FaceObservation, FaceQuality
from app.models.user import User
from app.services.face import crypto
from app.services.face.crypto import decrypt_embedding, encrypt_embedding
from app.services.face.engine import DetectedFace, FaceEngine, assess_quality
from app.services.face.search_service import FaceSearchError, embed_query_photo, search_faces
from app.services.pipeline.face_extractor import extract_faces, select_frames_for_track

DIM = 512


def unit(*idx_weights):
    v = np.zeros(DIM, dtype=np.float32)
    for i, w in idx_weights:
        v[i] = w
    return v / np.linalg.norm(v)


def textured(h, w, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


# ======================================================================
# Encryption at rest
# ======================================================================

def test_embedding_round_trip_and_ciphertext_hides_values():
    e = unit((0, 1.0), (5, 0.5))
    token = encrypt_embedding(e)
    assert e.astype("<f4").tobytes() not in token
    assert np.allclose(decrypt_embedding(token), e)


def test_wrong_key_yields_none_not_garbage(monkeypatch):
    token = encrypt_embedding(unit((0, 1.0)))
    crypto._fernet.cache_clear()
    monkeypatch.setattr(settings, "FACE_EMBEDDING_KEY", "x" * 43 + "=")
    try:
        assert decrypt_embedding(token) is None
    finally:
        crypto._fernet.cache_clear()


# ======================================================================
# Quality gate
# ======================================================================

def test_small_face_is_low_quality():
    q = assess_quality(textured(20, 20), 20, 20, 0.9)
    assert q.status == FaceQuality.LOW_QUALITY
    assert any("too small" in r for r in q.reasons)


def test_low_detection_confidence_is_low_quality():
    q = assess_quality(textured(80, 80), 80, 80, 0.3)
    assert q.status == FaceQuality.LOW_QUALITY
    assert any("confidence" in r for r in q.reasons)


def test_blurry_face_is_low_quality():
    flat = np.full((80, 80, 3), 128, dtype=np.uint8)
    q = assess_quality(flat, 80, 80, 0.9)
    assert q.status == FaceQuality.LOW_QUALITY
    assert any("blurry" in r for r in q.reasons)


def test_clear_face_is_usable():
    q = assess_quality(textured(80, 80), 80, 80, 0.9)
    assert q.status == FaceQuality.USABLE and q.reasons == []


# ======================================================================
# Frame selection & extraction
# ======================================================================

def _pdet(fn, ts, size, track=1):
    return {"class_name": "person", "track_id": track, "frame_number": fn, "timestamp": ts,
            "bbox_x1": 0.1, "bbox_y1": 0.1, "bbox_x2": 0.1 + size, "bbox_y2": 0.1 + 2 * size}


def test_frame_selection_prefers_large_and_spreads_in_time():
    dets = [_pdet(1, 1.0, 0.30), _pdet(2, 1.1, 0.29), _pdet(3, 2.0, 0.20), _pdet(4, 3.0, 0.10)]
    chosen = select_frames_for_track(dets, 3)
    assert [d["frame_number"] for d in chosen] == [1, 3, 4]  # frame 2 is too close to frame 1


def test_extraction_encrypts_usable_faces_only(tmp_path):
    frame = textured(400, 400)
    sampled = [(10, 1.0, frame), (20, 2.0, frame)]
    dets = [
        {**_pdet(10, 1.0, 0.4, track=1), "bbox_x1": 0.1, "bbox_x2": 0.5, "bbox_y1": 0.1, "bbox_y2": 0.9},
        {**_pdet(20, 2.0, 0.4, track=2), "bbox_x1": 0.1, "bbox_x2": 0.5, "bbox_y1": 0.1, "bbox_y2": 0.9},
        {"class_name": "car", "track_id": 3, "frame_number": 10, "timestamp": 1.0,
         "bbox_x1": 0.6, "bbox_y1": 0.6, "bbox_x2": 0.9, "bbox_y2": 0.9},
    ]
    sizes = {1: 80, 2: 12}  # track 2's face is too small to compare
    calls = []

    def fake_detect(region):
        track = 1 if len(calls) == 0 else 2
        calls.append(region.shape)
        s = sizes[track]
        cx = region.shape[1] / 2
        return [DetectedFace(bbox=(cx - s / 2, 10, cx + s / 2, 10 + s), det_score=0.9,
                             embedding=unit((track, 1.0)))]

    obs = extract_faces(sampled, dets, evidence_id=7, analysis_job_id=3,
                        derived_dir=str(tmp_path), detect_fn=fake_detect)
    assert len(calls) == 2  # the car track is never sent to the face model
    by_track = {o["track_number"]: o for o in obs}
    assert by_track[1]["quality_status"] == FaceQuality.USABLE
    assert np.allclose(decrypt_embedding(by_track[1]["embedding_encrypted"]), unit((1, 1.0)))
    assert by_track[2]["quality_status"] == FaceQuality.LOW_QUALITY
    assert by_track[2]["embedding_encrypted"] is None
    assert (tmp_path / "faces" / "7" / by_track[1]["crop_filename"]).exists()
    assert "_job3_" in by_track[1]["crop_filename"]


# ======================================================================
# Query photo handling
# ======================================================================

def _png(h=200, w=200):
    import cv2
    ok, buf = cv2.imencode(".png", textured(h, w))
    return buf.tobytes()


def test_query_with_no_face_is_rejected():
    with pytest.raises(FaceSearchError, match="No face"):
        embed_query_photo(_png(), detect_fn=lambda img: [])


def test_query_with_two_clear_faces_is_rejected():
    two = [DetectedFace((10, 10, 90, 90), 0.9, unit((0, 1.0))),
           DetectedFace((100, 10, 180, 90), 0.9, unit((1, 1.0)))]
    with pytest.raises(FaceSearchError, match="2 clear faces"):
        embed_query_photo(_png(), detect_fn=lambda img: two)


def test_query_with_only_tiny_face_is_rejected():
    tiny = [DetectedFace((10, 10, 20, 20), 0.9, unit((0, 1.0)))]
    with pytest.raises(FaceSearchError, match="not clear enough"):
        embed_query_photo(_png(), detect_fn=lambda img: tiny)


def test_non_image_upload_is_rejected():
    with pytest.raises(FaceSearchError, match="not a readable image"):
        embed_query_photo(b"definitely not an image", detect_fn=lambda img: [])


# ======================================================================
# Search semantics (DB)
# ======================================================================

async def _seed(db, face_stage="COMPLETED"):
    tag = uuid.uuid4().hex[:8]
    u = User(full_name="F", username=f"f_{tag}", email=f"{tag}@t.local",
             password_hash=hash_password("Pw123456!"), role_id=3, is_active=True)
    db.add(u)
    await db.commit()
    await db.refresh(u)
    ev = Evidence(original_filename=f"{tag}.mp4", stored_filename=f"{tag}_s.mp4",
                  file_path=f"/tmp/{tag}.mp4", file_size=1, mime_type="video/mp4",
                  sha256_hash=tag.ljust(64, "e"), status=EvidenceStatus.COMPLETED, uploaded_by=u.id)
    db.add(ev)
    await db.commit()
    await db.refresh(ev)
    jobs = []
    for hour in (1, 2):
        j = AnalysisJob(evidence_id=ev.id, status=JobStatus.COMPLETED, current_stage=JobStage.FINALIZING,
                        progress=100.0, completed_at=datetime(2026, 5, 1, hour), face_stage=face_stage)
        db.add(j)
        await db.commit()
        await db.refresh(j)
        jobs.append(j)
    return u, ev, jobs


def _track(job, n):
    return Track(analysis_job_id=job.id, evidence_id=job.evidence_id, track_number=n, class_name="person",
                 first_seen_timestamp=1.0, last_seen_timestamp=5.0, first_seen_frame=10,
                 last_seen_frame=50, duration=4.0, observation_count=5, keyframe_count=1)


def _face(job, track, fn, emb=None, quality=FaceQuality.USABLE):
    return FaceObservation(
        evidence_id=job.evidence_id, analysis_job_id=job.id, track_number=track, frame_number=fn,
        timestamp=fn / 10.0, bbox_x1=0.1, bbox_y1=0.1, bbox_x2=0.2, bbox_y2=0.2,
        face_width_px=80, face_height_px=90, det_score=0.9, sharpness=50.0,
        quality_status=quality, quality_reasons=[] if quality == FaceQuality.USABLE else ["too small"],
        embedding_encrypted=encrypt_embedding(emb) if emb is not None else None,
        embedding_model="test", embedding_dim=DIM,
        crop_filename=f"face_ev{job.evidence_id}_job{job.id}_trk{track}_fn{fn}.jpg", crop_sha256="0" * 64,
    )


@pytest.mark.asyncio
async def test_search_returns_possible_match_for_similar_face_only(db_session):
    _, ev, (_, job) = await _seed(db_session)
    db_session.add_all([
        _track(job, 1), _track(job, 2), _track(job, 3),
        _face(job, 1, 10, unit((0, 1.0), (1, 0.2))),   # resembles the query
        _face(job, 1, 20, unit((0, 1.0), (2, 0.4))),
        _face(job, 2, 10, unit((3, 1.0))),             # a different person
    ])
    await db_session.commit()

    res = await search_faces(db_session, job, unit((0, 1.0)))
    assert res["status"] == "POSSIBLE_MATCH_FOUND"
    assert [m["track_id"] for m in res["matches"]] == [1]
    m = res["matches"][0]
    assert m["status"] == "POSSIBLE_MATCH" and m["supporting_observations"] == 2
    assert res["coverage"] == {"person_tracks": 3, "tracks_with_usable_face": 2,
                               "tracks_with_low_quality_face_only": 0, "tracks_with_no_face_visible": 1}


@pytest.mark.asyncio
async def test_search_is_scoped_to_one_analysis_run(db_session):
    _, ev, (old, new) = await _seed(db_session)
    db_session.add_all([_track(old, 1), _face(old, 1, 10, unit((0, 1.0))),
                        _track(new, 1), _face(new, 1, 10, unit((4, 1.0)))])
    await db_session.commit()
    assert (await search_faces(db_session, new, unit((0, 1.0))))["matches"] == []
    assert len((await search_faces(db_session, old, unit((0, 1.0))))["matches"]) == 1


@pytest.mark.asyncio
async def test_no_match_says_unchecked_people_may_still_be_present(db_session):
    _, ev, (_, job) = await _seed(db_session)
    db_session.add_all([_track(job, 1), _track(job, 2),
                        _face(job, 1, 10, unit((3, 1.0))),
                        _face(job, 2, 10, quality=FaceQuality.LOW_QUALITY)])
    await db_session.commit()
    res = await search_faces(db_session, job, unit((0, 1.0)))
    assert res["status"] == "NO_MATCH_AMONG_USABLE_FACES"
    assert "may still be present" in res["message"]


@pytest.mark.asyncio
async def test_only_low_quality_faces_means_cannot_tell(db_session):
    _, ev, (_, job) = await _seed(db_session)
    db_session.add_all([_track(job, 1), _face(job, 1, 10, quality=FaceQuality.LOW_QUALITY)])
    await db_session.commit()
    res = await search_faces(db_session, job, unit((0, 1.0)))
    assert res["status"] == "NO_USABLE_FACES"
    assert "neither confirmed nor ruled out" in res["message"]


@pytest.mark.asyncio
async def test_run_without_face_extraction_is_reported_not_treated_as_no_match(db_session):
    _, ev, (_, job) = await _seed(db_session, face_stage="NOT_RUN")
    res = await search_faces(db_session, job, None)
    assert res["status"] == "FACE_EXTRACTION_NOT_AVAILABLE"
    assert "Re-run analysis" in res["message"]


@pytest.mark.asyncio
async def test_threshold_is_respected(db_session):
    _, ev, (_, job) = await _seed(db_session)
    db_session.add_all([_track(job, 1), _face(job, 1, 10, unit((0, 0.6), (1, 0.8)))])  # cos = 0.6
    await db_session.commit()
    assert (await search_faces(db_session, job, unit((0, 1.0)), threshold=0.5))["matches"]
    assert not (await search_faces(db_session, job, unit((0, 1.0)), threshold=0.7))["matches"]


# ======================================================================
# Routes: access control, privacy, audit
# ======================================================================

def _auth(u, role):
    return {"Authorization": f"Bearer {create_access_token(u.id, u.username, role)}"}


def _leaks_embedding(node) -> bool:
    """True if any object in the JSON tree carries an embedding field."""
    if isinstance(node, dict):
        return any(k in ("embedding", "embedding_encrypted") or _leaks_embedding(v) for k, v in node.items())
    if isinstance(node, list):
        return any(_leaks_embedding(v) for v in node)
    return False


@pytest.fixture
def fake_query(monkeypatch):
    monkeypatch.setattr(FaceEngine, "load", classmethod(lambda cls: True))
    monkeypatch.setattr(faces_routes, "embed_query_photo",
                        lambda b: {"embedding": unit((0, 1.0)), "photo_sha256": "ignored"})


@pytest.mark.asyncio
async def test_viewer_cannot_search_or_list_faces(async_client, db_session, fake_query):
    u, ev, _ = await _seed(db_session)
    u.role_id = 4
    await db_session.commit()
    h = _auth(u, "Viewer")
    r = await async_client.post(f"/api/v1/evidence/{ev.id}/faces/search", headers=h,
                                files={"photo": ("q.png", _png(), "image/png")})
    assert r.status_code == 403
    assert (await async_client.get(f"/api/v1/evidence/{ev.id}/faces", headers=h)).status_code == 403


@pytest.mark.asyncio
async def test_search_response_never_contains_embeddings_and_is_audited(async_client, db_session, fake_query):
    u, ev, (_, job) = await _seed(db_session)
    db_session.add_all([_track(job, 1), _face(job, 1, 10, unit((0, 1.0)))])
    await db_session.commit()

    photo = _png()
    r = await async_client.post(f"/api/v1/evidence/{ev.id}/faces/search", headers=_auth(u, "Investigator"),
                                files={"photo": ("q.png", photo, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "POSSIBLE_MATCH_FOUND"
    assert not _leaks_embedding(body)
    assert body["matches"][0]["observations"][0]["crop_url"].startswith(f"/api/v1/evidence/{ev.id}/faces/crops/")

    import hashlib
    entry = (await db_session.execute(
        select(AuditLog).where(AuditLog.action == "FACE_SEARCH", AuditLog.resource_id == str(ev.id))
    )).scalars().one()
    meta = json.loads(entry.metadata_json)
    assert meta["query_photo_sha256"] == hashlib.sha256(photo).hexdigest()
    assert meta["matched_track_ids"] == [1]


@pytest.mark.asyncio
async def test_rejected_query_photo_is_audited(async_client, db_session, monkeypatch):
    monkeypatch.setattr(FaceEngine, "load", classmethod(lambda cls: True))

    def reject(b):
        raise FaceSearchError("No face was found in the uploaded photo.")

    monkeypatch.setattr(faces_routes, "embed_query_photo", reject)
    u, ev, _ = await _seed(db_session)
    r = await async_client.post(f"/api/v1/evidence/{ev.id}/faces/search", headers=_auth(u, "Investigator"),
                                files={"photo": ("q.png", _png(), "image/png")})
    assert r.status_code == 422
    assert (await db_session.execute(select(AuditLog).where(
        AuditLog.action == "FACE_SEARCH_REJECTED", AuditLog.resource_id == str(ev.id)))).scalars().first()


@pytest.mark.asyncio
async def test_face_list_classifies_every_person_track(async_client, db_session):
    u, ev, (_, job) = await _seed(db_session)
    db_session.add_all([_track(job, 1), _track(job, 2), _track(job, 3),
                        _face(job, 1, 10, unit((0, 1.0))),
                        _face(job, 2, 10, quality=FaceQuality.LOW_QUALITY)])
    await db_session.commit()
    r = await async_client.get(f"/api/v1/evidence/{ev.id}/faces", headers=_auth(u, "Investigator"))
    statuses = {t["track_id"]: t["face_status"] for t in r.json()["tracks"]}
    assert statuses == {1: "USABLE_FACE", 2: "LOW_QUALITY", 3: "NO_FACE_VISIBLE"}
    assert not _leaks_embedding(r.json())


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["..%2F..%2Fguardianeye.db", "face_ev999_job1_trk1_fn1.jpg", "secret.jpg"])
async def test_crop_endpoint_rejects_foreign_or_malformed_names(async_client, db_session, name):
    u, ev, _ = await _seed(db_session)
    r = await async_client.get(f"/api/v1/evidence/{ev.id}/faces/crops/{name}", headers=_auth(u, "Investigator"))
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_crop_endpoint_requires_authentication(async_client, db_session):
    u, ev, _ = await _seed(db_session)
    r = await async_client.get(f"/api/v1/evidence/{ev.id}/faces/crops/face_ev{ev.id}_job1_trk1_fn1.jpg")
    assert r.status_code == 401
