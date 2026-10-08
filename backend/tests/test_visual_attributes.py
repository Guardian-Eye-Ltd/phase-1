"""
Visual attribute extraction: colour naming, skin vs cloth, pose regions,
garment/vehicle extraction, carried items, licence plates, and the entities API.

CLIP and OCR are stubbed so these tests are fast and deterministic; the real
models were validated separately on real frames.
"""
import uuid
from datetime import datetime

import numpy as np
import pytest

import app.services.pipeline.attribute_extractor as ax
from app.authentication.jwt import create_access_token
from app.authentication.password import hash_password
from app.models.analysis import AnalysisJob, Detection, JobStage, JobStatus, Track
from app.models.evidence import Evidence, EvidenceStatus
from app.models.observation import EntityType, ObservationSource, ObservationStatus, TrackAttributeAggregate
from app.models.user import User
from app.services.pipeline.alpr_service import (
    PlateRead, _merge_line_fragments, correct_to_indian_format, normalise_plate,
    read_vehicle_plates, validation_score, vote_plate,
)
from app.services.pipeline.attribute_aggregator import AttributeAggregator
from app.services.pipeline.body_regions import (
    face_skin_patch, match_pose_to_boxes, person_regions, vehicle_body_region,
)
from app.services.pipeline.color_naming import read_color, skin_reference
from app.services.semantic.query_parser import QueryIntentParser

SKIN_BGR = (140, 170, 220)
BEIGE_BGR = (175, 205, 225)


def patch(bgr, h=40, w=40):
    return np.full((h, w, 3), bgr, dtype=np.uint8)


# ======================================================================
# Colour naming
# ======================================================================

@pytest.mark.parametrize("bgr,name", [
    ((0, 0, 220), "red"), ((210, 90, 30), "blue"), ((40, 160, 40), "green"),
    ((15, 15, 15), "black"), ((240, 240, 240), "white"), ((128, 128, 128), "grey"),
    ((40, 70, 120), "brown"), (BEIGE_BGR, "beige"), ((20, 210, 230), "yellow"),
])
def test_solid_colours_are_named(bgr, name):
    assert read_color(patch(bgr), exclude_skin=False).color == name


def test_mixed_region_names_no_colour():
    mixed = np.concatenate([patch((0, 0, 220), w=13), patch((210, 90, 30), w=13), patch((40, 160, 40), w=14)], axis=1)
    rd = read_color(mixed, exclude_skin=False, min_pixels=10)
    assert rd.color is None
    assert set(rd.distribution) == {"red", "blue", "green"}


def test_generic_skin_range_catches_beige_cloth():
    """The documented weakness that motivates the person-specific reference."""
    assert read_color(patch(BEIGE_BGR)).skin_fraction > 0.9


def test_person_skin_reference_separates_beige_cloth_from_skin():
    ref = skin_reference(patch(SKIN_BGR, 20, 20))
    assert ref is not None
    assert read_color(patch(SKIN_BGR), skin_ref_lab=ref).skin_fraction > 0.9
    beige = read_color(patch(BEIGE_BGR), skin_ref_lab=ref)
    assert beige.skin_fraction < 0.1 and beige.color == "beige"


def test_skin_reference_needs_skin_pixels():
    assert skin_reference(patch((200, 60, 10), 20, 20)) is None


# ======================================================================
# Pose-guided regions
# ======================================================================

def _kps(**over):
    """Upright person in a 100x300 box at (0,0)."""
    k = np.zeros((17, 2))
    k[0] = (50, 30); k[1] = (45, 25); k[2] = (55, 25)
    k[5] = (30, 70); k[6] = (70, 70); k[11] = (35, 150); k[12] = (65, 150)
    k[13] = (37, 210); k[14] = (63, 210); k[15] = (38, 280); k[16] = (62, 280)
    conf = np.ones(17)
    for idx, c in over.items():
        conf[int(idx[1:])] = c
    return k, conf


def test_regions_follow_keypoints():
    k, c = _kps()
    r = person_regions((0, 0, 100, 300), k, c)
    assert r.source == "POSE"
    assert 70 <= r.upper[1] < r.upper[3] <= 150
    assert 210 <= r.lower[1] < r.lower[3] <= 280      # knee to ankle


def test_hidden_knees_mean_no_lower_region():
    k, c = _kps(k13=0.1, k14=0.1)
    r = person_regions((0, 0, 100, 300), k, c)
    assert r.lower is None and "not visible" in r.lower_reason


def test_hidden_shoulders_mean_no_upper_region():
    k, c = _kps(k5=0.1, k6=0.1)
    assert person_regions((0, 0, 100, 300), k, c).upper is None


def test_without_pose_lower_body_is_never_assumed():
    r = person_regions((0, 0, 100, 300), None, None)
    assert r.source == "BOX" and r.upper is not None and r.lower is None


def test_face_patch_requires_nose_and_eyes():
    k, c = _kps()
    assert face_skin_patch(k, c) is not None
    k2, c2 = _kps(k0=0.1)
    assert face_skin_patch(k2, c2) is None


def test_pose_boxes_are_matched_by_overlap_not_order():
    persons = [(0, 0, 100, 300), (400, 0, 500, 300)]
    pose = [(402, 5, 498, 295), (2, 3, 99, 298)]       # reversed order
    assert match_pose_to_boxes(persons, pose) == [1, 0]


def test_unmatched_pose_is_left_unassigned():
    assert match_pose_to_boxes([(0, 0, 100, 300)], [(500, 0, 600, 300)]) == [None]


def test_vehicle_body_region_is_inside_box():
    x1, y1, x2, y2 = vehicle_body_region((100, 100, 300, 200))
    assert 100 < x1 < x2 < 300 and 100 < y1 < y2 < 200


# ======================================================================
# Garment extraction on synthetic people (CLIP stubbed out)
# ======================================================================

@pytest.fixture
def no_clip(monkeypatch):
    monkeypatch.setattr(ax.clip_attributes, "classify", lambda crop, sets: {})


def _person_frame(torso_bgr, legs_bgr=(210, 90, 30)):
    frame = np.full((320, 120, 3), (90, 140, 60), np.uint8)   # green background
    frame[10:60, 20:80] = SKIN_BGR                              # head/face
    frame[60:155, 10:90] = torso_bgr                            # torso
    frame[155:300, 20:80] = legs_bgr                            # legs
    return frame


def _enrich(frame, kps=None, conf=None):
    det = {"class_name": "person", "bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 100 / 120, "bbox_y2": 300 / 320}
    k, c = _kps() if kps is None else (kps, conf)
    ax.enrich_person(frame, det, (0, 0, 100, 300), k, c)
    return det


def test_clothed_person_gets_top_and_trouser_colours(no_clip):
    det = _enrich(_person_frame((0, 0, 200)))
    assert det["upper_garment_presence"] == "present"
    assert det["upper_garment_color"] == "red"
    assert det["upper_garment_color_source"] == "POSE_CROP_COLOR_MODEL"
    assert det["lower_garment_color"] == "blue"


def test_shirtless_person_gets_no_shirt_colour(no_clip):
    det = _enrich(_person_frame(SKIN_BGR))
    assert det["upper_garment_presence"] == "absent"
    assert "upper_garment_color" not in det


def test_beige_top_is_not_mistaken_for_skin(no_clip):
    det = _enrich(_person_frame(BEIGE_BGR))
    assert det["upper_garment_presence"] == "present"
    assert det["upper_garment_color"] == "beige"


def test_hidden_legs_produce_no_trouser_claim(no_clip):
    k, c = _kps(k13=0.1, k14=0.1)
    det = _enrich(_person_frame((0, 0, 200)), k, c)
    assert "lower_garment_color" not in det
    assert "not visible" in det["lower_garment_not_visible"]


def test_tiny_person_is_skipped(no_clip):
    det = {"class_name": "person", "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 0.1, "bbox_y2": 0.1}
    ax.enrich_person(np.zeros((300, 300, 3), np.uint8), det, (0, 0, 20, 40), None, None)
    assert "too small" in det["attributes_skipped"]
    assert "upper_garment_color" not in det


# ======================================================================
# Carried items: grounded in detected bags only
# ======================================================================

def _box(cls, x1, y1, x2, y2, conf=0.8):
    return {"class_name": cls, "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2, "confidence": conf}


def test_detected_bag_inside_person_is_carried():
    person, bag = _box("person", 0.1, 0.1, 0.3, 0.9), _box("backpack", 0.15, 0.3, 0.25, 0.5, 0.7)
    ax.associate_carried_items([person, bag])
    assert person["carried_item"] == "backpack" and person["carries_bag"] is True
    assert person["carried_item_source"] == "YOLO_DETECTOR"


def test_no_detected_bag_means_no_claim_not_false():
    person = _box("person", 0.1, 0.1, 0.3, 0.9)
    ax.associate_carried_items([person, _box("suitcase", 0.6, 0.6, 0.7, 0.8)])
    assert "carried_item" not in person and "carries_bag" not in person


# ======================================================================
# Vehicle colour: CLIP + body pixels
# ======================================================================

def test_vehicle_pixels_correct_a_misled_clip(monkeypatch):
    """A blue bus with a big green graphic: CLIP says green, the paint says blue."""
    monkeypatch.setattr(ax.clip_attributes, "classify", lambda crop, sets: {
        "color": {"green": 0.46, "grey": 0.23, "blue": 0.10}, "body": {"bus": 0.96}})
    frame = np.full((200, 300, 3), (200, 80, 20), np.uint8)   # blue body
    det = {"class_name": "bus", "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 1, "bbox_y2": 1}
    ax.enrich_vehicle(frame, det, (0, 0, 300, 200))
    assert det["vehicle_color"] == "blue"
    assert det["vehicle_color_source"] == "COMBINED"
    assert det["vehicle_body_style"] == "bus"


def test_tiny_vehicle_is_skipped(monkeypatch):
    monkeypatch.setattr(ax.clip_attributes, "classify", lambda crop, sets: {})
    det = {"class_name": "car"}
    ax.enrich_vehicle(np.zeros((100, 100, 3), np.uint8), det, (0, 0, 20, 15))
    assert "vehicle_color" not in det and "too small" in det["attributes_skipped"]


# ======================================================================
# Licence plates
# ======================================================================

def test_plate_normalisation_and_validation():
    assert normalise_plate("kl-01 ab 1234") == "KL01AB1234"
    assert validation_score("KL01AB1234") == 1.0
    assert validation_score("ABC123X") == 0.7
    assert validation_score("ABCDEF") == 0.0          # no digit
    assert validation_score("12") == 0.0              # too short


@pytest.mark.parametrize("raw,fixed", [
    ("KLO1AB1234", "KL01AB1234"),   # letter O read where the format needs a digit
    ("KLO1A81234", "KL01AB1234"),   # two confusions: O->0 and 8->B
    ("K101AB1234", "KI01AB1234"),
    ("KL01AB1234", "KL01AB1234"),
])
def test_format_guided_correction(raw, fixed):
    assert correct_to_indian_format(raw) == fixed


def test_valid_reads_are_never_rewritten():
    """KL-0-IAB-1234 is itself a valid plate; 'fixing' it would be guessing."""
    assert correct_to_indian_format("KL0IAB1234") == "KL0IAB1234"


def test_correction_never_invents_or_drops_characters():
    out = correct_to_indian_format("KLO1AB1234")
    assert len(out) == len("KLO1AB1234")
    assert correct_to_indian_format("HELLOWORLD") is None


def test_vote_resolves_a_minority_misread():
    v = vote_plate([("KL01AB1234", .9), ("KL01AB1234", .85), ("KL01AB1284", .6), ("KL01AB1234", .8)])
    assert v["status"] == "OBSERVED" and v["value"] == "KL01AB1234"


def test_vote_withholds_a_genuine_disagreement():
    v = vote_plate([("KL01AB1234", .5), ("KL01A81234", .5)])
    assert v["status"] == "WITHHELD" and "disagree" in v["reason"]


def test_vote_withholds_a_single_read():
    assert vote_plate([("KL01AB1234", .95)])["status"] == "WITHHELD"


def test_split_plate_fragments_are_joined():
    results = [([(0, 0), (40, 0), (40, 20), (0, 20)], "KL01", 0.9),
               ([(48, 1), (100, 1), (100, 21), (48, 21)], "AB1234", 0.8)]
    texts = [t for t, _ in _merge_line_fragments(results)]
    assert "KL01AB1234" in texts


def _vehicle_dets(track, widths):
    return [{"class_name": "car", "track_id": track, "frame_number": i, "timestamp": i * 1.0,
             "bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": w / 1000, "bbox_y2": 0.3}
            for i, w in enumerate(widths)]


def test_plate_stage_skips_small_vehicles_and_reads_large_ones():
    frames = [(i, i * 1.0, np.zeros((600, 1000, 3), np.uint8)) for i in range(6)]
    dets = _vehicle_dets(1, [80, 90, 100]) + [
        {**d, "track_id": 2, "frame_number": d["frame_number"] + 3, "timestamp": d["timestamp"] + 3}
        for d in _vehicle_dets(2, [300, 280, 260])]
    calls = []

    def fake_read(crop):
        calls.append(crop.shape[1])
        return [PlateRead("KL01AB1234", "KL01AB1234", 0.9, 1.0)]

    obs = read_vehicle_plates(frames, dets, read_fn=fake_read)
    assert all(w >= 120 for w in calls)                 # track 1 never sent to OCR
    assert {o["track_number"] for o in obs} == {2}
    assert all(o["source"] == "ALPR_OCR" for o in obs)


def test_plate_stage_stops_when_clearest_frame_has_no_text():
    frames = [(i, i * 1.0, np.zeros((600, 1000, 3), np.uint8)) for i in range(3)]
    calls = []
    read_vehicle_plates(frames, _vehicle_dets(1, [300, 280, 260]),
                        read_fn=lambda crop: calls.append(1) or [])
    assert len(calls) == 1


def test_aggregator_votes_plates_character_by_character():
    obs = [{"track_number": 7, "entity_type": "VEHICLE", "attribute": "license_plate_text", "value": v,
            "confidence": c, "frame_number": i, "timestamp": float(i), "source": "ALPR_OCR"}
           for i, (v, c) in enumerate([("KL01AB1234", .9), ("KL01AB1234", .85), ("KL01AB1284", .6)])]
    agg = [a for a in AttributeAggregator.aggregate(obs) if a["attribute"] == "license_plate_text"][0]
    assert agg["status"] == "OBSERVED" and agg["value"] == "KL01AB1234"
    assert agg["confidence_breakdown"]["method"] == "character_vote"


def _obs(track, attr, value, n, conf=0.8):
    return [{"track_number": track, "entity_type": "PERSON", "attribute": attr, "value": value,
             "confidence": conf, "frame_number": i, "timestamp": float(i), "source": "POSE_CROP_COLOR_MODEL"}
            for i in range(n)]


def test_shirt_colour_is_contradicted_when_track_is_bare_chested():
    """Real case: presence 39/42 'absent' but a 3/3 'white' colour from misread frames."""
    obs = _obs(1, "upper_garment_presence", "absent", 39) + _obs(1, "upper_garment_presence", "present", 3) \
        + _obs(1, "upper_garment_color", "white", 3)
    aggs = {a["attribute"]: a for a in AttributeAggregator.aggregate(obs, track_observation_counts={1: 42})}
    assert aggs["upper_garment_presence"]["status"] == "OBSERVED"
    assert aggs["upper_garment_color"]["status"] == "CONTRADICTED"
    assert "absent" in aggs["upper_garment_color"]["confidence_breakdown"]["contradicted_by"]


def test_shirt_colour_stands_when_garment_is_present():
    obs = _obs(2, "upper_garment_presence", "present", 10) + _obs(2, "upper_garment_color", "red", 10)
    aggs = {a["attribute"]: a for a in AttributeAggregator.aggregate(obs, track_observation_counts={2: 10})}
    assert aggs["upper_garment_color"]["status"] == "OBSERVED"


def test_observation_source_is_carried_through():
    det = {"track_id": 3, "class_name": "person", "timestamp": 1.0, "frame_number": 10,
           "upper_garment_color": "red", "upper_garment_color_confidence": 0.8,
           "upper_garment_color_source": "POSE_CROP_COLOR_MODEL"}
    obs = AttributeAggregator.extract_observations([det])
    assert obs[0]["source"] == "POSE_CROP_COLOR_MODEL"


# ======================================================================
# Query parsing
# ======================================================================

def _attrs(q):
    return QueryIntentParser.parse_query(q)["structured_intent"]["attributes"]


def test_colour_synonyms_map_to_stored_names():
    assert _attrs("find the silver car")["vehicle_color"] == "grey"
    assert _attrs("find the gray car")["vehicle_color"] == "grey"
    assert _attrs("find a person wearing a cream jacket")["upper_garment_color"] == "beige"


def test_shirtless_query():
    si = QueryIntentParser.parse_query("find the shirtless man")["structured_intent"]
    assert si["intent"] == "ATTRIBUTE_SEARCH" and si["attributes"] == {"upper_garment_presence": "absent"}


def test_specific_and_generic_bag_queries():
    assert _attrs("find the person carrying a suitcase")["carried_item"] == "suitcase"
    generic = _attrs("find someone carrying a bag")
    assert generic["carries_bag"] == "true" and "carried_item" not in generic


# ======================================================================
# Entities endpoint
# ======================================================================

@pytest.mark.asyncio
async def test_entities_explain_what_could_not_be_determined(async_client, db_session):
    tag = uuid.uuid4().hex[:8]
    u = User(full_name="E", username=f"e_{tag}", email=f"{tag}@t.local",
             password_hash=hash_password("Pw123456!"), role_id=3, is_active=True)
    db_session.add(u); await db_session.commit(); await db_session.refresh(u)
    ev = Evidence(original_filename="r.mp4", stored_filename=f"{tag}.mp4", file_path="/tmp/x", file_size=1,
                  mime_type="video/mp4", sha256_hash=tag.ljust(64, "a"), status=EvidenceStatus.COMPLETED,
                  uploaded_by=u.id, resolution="1920x1080")
    db_session.add(ev); await db_session.commit(); await db_session.refresh(ev)
    job = AnalysisJob(evidence_id=ev.id, status=JobStatus.COMPLETED, current_stage=JobStage.FINALIZING,
                      progress=100.0, completed_at=datetime(2026, 5, 1))
    db_session.add(job); await db_session.commit(); await db_session.refresh(job)

    def track(n, cls):
        return Track(analysis_job_id=job.id, evidence_id=ev.id, track_number=n, class_name=cls,
                     first_seen_timestamp=1, last_seen_timestamp=4, first_seen_frame=1, last_seen_frame=40,
                     duration=3, observation_count=6, keyframe_count=1)

    db_session.add_all([
        track(1, "person"), track(2, "car"),
        Detection(analysis_job_id=job.id, evidence_id=ev.id, frame_number=1, timestamp=1, class_name="car",
                  confidence=0.9, bbox_x1=0.1, bbox_y1=0.1, bbox_x2=0.15, bbox_y2=0.2, track_id=2),  # ~96px wide
        TrackAttributeAggregate(
            evidence_id=ev.id, analysis_job_id=job.id, track_number=1, entity_type=EntityType.PERSON,
            attribute="upper_garment_presence", value="absent", confidence=0.8, observation_count=6,
            supporting_count=6, dissenting_count=0, source=ObservationSource.POSE_CROP_COLOR_MODEL,
            status=ObservationStatus.OBSERVED, confidence_breakdown={}),
        TrackAttributeAggregate(
            evidence_id=ev.id, analysis_job_id=job.id, track_number=2, entity_type=EntityType.VEHICLE,
            attribute="vehicle_color", value="black", confidence=0.7, observation_count=6,
            supporting_count=5, dissenting_count=1, source=ObservationSource.COMBINED,
            status=ObservationStatus.OBSERVED, confidence_breakdown={}),
    ])
    await db_session.commit()

    r = await async_client.get(f"/api/v1/evidence/{ev.id}/entities",
                               headers={"Authorization": f"Bearer {create_access_token(u.id, u.username, 'Investigator')}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["attributes_extracted"] is True
    ents = {e["track_id"]: e for e in body["entities"]}
    assert ents[1]["attributes"]["upper_garment_presence"]["value"] == "absent"
    assert "lower_garment" in ents[1]["not_determined"]
    assert ents[2]["attributes"]["vehicle_color"]["value"] == "black"
    assert "too small" in ents[2]["not_determined"]["license_plate_text"]
    assert "misidentified" in ents[2]["not_determined"]["vehicle_make_model"]
