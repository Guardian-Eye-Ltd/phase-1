"""
Structured answers for semantic search.

Plate and attribute questions have exact answers in TrackAttributeAggregate
(the per-track consensus the People & Vehicles view shows). Vector similarity
over narrative text cannot answer "what is the vehicle number" — an embedding
of "KA3K961" is not near an embedding of the question — so these intents are
answered from the database first and only fall back to vector search when the
structured lookup has nothing to say.

Results use the same item shape as vector results so the search UI renders
them unchanged.
"""
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.semantic import DocumentType, ForensicDocument

logger = logging.getLogger(__name__)

MAX_LISTED_VEHICLES = 25

_ATTR_LABELS = {
    "vehicle_color": "colour",
    "upper_garment_color": "upper garment",
    "lower_garment_color": "lower garment",
    "upper_garment_presence": "upper garment presence",
    "headwear": "headwear",
    "carried_item": "carried item",
    "carries_bag": "carrying a bag",
    "license_plate_text": "plate",
}


def _cap(text: str) -> str:
    """Upper-case the first letter only (str.capitalize would lower-case plate text)."""
    return text[:1].upper() + text[1:]


def _ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _item(*, title: str, summary: str, track_id: Optional[int], start: Optional[float], end: Optional[float],
          confidence_level: str, confidence_reason: str, model_confidence: Optional[float],
          attribute_chips: List[Dict[str, str]], verified: bool, supporting_frames: Optional[int],
          doc_id: Optional[int] = None) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "title": title,
        "summary": summary,
        "score": 1.0,
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "query_relevance": 100.0,
        "evidence_support": confidence_level,
        "model_confidence": model_confidence,
        "track_id": track_id,
        "keyframe_id": None,
        "start_time": start,
        "end_time": end,
        "why_explanation": {
            "semantic_match_score": 1.0,
            "semantic_match_percentage": "structured",
            "query_relevance": 100.0,
            "query_relevance_formatted": "Direct lookup",
            "entity_match": True,
            "entity_match_details": "✓ Per-track attribute consensus",
            "visual_attribute_matches": attribute_chips,
            "supporting_frames": supporting_frames,
            "supporting_keyframes": 0,
            "supporting_track": track_id,
            "evidence_support": confidence_level,
            "document_type": "TRACK_ATTRIBUTE_AGGREGATE",
            "source_type": "OBSERVATION",
            "temporal_range": f"{start:.2f}s - {end:.2f}s" if start is not None and end is not None else "N/A",
        },
        "verification_status": "VERIFIED" if verified else "UNVERIFIED",
    }


def _plate_item(v: Dict[str, Any]) -> Dict[str, Any]:
    p = v["plate"]
    who = f"{v['class_name']} Track #{v['track_id']}"
    colour = f"{v['vehicle_color']} " if v.get("vehicle_color") else ""
    seen = f"seen {_ts(v['start_time'])}–{_ts(v['end_time'])}"
    reads = p.get("reads_total")
    conf = p.get("confidence")

    if p["status"] == "READ":
        issue = p.get("format_issue") or (None if p.get("format_complete") else "it does not form a complete plate number")
        partial = f" Not a complete, valid number: {issue}." if issue else ""
        return _item(
            title=_cap(f"{colour}{who} — plate {p['text']}"),
            summary=(f"Licence plate (vehicle number) of {colour}{who} read as {p['text']} ({seen}). "
                     f"{reads} OCR read(s); every character agreed by at least two reads.{partial} "
                     f"Plate OCR is a degraded capability: check the footage before relying on it."),
            track_id=v["track_id"], start=v["start_time"], end=v["end_time"],
            confidence_level="MEDIUM" if (conf or 0) >= 0.35 else "LOW",
            confidence_reason=f"OCR consensus across {reads} read(s) of this vehicle (plate tier {p.get('evidence_tier')}).",
            model_confidence=conf,
            attribute_chips=[{"name": "PLATE", "status": "VERIFIED", "formatted": f"✓ PLATE {p['text']}"}],
            verified=True, supporting_frames=reads,
        )
    if p["status"] == "UNCONFIRMED":
        return _item(
            title=_cap(f"{colour}{who} — plate not confirmed"),
            summary=(f"OCR read text on {colour}{who} ({seen}) but could not confirm it: {p['reason']}. "
                     f"Best guess {p['text']} — not confirmed; do not treat it as the vehicle number."),
            track_id=v["track_id"], start=v["start_time"], end=v["end_time"],
            confidence_level="LOW",
            confidence_reason="Plate withheld: OCR reads were too few or inconsistent.",
            model_confidence=conf,
            attribute_chips=[{"name": "PLATE", "status": "NOT_VERIFIED",
                              "formatted": f"Unconfirmed read: {p['text']}"}],
            verified=False, supporting_frames=reads,
        )
    return _item(
        title=_cap(f"{colour}{who} — no plate read"),
        summary=f"No licence plate text was read for {colour}{who} ({seen}): {p['reason']}.",
        track_id=v["track_id"], start=v["start_time"], end=v["end_time"],
        confidence_level="LOW",
        confidence_reason="No plate text available for this vehicle.",
        model_confidence=None,
        attribute_chips=[{"name": "PLATE", "status": "NOT_VERIFIED", "formatted": "Plate not readable"}],
        verified=False, supporting_frames=v.get("observation_count"),
    )


async def _plate_lookup(db, evidence_id, job_id, structured) -> Dict[str, Any]:
    from app.services.agents.investigation_tools import InvestigationToolSystem as T

    entity = structured.get("entity") or "vehicle"
    track_id = structured.get("track_id")
    colour = (structured.get("attributes") or {}).get("vehicle_color")
    res = await T.list_vehicle_plates(db, evidence_id, job_id, entity=entity,
                                      track_number=track_id, vehicle_color=colour)
    vehicles = res["vehicles"]
    scope = " ".join(x for x in (colour, entity) if x)

    if not vehicles:
        if track_id is not None:
            answer = (f"Track #{track_id} is not a {entity} track in analysis job #{job_id}, "
                      f"so it has no plate.")
        elif colour:
            answer = (f"No {scope} track in analysis job #{job_id} has a confirmed {colour} colour, "
                      f"so no plate can be attributed to a {scope}.")
        else:
            answer = f"No {entity} tracks were found in analysis job #{job_id}."
        return {"answer": answer, "results": [], "vehicles": [], "definitive": True}

    c = res["counts"]
    read = [v for v in vehicles if v["plate"]["status"] == "READ"]
    if read:
        listed = ", ".join(
            f"{v['plate']['text']} (Track #{v['track_id']}"
            + ("" if v["plate"].get("format_complete") in (True, None) else "; not a valid complete number")
            + ")"
            for v in read[:5]
        )
        lead = f"Plate read for {c['READ']} of {len(vehicles)} {scope} track(s): {listed}."
    else:
        lead = f"No plate could be read with agreeing OCR reads for any of the {len(vehicles)} {scope} track(s)."
    tail = []
    if c["UNCONFIRMED"]:
        tail.append(f"{c['UNCONFIRMED']} had inconsistent reads (shown as unconfirmed)")
    if c["NOT_READ"]:
        tail.append(f"{c['NOT_READ']} had no readable plate")
    answer = " ".join(x for x in (lead, _cap("; ".join(tail)) + "." if tail else "", f"(analysis job #{job_id})") if x)
    return {"answer": answer, "results": [_plate_item(v) for v in vehicles[:MAX_LISTED_VEHICLES]],
            "vehicles": vehicles, "definitive": True}


def _same_length_diff(a: str, b: str) -> Optional[int]:
    return sum(1 for x, y in zip(a, b) if x != y) if len(a) == len(b) else None


async def _plate_search(db, evidence_id, job_id, plate: str) -> Dict[str, Any]:
    from app.services.agents.investigation_tools import InvestigationToolSystem as T

    wanted = "".join(ch for ch in plate.upper() if ch.isalnum())
    hits = await T.find_vehicle_by_plate(db, evidence_id, job_id, wanted)
    if not hits:
        hits = await T.find_vehicle_by_plate(db, evidence_id, job_id, wanted, allow_fuzzy=True)
    listing = await T.list_vehicle_plates(db, evidence_id, job_id, entity="vehicle")
    by_track = {v["track_id"]: v for v in listing["vehicles"]}

    results, shown = [], []
    for h in hits:
        v = by_track.get(h["track_id"])
        if v is None:
            continue
        item = _plate_item(v)
        if h["match_type"] == "POSSIBLE_PLATE_MATCH":
            item["title"] = f"Possible match — {item['title']}"
            item["summary"] = (f"Not an exact match: {h['character_differences']} character(s) differ from "
                               f"{wanted}. " + item["summary"])
            item["confidence_level"] = item["why_explanation"]["evidence_support"] = "LOW"
            item["verification_status"] = "PARTIALLY_SUPPORTED"
            item["why_explanation"]["visual_attribute_matches"] = [{
                "name": "PLATE", "status": "NOT_VERIFIED",
                "formatted": f"Near miss: {h['plate_text']} vs {wanted}",
            }]
        results.append(item)
        shown.append(v)

    # Unconfirmed reads are shown only as unconfirmed, never as an identification.
    for v in listing["vehicles"]:
        p = v["plate"]
        if p["status"] == "UNCONFIRMED" and v["track_id"] not in {h["track_id"] for h in hits}:
            text = "".join(ch for ch in (p["text"] or "").upper() if ch.isalnum())
            d = _same_length_diff(text, wanted)
            if text == wanted or (d is not None and d <= 1):
                results.append(_plate_item(v))
                shown.append(v)

    exact = [h for h in hits if h["match_type"] == "EXACT_PLATE_MATCH"]
    if exact:
        answer = (f"Plate {wanted} was read on " +
                  ", ".join(f"{by_track[h['track_id']]['class_name']} Track #{h['track_id']}"
                            for h in exact if h["track_id"] in by_track) +
                  f" (analysis job #{job_id}). OCR is a degraded capability: verify against the footage.")
    elif results:
        answer = (f"No vehicle has a plate read exactly as {wanted}. Similar or unconfirmed reads are "
                  f"listed below; none is an identification (analysis job #{job_id}).")
    else:
        read = [v for v in listing["vehicles"] if v["plate"]["status"] == "READ"]
        known = ", ".join(f"{v['plate']['text']} (Track #{v['track_id']})" for v in read[:5])
        answer = (f"No plate matching {wanted} was read in analysis job #{job_id}. "
                  + (f"Plates read in this video: {known}." if known else
                     "No plate in this video was read with agreeing OCR reads."))
    return {"answer": answer, "results": results, "vehicles": shown, "definitive": True}


async def _attribute_search(db, evidence_id, job_id, structured) -> Optional[Dict[str, Any]]:
    from app.services.agents.investigation_tools import InvestigationToolSystem as T

    attrs = {k: v for k, v in (structured.get("attributes") or {}).items() if k != "clothing" and v is not None}
    if not attrs:
        return None
    entity = structured.get("entity")
    tracks = await T.find_tracks_by_attributes(db, evidence_id, job_id, attrs, entity=entity)
    wanted = ", ".join(f"{_ATTR_LABELS.get(k, k)} {v}" for k, v in attrs.items())
    if not tracks:
        return {
            "answer": (f"No tracked {entity or 'entity'} has a confirmed consensus for {wanted} in analysis "
                       f"job #{job_id}."),
            "results": [], "definitive": False,
        }

    docs = (await db.execute(select(ForensicDocument).where(
        ForensicDocument.evidence_id == evidence_id,
        ForensicDocument.analysis_job_id == job_id,
        ForensicDocument.document_type == DocumentType.TRACK,
        ForensicDocument.track_id.in_([t["track_id"] for t in tracks]),
    ))).scalars().all()
    doc_by_track = {d.track_id: d for d in docs}

    results = []
    for t in tracks:
        chips, parts = [], []
        for attr, d in t["attributes"].items():
            chips.append({"name": _ATTR_LABELS.get(attr, attr).upper(), "status": "VERIFIED",
                          "formatted": f"✓ {str(d['value']).upper()} ({d['supporting_count']}/{d['observation_count']} frames)"})
            parts.append(f"{_ATTR_LABELS.get(attr, attr)} {d['value']} in {d['supporting_count']} of "
                         f"{d['observation_count']} observed frames")
        doc = doc_by_track.get(t["track_id"])
        who = f"{(t['class_name'] or 'entity')} Track #{t['track_id']}"
        summary = (doc.content if doc else f"{who} seen {_ts(t['start_time'])}–{_ts(t['end_time'])}.")
        conf = t["match_confidence"]
        results.append(_item(
            title=_cap(f"{who} ({_ts(t['start_time'])} - {_ts(t['end_time'])})"),
            summary=f"{summary} Matched: {'; '.join(parts)}.",
            track_id=t["track_id"], start=t["start_time"], end=t["end_time"],
            confidence_level="HIGH" if conf >= 0.65 else ("MEDIUM" if conf >= 0.4 else "LOW"),
            confidence_reason="Attribute consensus across the track's frames (not a single frame).",
            model_confidence=conf, attribute_chips=chips, verified=True,
            supporting_frames=t.get("observation_count"), doc_id=doc.id if doc else None,
        ))
    answer = (f"{len(tracks)} {entity or 'entity'} track(s) match {wanted} "
              f"(analysis job #{job_id}): " + ", ".join(f"Track #{t['track_id']}" for t in tracks[:10]) + ".")
    return {"answer": answer, "results": results, "definitive": True}


async def structured_search(db: AsyncSession, evidence_id: int, job_id: int,
                            structured: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    {"answer", "results", "definitive"} for plate / attribute intents, else None.
    definitive=False means "nothing matched structurally" and the caller may
    still run vector search, prefixing this answer so the two are not confused.
    """
    intent = structured.get("intent")
    attrs = structured.get("attributes") or {}
    if intent == "PLATE_LOOKUP":
        return await _plate_lookup(db, evidence_id, job_id, structured)
    if intent == "ATTRIBUTE_SEARCH" and attrs.get("license_plate_text"):
        return await _plate_search(db, evidence_id, job_id, attrs["license_plate_text"])
    if intent == "ATTRIBUTE_SEARCH":
        return await _attribute_search(db, evidence_id, job_id, structured)
    return None
