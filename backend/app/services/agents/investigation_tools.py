import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct, and_

from app.models.analysis import Track, Keyframe, Detection, ActivityInterval, PossibleInteraction
from app.models.semantic import ForensicDocument, DocumentType
from app.models.observation import (
    VisualAttributeObservation, TrackAttributeAggregate,
    ObservationStatus, EntityType,
)
from app.services.semantic.hybrid_search_engine import HybridSearchEngine
from app.services.pipeline.event_engine import EventEngine
from app.services.agents.taxonomy import (
    classes_for_entity,
    normalize_entity_class,
    all_vehicle_classes,
)

logger = logging.getLogger(__name__)

# Only these can answer a search. WITHHELD (weak/contested) and CONTRADICTED
# (conflicts with another attribute of the same track) never match.
_ASSERTED_STATUSES = (ObservationStatus.OBSERVED, ObservationStatus.SUPPORTED, ObservationStatus.VERIFIED)


def _scope_filter(model, evidence_id: int, analysis_job_id: Optional[int]):
    """
    (evidence_id, analysis_job_id) WHERE clauses. A missing job id matches
    nothing — it must never widen the query to every run of the evidence.
    """
    if analysis_job_id is None:
        return [model.evidence_id == evidence_id, model.analysis_job_id.is_(None)]
    return [model.evidence_id == evidence_id, model.analysis_job_id == analysis_job_id]


async def _resolve_job(db: AsyncSession, evidence_id: int, analysis_job_id: Optional[int]) -> Optional[int]:
    if analysis_job_id is not None:
        return analysis_job_id
    from app.services.analysis_jobs import get_active_analysis_job_id
    return await get_active_analysis_job_id(db, evidence_id)


class InvestigationToolSystem:
    """
    Controlled tool system providing strict, evidence-grounded database queries for agents.
    All tools accept analysis_job_id so investigation results never silently merge
    multiple analysis runs for the same evidence.
    """

    # ------------------------------------------------------------------
    # Deterministic aggregation tools (Section 5 & 8 of the brief)
    # ------------------------------------------------------------------

    @classmethod
    async def count_entities(
        cls,
        db: AsyncSession,
        evidence_id: int,
        analysis_job_id: int,
        entity: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Count DISTINCT tracked entities of a canonical type (e.g. "vehicle",
        "car", "person") inside ONE analysis job, optionally within a time
        window. Returns deterministic numeric evidence; never calls the LLM.
        """
        classes = [c.lower() for c in classes_for_entity(entity)]
        if not classes:
            return {
                "intent": "COUNT",
                "entity": entity,
                "aggregation": "DISTINCT_TRACK_COUNT",
                "analysis_job_id": analysis_job_id,
                "total": 0,
                "breakdown": {},
                "track_ids": [],
                "evidence_basis": "TRACKS",
                "diagnostic": {"reason": "unknown_entity"},
            }

        stmt = select(Track).where(
            Track.evidence_id == evidence_id,
            Track.analysis_job_id == analysis_job_id,
            func.lower(Track.class_name).in_(classes),
        )
        if start_time is not None:
            stmt = stmt.where(Track.last_seen_timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(Track.first_seen_timestamp <= end_time)

        rows = (await db.execute(stmt)).scalars().all()

        # Normalize each track to its canonical bucket, then deduplicate.
        # (analysis_job_id, track_number) is the authoritative identity.
        seen: Dict[int, str] = {}
        breakdown: Dict[str, int] = {}
        for t in rows:
            if t.track_number in seen:
                continue
            canonical = normalize_entity_class(t.class_name) or t.class_name.lower()
            # When the user asked for a specific class, exclude others.
            if entity.lower() != "vehicle" and canonical != entity.lower():
                continue
            seen[t.track_number] = canonical
            breakdown[canonical] = breakdown.get(canonical, 0) + 1

        counted_ids = sorted(seen.keys())
        total = len(counted_ids)

        result = {
            "intent": "COUNT",
            "entity": entity.lower(),
            "scope": "ENTIRE_VIDEO" if (start_time is None and end_time is None) else "TIME_WINDOW",
            "aggregation": "DISTINCT_TRACK_COUNT",
            "analysis_job_id": analysis_job_id,
            "total": total,
            "breakdown": breakdown,
            "track_ids": counted_ids,
            "evidence_basis": "TRACKS",
            "diagnostic": {
                "candidate_track_rows": len(rows),
                "unique_track_ids": counted_ids,
                "counted_track_ids": counted_ids,
                "classes_included": classes,
            },
        }
        logger.info(
            "[COUNT] evidence=%s job=%s entity=%s total=%s breakdown=%s",
            evidence_id, analysis_job_id, entity, total, breakdown,
        )
        return result

    @classmethod
    async def count_entities_at_time(
        cls,
        db: AsyncSession,
        evidence_id: int,
        analysis_job_id: int,
        entity: str,
        timestamp: float,
    ) -> Dict[str, Any]:
        """
        Count DISTINCT tracks of entity that are ACTIVE at a given timestamp.
        (first_seen <= t <= last_seen). Never sums per-frame VLM counts.
        """
        classes = [c.lower() for c in classes_for_entity(entity)]
        stmt = select(Track).where(
            Track.evidence_id == evidence_id,
            Track.analysis_job_id == analysis_job_id,
            func.lower(Track.class_name).in_(classes),
            Track.first_seen_timestamp <= timestamp,
            Track.last_seen_timestamp >= timestamp,
        )
        rows = (await db.execute(stmt)).scalars().all()

        seen: Dict[int, str] = {}
        breakdown: Dict[str, int] = {}
        for t in rows:
            if t.track_number in seen:
                continue
            canonical = normalize_entity_class(t.class_name) or t.class_name.lower()
            if entity.lower() != "vehicle" and canonical != entity.lower():
                continue
            seen[t.track_number] = canonical
            breakdown[canonical] = breakdown.get(canonical, 0) + 1

        counted_ids = sorted(seen.keys())
        result = {
            "intent": "FRAME_COUNT",
            "entity": entity.lower(),
            "timestamp": timestamp,
            "scope": "TIMESTAMP",
            "aggregation": "DISTINCT_ACTIVE_TRACK_COUNT",
            "analysis_job_id": analysis_job_id,
            "count": len(counted_ids),
            "breakdown": breakdown,
            "track_ids": counted_ids,
            "evidence_basis": "TRACKS",
            "diagnostic": {
                "candidate_track_rows": len(rows),
                "classes_included": classes,
            },
        }
        logger.info(
            "[FRAME_COUNT] evidence=%s job=%s entity=%s t=%.2f count=%s",
            evidence_id, analysis_job_id, entity, timestamp, result["count"],
        )
        return result

    # ------------------------------------------------------------------
    # Structured attribute search over temporally aggregated observations
    # ------------------------------------------------------------------

    @classmethod
    async def find_tracks_by_attributes(
        cls,
        db: AsyncSession,
        evidence_id: int,
        analysis_job_id: int,
        attributes: Dict[str, Any],
        entity: Optional[str] = None,
        include_withheld: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Find tracks whose aggregated attributes match ALL requested key/values.

        Queries TrackAttributeAggregate (temporal consensus), never raw
        per-frame observations — a single frame is not evidence of an attribute.
        WITHHELD aggregates are excluded by default so contested or thin
        observations cannot surface as findings.
        """
        if not attributes:
            return []

        # "clothing" is a descriptive list, not an aggregated column.
        queryable = {
            k: v for k, v in attributes.items()
            if k not in ("clothing",) and v is not None
        }
        if not queryable:
            return []

        entity_type = None
        if entity:
            e = entity.lower()
            if e == "person":
                entity_type = EntityType.PERSON
            elif e in ("car", "truck", "bus", "motorcycle", "van", "vehicle", "bicycle"):
                entity_type = EntityType.VEHICLE

        # Each attribute is matched independently; a track must satisfy all.
        matched_per_attr: List[set] = []
        agg_by_track: Dict[int, Dict[str, Any]] = {}

        for attr, value in queryable.items():
            conds = [
                TrackAttributeAggregate.evidence_id == evidence_id,
                TrackAttributeAggregate.analysis_job_id == analysis_job_id,
                TrackAttributeAggregate.attribute == attr,
                func.lower(TrackAttributeAggregate.value) == str(value).strip().lower(),
            ]
            if entity_type is not None:
                conds.append(TrackAttributeAggregate.entity_type == entity_type)
            if not include_withheld:
                conds.append(TrackAttributeAggregate.status.in_(_ASSERTED_STATUSES))

            rows = (await db.execute(select(TrackAttributeAggregate).where(*conds))).scalars().all()
            matched_per_attr.append({r.track_number for r in rows})
            for r in rows:
                agg_by_track.setdefault(r.track_number, {})[attr] = r

        if not matched_per_attr:
            return []
        common = set.intersection(*matched_per_attr)
        if not common:
            logger.info(
                "[ATTR_SEARCH] job=%s attrs=%s -> no track satisfied all attributes",
                analysis_job_id, queryable,
            )
            return []

        # Enrich with track timing + supporting frame timestamps.
        results: List[Dict[str, Any]] = []
        # A specific class ("car", "bus") is filtered by the detector class using
        # the same taxonomy as counting, so "find white cars" and "how many cars"
        # agree, and a train the detector labelled "bus" is not a "car".
        allowed_classes = None
        if entity and entity.lower() not in ("person", "vehicle"):
            allowed_classes = {c.lower() for c in classes_for_entity(entity)}

        for track_number in sorted(common):
            trk = (await db.execute(
                select(Track).where(
                    Track.evidence_id == evidence_id,
                    Track.analysis_job_id == analysis_job_id,
                    Track.track_number == track_number,
                )
            )).scalars().first()
            if allowed_classes is not None and (trk is None or trk.class_name.lower() not in allowed_classes):
                continue

            attr_detail = {}
            overall = []
            for attr, agg in agg_by_track.get(track_number, {}).items():
                supporting = (await db.execute(
                    select(VisualAttributeObservation.timestamp).where(
                        VisualAttributeObservation.analysis_job_id == analysis_job_id,
                        VisualAttributeObservation.track_number == track_number,
                        VisualAttributeObservation.attribute == attr,
                        func.lower(VisualAttributeObservation.value) == str(agg.value).lower(),
                    ).order_by(VisualAttributeObservation.timestamp)
                )).scalars().all()

                attr_detail[attr] = {
                    "value": agg.value,
                    "confidence": agg.confidence,
                    "observation_count": agg.observation_count,
                    "supporting_count": agg.supporting_count,
                    "dissenting_count": agg.dissenting_count,
                    "status": agg.status.value,
                    "confidence_breakdown": agg.confidence_breakdown,
                    "supporting_frames": [round(float(t), 2) for t in supporting],
                }
                overall.append(agg.confidence)

            results.append({
                "track_id": track_number,
                "class_name": trk.class_name if trk else None,
                "start_time": trk.first_seen_timestamp if trk else None,
                "end_time": trk.last_seen_timestamp if trk else None,
                "duration": trk.duration if trk else None,
                "observation_count": trk.observation_count if trk else None,
                "attributes": attr_detail,
                "match_confidence": round(min(overall), 4) if overall else 0.0,
                "evidence_basis": "TRACK_ATTRIBUTE_AGGREGATE",
            })

        results.sort(key=lambda r: r["match_confidence"], reverse=True)
        logger.info(
            "[ATTR_SEARCH] job=%s attrs=%s -> %d matching track(s)",
            analysis_job_id, queryable, len(results),
        )
        return results

    @classmethod
    async def find_vehicle_by_plate(
        cls,
        db: AsyncSession,
        evidence_id: int,
        analysis_job_id: int,
        plate_text: str,
        allow_fuzzy: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Exact plate lookup against aggregated OCR consensus.

        Fuzzy matching is opt-in and its results are labelled
        POSSIBLE_PLATE_MATCH so a near-miss is never presented as an identification.
        """
        normalized = "".join(ch for ch in plate_text.upper() if ch.isalnum())

        rows = (await db.execute(
            select(TrackAttributeAggregate).where(
                TrackAttributeAggregate.evidence_id == evidence_id,
                TrackAttributeAggregate.analysis_job_id == analysis_job_id,
                TrackAttributeAggregate.attribute == "license_plate_text",
                TrackAttributeAggregate.status.in_(_ASSERTED_STATUSES),
            )
        )).scalars().all()

        exact, fuzzy = [], []
        for r in rows:
            candidate = "".join(ch for ch in r.value.upper() if ch.isalnum())
            if candidate == normalized:
                exact.append((r, "EXACT_PLATE_MATCH", 0))
            elif allow_fuzzy and len(candidate) == len(normalized):
                diff = sum(1 for a, b in zip(candidate, normalized) if a != b)
                if diff <= 2:
                    fuzzy.append((r, "POSSIBLE_PLATE_MATCH", diff))

        chosen = exact if exact else fuzzy
        results = [
            {
                "track_id": r.track_number,
                "plate_text": r.value,
                "match_type": label,
                "character_differences": diff,
                "confidence": r.confidence,
                "observation_count": r.observation_count,
                "supporting_count": r.supporting_count,
                "status": r.status.value,
                "evidence_basis": "TRACK_ATTRIBUTE_AGGREGATE",
            }
            for r, label, diff in chosen
        ]
        logger.info(
            "[PLATE_SEARCH] job=%s query=%s -> %d match(es) (exact=%d fuzzy=%d)",
            analysis_job_id, normalized, len(results), len(exact), len(fuzzy),
        )
        return results

    @classmethod
    async def get_visual_attributes(
        cls,
        db: AsyncSession,
        evidence_id: int,
        analysis_job_id: int,
        track_number: int,
        include_withheld: bool = True,
    ) -> Dict[str, Any]:
        """All aggregated attributes for one track, for 'show evidence' views."""
        conds = [
            TrackAttributeAggregate.evidence_id == evidence_id,
            TrackAttributeAggregate.analysis_job_id == analysis_job_id,
            TrackAttributeAggregate.track_number == track_number,
        ]
        if not include_withheld:
            conds.append(TrackAttributeAggregate.status.in_(_ASSERTED_STATUSES))
        rows = (await db.execute(select(TrackAttributeAggregate).where(*conds))).scalars().all()
        return {
            "track_id": track_number,
            "analysis_job_id": analysis_job_id,
            "attributes": {
                r.attribute: {
                    "value": r.value,
                    "confidence": r.confidence,
                    "status": r.status.value,
                    "observation_count": r.observation_count,
                    "supporting_count": r.supporting_count,
                    "confidence_breakdown": r.confidence_breakdown,
                }
                for r in rows
            },
        }

    # ------------------------------------------------------------------
    # Attribute / semantic tools (now job-scoped)
    # ------------------------------------------------------------------

    @classmethod
    async def find_persons(
        cls,
        db: AsyncSession,
        evidence_id: int,
        analysis_job_id: Optional[int] = None,
        upper_color: Optional[str] = None,
        lower_color: Optional[str] = None,
        has_bag: Optional[bool] = None,
        headwear: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Finds person tracks matching visual attributes or temporal window."""
        analysis_job_id = await _resolve_job(db, evidence_id, analysis_job_id)
        conds = _scope_filter(Track, evidence_id, analysis_job_id) + [Track.class_name == "person"]
        stmt = select(Track).where(*conds)
        if start_time is not None:
            stmt = stmt.where(Track.last_seen_timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(Track.first_seen_timestamp <= end_time)

        tracks = (await db.execute(stmt)).scalars().all()
        results: List[Dict[str, Any]] = []

        for trk in tracks:
            doc_conds = _scope_filter(ForensicDocument, evidence_id, analysis_job_id) + [
                ForensicDocument.track_id == trk.track_number,
                ForensicDocument.document_type == DocumentType.TRACK,
            ]
            docs = (await db.execute(select(ForensicDocument).where(*doc_conds))).scalars().all()
            doc_meta = docs[0].metadata_json if docs else {}

            if upper_color and doc_meta.get("upper_garment_color") and upper_color.lower() not in str(doc_meta["upper_garment_color"]).lower():
                continue
            if lower_color and doc_meta.get("lower_garment_color") and lower_color.lower() not in str(doc_meta["lower_garment_color"]).lower():
                continue
            if headwear and doc_meta.get("headwear") and headwear.lower() not in str(doc_meta["headwear"]).lower():
                continue
            if has_bag is True and not doc_meta.get("carries_bag"):
                continue

            results.append({
                "track_id": trk.track_number,
                "class_name": trk.class_name,
                "start_time": trk.first_seen_timestamp,
                "end_time": trk.last_seen_timestamp,
                "duration": trk.duration,
                "upper_garment_color": doc_meta.get("upper_garment_color"),
                "lower_garment_color": doc_meta.get("lower_garment_color"),
                "headwear": doc_meta.get("headwear"),
                "carries_bag": doc_meta.get("carries_bag", False),
                "summary": docs[0].content if docs else f"Person Track #{trk.track_number}",
            })

        logger.info(f"[TOOL] find_persons job={analysis_job_id} returned {len(results)} records.")
        return results

    find_person_tracks = find_persons  # Backward-compatible alias

    @classmethod
    async def find_vehicles(
        cls,
        db: AsyncSession,
        evidence_id: int,
        analysis_job_id: Optional[int] = None,
        color: Optional[str] = None,
        body_type: Optional[str] = None,
        license_plate: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Finds vehicle tracks (canonical taxonomy) matching attributes."""
        analysis_job_id = await _resolve_job(db, evidence_id, analysis_job_id)
        vehicle_classes = all_vehicle_classes()
        conds = _scope_filter(Track, evidence_id, analysis_job_id) + [
            func.lower(Track.class_name).in_([c.lower() for c in vehicle_classes]),
        ]
        stmt = select(Track).where(*conds)
        if start_time is not None:
            stmt = stmt.where(Track.last_seen_timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(Track.first_seen_timestamp <= end_time)

        tracks = (await db.execute(stmt)).scalars().all()
        results: List[Dict[str, Any]] = []
        for trk in tracks:
            doc_conds = _scope_filter(ForensicDocument, evidence_id, analysis_job_id) + [
                ForensicDocument.track_id == trk.track_number,
                ForensicDocument.document_type == DocumentType.TRACK,
            ]
            docs = (await db.execute(select(ForensicDocument).where(*doc_conds))).scalars().all()
            doc_meta = docs[0].metadata_json if docs else {}

            if color and doc_meta.get("vehicle_color") and color.lower() not in str(doc_meta["vehicle_color"]).lower():
                continue
            if body_type and doc_meta.get("vehicle_body_style") and body_type.lower() not in str(doc_meta["vehicle_body_style"]).lower():
                continue
            if license_plate and doc_meta.get("license_plate_number") and license_plate.upper() not in str(doc_meta["license_plate_number"]).upper():
                continue

            results.append({
                "track_id": trk.track_number,
                "class_name": trk.class_name,
                "start_time": trk.first_seen_timestamp,
                "end_time": trk.last_seen_timestamp,
                "duration": trk.duration,
                "vehicle_color": doc_meta.get("vehicle_color"),
                "vehicle_body_style": doc_meta.get("vehicle_body_style"),
                "license_plate_number": doc_meta.get("license_plate_number"),
                "summary": docs[0].content if docs else f"Vehicle Track #{trk.track_number}",
            })

        logger.info(f"[TOOL] find_vehicles job={analysis_job_id} returned {len(results)} records.")
        return results

    @classmethod
    async def get_behavioral_events(
        cls,
        db: AsyncSession,
        evidence_id: int,
        analysis_job_id: Optional[int] = None,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves stored behavioral events scoped to one analysis job."""
        analysis_job_id = await _resolve_job(db, evidence_id, analysis_job_id)
        conds = _scope_filter(ForensicDocument, evidence_id, analysis_job_id) + [
            ForensicDocument.document_type == DocumentType.BEHAVIORAL_EVENT,
        ]
        stmt = select(ForensicDocument).where(*conds)
        if start_time is not None:
            stmt = stmt.where(ForensicDocument.end_time >= start_time)
        if end_time is not None:
            stmt = stmt.where(ForensicDocument.start_time <= end_time)

        docs = (await db.execute(stmt)).scalars().all()
        results = []
        for d in docs:
            e_type = d.metadata_json.get("event_type", "BEHAVIORAL_EVENT")
            if event_type and event_type.lower() not in e_type.lower():
                continue
            results.append({
                "document_id": d.id,
                "title": d.title,
                "event_type": e_type,
                "summary": d.content,
                "start_time": d.start_time,
                "end_time": d.end_time,
                "involved_track_ids": d.metadata_json.get("involved_track_ids", []),
                "confidence": d.metadata_json.get("confidence", 0.90),
            })
        logger.info(f"[TOOL] get_behavioral_events job={analysis_job_id} returned {len(results)} events.")
        return results

    @classmethod
    async def search_evidence(
        cls,
        db: AsyncSession,
        evidence_id: int,
        query_text: str,
        top_k: int = 5,
        analysis_job_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs hybrid evidence search. analysis_job_id is accepted for interface
        parity; HybridSearchEngine already resolves the latest completed job
        internally and scopes its ChromaDB collection accordingly.
        """
        return await HybridSearchEngine.execute_search(
            db=db,
            evidence_id=evidence_id,
            query_text=query_text,
            top_k=top_k,
        )

    @classmethod
    async def get_track_timeline(
        cls,
        db: AsyncSession,
        evidence_id: int,
        track_id: int,
        analysis_job_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieves full chronological timeline of a track, scoped to one job."""
        analysis_job_id = await _resolve_job(db, evidence_id, analysis_job_id)
        t_conds = _scope_filter(Track, evidence_id, analysis_job_id) + [Track.track_number == track_id]
        trk = (await db.execute(select(Track).where(*t_conds))).scalars().first()
        if not trk:
            return {"error": f"Track #{track_id} not found."}

        d_conds = _scope_filter(Detection, evidence_id, analysis_job_id) + [Detection.track_id == track_id]
        detections = (await db.execute(select(Detection).where(*d_conds).order_by(Detection.timestamp))).scalars().all()

        i_conds = _scope_filter(PossibleInteraction, evidence_id, analysis_job_id) + [
            (PossibleInteraction.entity_a_track_id == track_id) | (PossibleInteraction.entity_b_track_id == track_id),
        ]
        interactions = (await db.execute(select(PossibleInteraction).where(*i_conds))).scalars().all()

        return {
            "track_id": trk.track_number,
            "class_name": trk.class_name,
            "first_seen": trk.first_seen_timestamp,
            "last_seen": trk.last_seen_timestamp,
            "duration": trk.duration,
            "detections_count": len(detections),
            "spatial_interactions": [
                {
                    "other_track_id": i.entity_b_track_id if i.entity_a_track_id == track_id else i.entity_a_track_id,
                    "start_time": i.start_time,
                    "end_time": i.end_time,
                    "label": i.label,
                }
                for i in interactions
            ],
        }
