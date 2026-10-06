import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.analysis import Track, Keyframe, Detection, ActivityInterval, PossibleInteraction
from app.models.semantic import ForensicDocument, DocumentType
from app.services.semantic.hybrid_search_engine import HybridSearchEngine
from app.services.pipeline.event_engine import EventEngine

logger = logging.getLogger(__name__)

class InvestigationToolSystem:
    """
    Controlled tool system providing strict, evidence-grounded database queries for agents.
    Agents NEVER execute arbitrary SQL or raw shell commands. All inputs are sanitized.
    """

    @classmethod
    async def find_persons(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        upper_color: Optional[str] = None,
        lower_color: Optional[str] = None,
        has_bag: Optional[bool] = None,
        headwear: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Finds person tracks matching visual attributes or temporal window."""
        stmt = select(Track).where(Track.evidence_id == evidence_id, Track.class_name == "person")
        
        if start_time is not None:
            stmt = stmt.where(Track.last_seen_timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(Track.first_seen_timestamp <= end_time)

        res = await db.execute(stmt)
        tracks = res.scalars().all()

        results = []
        for trk in tracks:
            doc_res = await db.execute(
                select(ForensicDocument).where(
                    ForensicDocument.evidence_id == evidence_id,
                    ForensicDocument.track_id == trk.track_number,
                    ForensicDocument.document_type == DocumentType.TRACK
                )
            )
            docs = doc_res.scalars().all()
            doc_meta = docs[0].metadata_json if docs else {}

            if upper_color and doc_meta.get("upper_garment_color"):
                if upper_color.lower() not in str(doc_meta.get("upper_garment_color")).lower():
                    continue
            if lower_color and doc_meta.get("lower_garment_color"):
                if lower_color.lower() not in str(doc_meta.get("lower_garment_color")).lower():
                    continue
            if headwear and doc_meta.get("headwear"):
                if headwear.lower() not in str(doc_meta.get("headwear")).lower():
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
                "summary": docs[0].content if docs else f"Person Track #{trk.track_number}"
            })

        logger.info(f"[TOOL] find_persons returned {len(results)} records.")
        return results

    # Alias for backward compatibility
    find_person_tracks = find_persons

    @classmethod
    async def find_vehicles(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        color: Optional[str] = None,
        body_type: Optional[str] = None,
        license_plate: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Finds vehicle tracks (car, truck, bus, motorcycle, van) matching fine-grained vehicle attributes."""
        stmt = select(Track).where(
            Track.evidence_id == evidence_id,
            Track.class_name.in_(["car", "truck", "bus", "motorcycle", "van", "vehicle"])
        )
        if start_time is not None:
            stmt = stmt.where(Track.last_seen_timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(Track.first_seen_timestamp <= end_time)

        res = await db.execute(stmt)
        tracks = res.scalars().all()

        results = []
        for trk in tracks:
            doc_res = await db.execute(
                select(ForensicDocument).where(
                    ForensicDocument.evidence_id == evidence_id,
                    ForensicDocument.track_id == trk.track_number,
                    ForensicDocument.document_type == DocumentType.TRACK
                )
            )
            docs = doc_res.scalars().all()
            doc_meta = docs[0].metadata_json if docs else {}

            if color and doc_meta.get("vehicle_color"):
                if color.lower() not in str(doc_meta.get("vehicle_color")).lower():
                    continue
            if body_type and doc_meta.get("vehicle_body_style"):
                if body_type.lower() not in str(doc_meta.get("vehicle_body_style")).lower():
                    continue
            if license_plate and doc_meta.get("license_plate_number"):
                if license_plate.upper() not in str(doc_meta.get("license_plate_number")).upper():
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
                "summary": docs[0].content if docs else f"Vehicle Track #{trk.track_number}"
            })

        logger.info(f"[TOOL] find_vehicles returned {len(results)} records.")
        return results

    @classmethod
    async def get_behavioral_events(
        cls,
        db: AsyncSession,
        evidence_id: int,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves structured behavioral events (loitering, concealment, altercation, sudden acceleration)."""
        stmt = select(ForensicDocument).where(
            ForensicDocument.evidence_id == evidence_id,
            ForensicDocument.document_type == DocumentType.BEHAVIORAL_EVENT
        )
        if start_time is not None:
            stmt = stmt.where(ForensicDocument.end_time >= start_time)
        if end_time is not None:
            stmt = stmt.where(ForensicDocument.start_time <= end_time)

        res = await db.execute(stmt)
        docs = res.scalars().all()

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
                "confidence": d.metadata_json.get("confidence", 0.90)
            })
        logger.info(f"[TOOL] get_behavioral_events returned {len(results)} events.")
        return results

    @classmethod
    async def search_evidence(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        query_text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Runs hybrid evidence search engine."""
        return await HybridSearchEngine.execute_search(
            db=db,
            evidence_id=evidence_id,
            query_text=query_text,
            top_k=top_k
        )

    @classmethod
    async def get_track_timeline(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        track_id: int
    ) -> Dict[str, Any]:
        """Retrieves complete chronological timeline of observations for a track."""
        t_res = await db.execute(
            select(Track).where(Track.evidence_id == evidence_id, Track.track_number == track_id)
        )
        trk = t_res.scalars().first()
        if not trk:
            return {"error": f"Track #{track_id} not found."}

        det_res = await db.execute(
            select(Detection).where(
                Detection.evidence_id == evidence_id,
                Detection.track_id == track_id
            ).order_by(Detection.timestamp)
        )
        detections = det_res.scalars().all()

        inter_res = await db.execute(
            select(PossibleInteraction).where(
                PossibleInteraction.evidence_id == evidence_id,
                (PossibleInteraction.entity_a_track_id == track_id) | (PossibleInteraction.entity_b_track_id == track_id)
            )
        )
        interactions = inter_res.scalars().all()

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
                    "label": i.label
                }
                for i in interactions
            ]
        }
