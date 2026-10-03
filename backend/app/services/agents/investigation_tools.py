import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.analysis import Track, Keyframe, Detection, ActivityInterval, PossibleInteraction
from app.models.semantic import ForensicDocument, VLMObservation
from app.services.semantic.hybrid_search_engine import HybridSearchEngine

logger = logging.getLogger(__name__)

class InvestigationToolSystem:
    """
    Controlled tool system providing strict, evidence-grounded database queries for agents.
    Agents NEVER execute arbitrary SQL or raw shell commands. All inputs are sanitized.
    """

    @classmethod
    async def find_person_tracks(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        upper_color: Optional[str] = None,
        lower_color: Optional[str] = None,
        carries_bag: Optional[bool] = None,
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
                    ForensicDocument.track_id == trk.track_number
                )
            )
            docs = doc_res.scalars().all()
            doc_meta = docs[0].metadata_json if docs else {}

            # Filter attributes if specified
            if upper_color and doc_meta.get("upper_garment_color"):
                if upper_color.lower() not in str(doc_meta.get("upper_garment_color")).lower():
                    continue
            if lower_color and doc_meta.get("lower_garment_color"):
                if lower_color.lower() not in str(doc_meta.get("lower_garment_color")).lower():
                    continue
            if carries_bag is True and not doc_meta.get("carries_bag"):
                continue

            results.append({
                "track_id": trk.track_number,
                "class_name": trk.class_name,
                "start_time": trk.first_seen_timestamp,
                "end_time": trk.last_seen_timestamp,
                "duration": trk.duration,
                "upper_garment_color": doc_meta.get("upper_garment_color"),
                "lower_garment_color": doc_meta.get("lower_garment_color"),
                "carries_bag": doc_meta.get("carries_bag", False),
                "summary": docs[0].content if docs else f"Person Track #{trk.track_number}"
            })

        logger.info(f"[TOOL] find_person_tracks returned {len(results)} records.")
        return results

    @classmethod
    async def find_object_tracks(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        class_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Finds object tracks (car, backpack, bicycle, truck, etc.)."""
        stmt = select(Track).where(Track.evidence_id == evidence_id)
        if class_name:
            stmt = stmt.where(Track.class_name.ilike(f"%{class_name}%"))
        if start_time is not None:
            stmt = stmt.where(Track.last_seen_timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(Track.first_seen_timestamp <= end_time)

        res = await db.execute(stmt)
        tracks = res.scalars().all()

        results = []
        for trk in tracks:
            results.append({
                "track_id": trk.track_number,
                "class_name": trk.class_name,
                "start_time": trk.first_seen_timestamp,
                "end_time": trk.last_seen_timestamp,
                "duration": trk.duration,
                "observation_count": trk.observation_count
            })
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
