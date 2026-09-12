import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.analysis import (
    Track, Keyframe, ActivityInterval, PossibleInteraction, Detection, AnalysisJob
)
from app.models.semantic import (
    ForensicDocument, DocumentType, DocumentSourceType
)

logger = logging.getLogger(__name__)

class ForensicDocumentGenerator:
    """
    Generates evidence-grounded ForensicDocument records from Phase 1B structured observations.
    Never allows an LLM to invent descriptions without underlying structured DB evidence.
    """

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Helper to format seconds into MM:SS timestamp string."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    @classmethod
    async def generate_documents_for_job(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        analysis_job_id: int
    ) -> List[ForensicDocument]:
        """
        Extracts Phase 1B database observations and builds evidence-grounded ForensicDocument objects.
        """
        documents: List[ForensicDocument] = []

        # 1. Retrieve all tracks
        track_res = await db.execute(
            select(Track).where(Track.analysis_job_id == analysis_job_id)
        )
        tracks = track_res.scalars().all()

        # 2. Retrieve all keyframes
        kf_res = await db.execute(
            select(Keyframe).where(Keyframe.analysis_job_id == analysis_job_id)
        )
        keyframes = kf_res.scalars().all()

        # 3. Retrieve activity intervals
        act_res = await db.execute(
            select(ActivityInterval).where(ActivityInterval.analysis_job_id == analysis_job_id)
        )
        activity_intervals = act_res.scalars().all()

        # 4. Retrieve spatial interactions
        inter_res = await db.execute(
            select(PossibleInteraction).where(PossibleInteraction.analysis_job_id == analysis_job_id)
        )
        interactions = inter_res.scalars().all()

        # 5. Retrieve all detections for object associations
        det_res = await db.execute(
            select(Detection).where(Detection.analysis_job_id == analysis_job_id)
        )
        detections = det_res.scalars().all()

        # Group detections by track_id and timestamp
        track_detections: Dict[int, List[Detection]] = {}
        frame_detections: Dict[int, List[Detection]] = {}
        for det in detections:
            if det.track_id is not None:
                track_detections.setdefault(det.track_id, []).append(det)
            frame_detections.setdefault(det.frame_number, []).append(det)

        # -------------------------------------------------------------
        # Part A: Generate Track Documents
        # -------------------------------------------------------------
        for trk in tracks:
            associated_dets = track_detections.get(trk.track_number, [])
            associated_objects = list(set([
                d.class_name for d in associated_dets if d.class_name != trk.class_name
            ]))
            
            # Find keyframes linking to this track
            linked_kf_ids = []
            for kf in keyframes:
                if kf.track_ids and trk.track_number in kf.track_ids.get("tracks", []):
                    linked_kf_ids.append(f"KF-{kf.id}")

            # Find spatial relations for this track
            trk_interactions = [
                i for i in interactions 
                if i.entity_a_track_id == trk.track_number or i.entity_b_track_id == trk.track_number
            ]
            spatial_summary_lines = []
            for inter in trk_interactions:
                other_track = inter.entity_b_track_id if inter.entity_a_track_id == trk.track_number else inter.entity_a_track_id
                start_str = cls.format_timestamp(inter.start_time)
                end_str = cls.format_timestamp(inter.end_time)
                spatial_summary_lines.append(f"Near Track {other_track} ({start_str}–{end_str})")

            start_time_str = cls.format_timestamp(trk.first_seen_timestamp)
            end_time_str = cls.format_timestamp(trk.last_seen_timestamp)

            assoc_obj_text = f" The track was associated with {', '.join(associated_objects)} detections in multiple frames." if associated_objects else ""
            spatial_text = f" Spatial interactions noted: {'; '.join(spatial_summary_lines)}." if spatial_summary_lines else ""

            content_text = (
                f"{trk.class_name.capitalize()} Track {trk.track_number} was observed between {start_time_str} and {end_time_str} "
                f"(Duration: {int(trk.duration)}s, Observations: {trk.observation_count})."
                f"{assoc_obj_text}{spatial_text}"
            )

            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                track_id=trk.track_number,
                keyframe_id=None,
                document_type=DocumentType.TRACK,
                source_type=DocumentSourceType.OBSERVATION,
                title=f"Track {trk.track_number} ({trk.class_name.capitalize()})",
                content=content_text,
                start_time=trk.first_seen_timestamp,
                end_time=trk.last_seen_timestamp,
                metadata_json={
                    "track_number": trk.track_number,
                    "class_name": trk.class_name,
                    "duration": trk.duration,
                    "associated_objects": associated_objects,
                    "observation_count": trk.observation_count,
                    "spatial_interactions": spatial_summary_lines,
                    "linked_keyframes": linked_kf_ids
                }
            )
            documents.append(doc)

        # -------------------------------------------------------------
        # Part B: Generate Keyframe Documents
        # -------------------------------------------------------------
        for kf in keyframes:
            ts_str = cls.format_timestamp(kf.timestamp)
            frame_dets = frame_detections.get(kf.frame_number, [])
            det_summary = ", ".join([f"{d.class_name} (conf: {d.confidence:.2f})" for d in frame_dets]) or "General visual activity"

            kf_content = (
                f"At {ts_str} (Frame {kf.frame_number}), Keyframe KF-{kf.id} was extracted due to {kf.selection_reason.value}. "
                f"Detections visible: {det_summary}."
            )

            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                track_id=None,
                keyframe_id=kf.id,
                document_type=DocumentType.KEYFRAME,
                source_type=DocumentSourceType.OBSERVATION,
                title=f"Keyframe KF-{kf.id} @ {ts_str}",
                content=kf_content,
                start_time=kf.timestamp,
                end_time=kf.timestamp,
                metadata_json={
                    "keyframe_id": kf.id,
                    "frame_number": kf.frame_number,
                    "selection_reason": kf.selection_reason.value,
                    "image_path": kf.image_path,
                    "sha256_hash": kf.sha256_hash,
                    "detections": [d.class_name for d in frame_dets]
                }
            )
            documents.append(doc)

        # -------------------------------------------------------------
        # Part C: Generate Activity Interval Documents
        # -------------------------------------------------------------
        for act in activity_intervals:
            start_str = cls.format_timestamp(act.start_time)
            end_str = cls.format_timestamp(act.end_time)

            act_level_str = str(act.activity_level.value).upper()
            if "LOW" in act_level_str:
                activity_phrase = f"Low visual activity was detected between {start_str} and {end_str}"
            else:
                activity_phrase = f"Visual activity increased ({act.activity_level.value}) between {start_str} and {end_str}"

            act_content = (
                f"{activity_phrase} (Frames {act.start_frame}–{act.end_frame}) with motion score {act.motion_score:.2f}."
            )

            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                track_id=None,
                keyframe_id=None,
                document_type=DocumentType.ACTIVITY_INTERVAL,
                source_type=DocumentSourceType.OBSERVATION,
                title=f"Activity Interval {start_str}–{end_str}",
                content=act_content,
                start_time=act.start_time,
                end_time=act.end_time,
                metadata_json={
                    "start_frame": act.start_frame,
                    "end_frame": act.end_frame,
                    "activity_level": act.activity_level.value,
                    "motion_score": act.motion_score
                }
            )
            documents.append(doc)

        # -------------------------------------------------------------
        # Part D: Generate Spatial Interaction Documents
        # -------------------------------------------------------------
        for inter in interactions:
            start_str = cls.format_timestamp(inter.start_time)
            end_str = cls.format_timestamp(inter.end_time)

            inter_content = (
                f"Track {inter.entity_a_track_id} was observed in close spatial proximity near Track {inter.entity_b_track_id} "
                f"between {start_str} and {end_str} with interaction confidence score {inter.confidence_score:.2f}."
            )

            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                track_id=inter.entity_a_track_id,
                keyframe_id=None,
                document_type=DocumentType.SPATIAL_RELATION,
                source_type=DocumentSourceType.OBSERVATION,
                title=f"Spatial Interaction: Track {inter.entity_a_track_id} & Track {inter.entity_b_track_id}",
                content=inter_content,
                start_time=inter.start_time,
                end_time=inter.end_time,
                metadata_json={
                    "entity_a_track_id": inter.entity_a_track_id,
                    "entity_b_track_id": inter.entity_b_track_id,
                    "confidence_score": inter.confidence_score,
                    "label": inter.label
                }
            )
            documents.append(doc)

        # Save generated documents to DB
        for doc in documents:
            db.add(doc)
        await db.commit()

        logger.info(f"Generated {len(documents)} ForensicDocument records for evidence {evidence_id}, job {analysis_job_id}.")
        return documents
