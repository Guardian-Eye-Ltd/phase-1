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
from app.services.pipeline.event_engine import EventEngine

logger = logging.getLogger(__name__)

class ForensicDocumentGenerator:
    """
    Generates structured, evidence-grounded ForensicDocument records from video observations.
    Synthesizes rich multi-modal surveillance narratives for semantic vector embedding.
    """

    @staticmethod
    def format_timestamp(seconds: Optional[float]) -> str:
        """Helper to format seconds into MM:SS timestamp string safely."""
        if seconds is None:
            return "00:00"
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
        Extracts observations from the database and builds structured narrative text
        suitable for sentence-transformer semantic embedding and hybrid search.
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

        # 5. Retrieve detections
        det_res = await db.execute(
            select(Detection).where(Detection.analysis_job_id == analysis_job_id)
        )
        detections = det_res.scalars().all()

        # Group detections by track_id and frame_number
        track_detections: Dict[int, List[Detection]] = {}
        frame_detections: Dict[int, List[Detection]] = {}
        all_det_dicts = []

        for det in detections:
            if det.track_id is not None:
                track_detections.setdefault(det.track_id, []).append(det)
            frame_detections.setdefault(det.frame_number, []).append(det)

            det_dict = {
                "track_id": det.track_id,
                "class_name": det.class_name,
                "timestamp": det.timestamp,
                "frame_number": det.frame_number,
                "bbox_x1": det.bbox_x1,
                "bbox_y1": det.bbox_y1,
                "bbox_x2": det.bbox_x2,
                "bbox_y2": det.bbox_y2,
                "upper_garment_color": getattr(det, 'upper_garment_color', 'unknown'),
                "lower_garment_color": getattr(det, 'lower_garment_color', 'unknown'),
                "upper_garment_type": getattr(det, 'upper_garment_type', 'unknown'),
                "lower_garment_type": getattr(det, 'lower_garment_type', 'unknown'),
                "headwear": getattr(det, 'headwear', 'unknown'),
                "vehicle_color": getattr(det, 'vehicle_color', 'unknown'),
                "vehicle_body_style": getattr(det, 'vehicle_body_style', 'unknown'),
                "license_plate_number": getattr(det, 'license_plate_number', None),
                "carries_bag": getattr(det, 'carries_bag', False),
                "keypoints": getattr(det, 'keypoints', None)
            }
            all_det_dicts.append(det_dict)

        # -----------------------------------------------------------------
        # Part A: Generate Track Narrative Documents
        # -----------------------------------------------------------------
        track_summaries_dicts = []
        for trk in tracks:
            associated_dets = track_detections.get(trk.track_number, [])
            associated_objects = list(set([
                d.class_name for d in associated_dets if d.class_name != trk.class_name
            ]))

            start_time_str = cls.format_timestamp(trk.first_seen_timestamp)
            end_time_str = cls.format_timestamp(trk.last_seen_timestamp)
            duration_sec = int(round(trk.duration)) if trk.duration else 0

            track_summaries_dicts.append({
                "track_number": trk.track_number,
                "class_name": trk.class_name,
                "first_seen_timestamp": trk.first_seen_timestamp,
                "last_seen_timestamp": trk.last_seen_timestamp,
                "first_seen_frame": trk.first_seen_frame,
                "last_seen_frame": trk.last_seen_frame,
                "duration": trk.duration,
                "observation_count": trk.observation_count
            })

            # Correlate spatial interactions with other tracks
            trk_interactions = [
                i for i in interactions 
                if i.entity_a_track_id == trk.track_number or i.entity_b_track_id == trk.track_number
            ]
            spatial_summary_lines = []
            for inter in trk_interactions:
                other_track = inter.entity_b_track_id if inter.entity_a_track_id == trk.track_number else inter.entity_a_track_id
                s_str = cls.format_timestamp(inter.start_time)
                e_str = cls.format_timestamp(inter.end_time)
                spatial_summary_lines.append(f"in spatial proximity with Track #{other_track} from {s_str} to {e_str}")

            # Extract attributes from detections safely
            upper_colors = [getattr(d, 'upper_garment_color', None) for d in associated_dets if getattr(d, 'upper_garment_color', None) and getattr(d, 'upper_garment_color', None) != 'unknown']
            lower_colors = [getattr(d, 'lower_garment_color', None) for d in associated_dets if getattr(d, 'lower_garment_color', None) and getattr(d, 'lower_garment_color', None) != 'unknown']
            upper_types = [getattr(d, 'upper_garment_type', None) for d in associated_dets if getattr(d, 'upper_garment_type', None) and getattr(d, 'upper_garment_type', None) != 'unknown']
            lower_types = [getattr(d, 'lower_garment_type', None) for d in associated_dets if getattr(d, 'lower_garment_type', None) and getattr(d, 'lower_garment_type', None) != 'unknown']
            headwears = [getattr(d, 'headwear', None) for d in associated_dets if getattr(d, 'headwear', None) and getattr(d, 'headwear', None) not in ['unknown', 'bare head']]
            veh_colors = [getattr(d, 'vehicle_color', None) for d in associated_dets if getattr(d, 'vehicle_color', None) and getattr(d, 'vehicle_color', None) != 'unknown']
            veh_styles = [getattr(d, 'vehicle_body_style', None) for d in associated_dets if getattr(d, 'vehicle_body_style', None) and getattr(d, 'vehicle_body_style', None) != 'unknown']
            plates = [getattr(d, 'license_plate_number', None) for d in associated_dets if getattr(d, 'license_plate_number', None)]

            carries_bag = any(getattr(d, 'carries_bag', False) for d in associated_dets) or any(obj in ["backpack", "handbag", "suitcase"] for obj in associated_objects)

            dom_upper_c = upper_colors[0] if upper_colors else None
            dom_lower_c = lower_colors[0] if lower_colors else None
            dom_upper_t = upper_types[0] if upper_types else None
            dom_lower_t = lower_types[0] if lower_types else None
            dom_headwear = headwears[0] if headwears else None
            dom_veh_c = veh_colors[0] if veh_colors else None
            dom_veh_s = veh_styles[0] if veh_styles else None
            license_plate = plates[0] if plates else None

            attr_parts = []
            if dom_upper_c or dom_upper_t:
                garment_desc = f"{dom_upper_c or ''} {dom_upper_t or 'upper garment'}".strip()
                attr_parts.append(f"wearing a {garment_desc}")
            if dom_lower_c or dom_lower_t:
                pants_desc = f"{dom_lower_c or ''} {dom_lower_t or 'lower garment'}".strip()
                attr_parts.append(f"wearing {pants_desc}")
            if dom_headwear:
                attr_parts.append(f"wearing {dom_headwear}")
            if carries_bag:
                attr_parts.append("carrying a bag or backpack")
            if dom_veh_c or dom_veh_s:
                veh_desc = f"{dom_veh_c or ''} {dom_veh_s or 'vehicle'}".strip()
                attr_parts.append(f"{veh_desc} body exterior")
            if license_plate:
                attr_parts.append(f"license plate number '{license_plate}'")

            attr_text = f" Visual attributes: {', '.join(attr_parts)}." if attr_parts else ""
            spatial_text = f" Spatial interactions: {'; '.join(spatial_summary_lines)}." if spatial_summary_lines else ""
            assoc_text = f" Associated object detections: {', '.join(associated_objects)}." if associated_objects else ""

            content_text = (
                f"A {trk.class_name} designated as Track #{trk.track_number} was observed on camera between "
                f"{start_time_str} and {end_time_str} (duration: {duration_sec}s across {trk.observation_count} frames)."
                f"{attr_text}{assoc_text}{spatial_text}"
            )

            title_text = f"{trk.class_name.capitalize()} Track #{trk.track_number} ({start_time_str} - {end_time_str})"

            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                track_id=trk.track_number,
                document_type=DocumentType.TRACK,
                source_type=DocumentSourceType.OBSERVATION,
                title=title_text,
                content=content_text,
                start_time=trk.first_seen_timestamp,
                end_time=trk.last_seen_timestamp,
                metadata_json={
                    "track_id": trk.track_number,
                    "class_name": trk.class_name,
                    "duration": trk.duration,
                    "upper_garment_color": dom_upper_c,
                    "lower_garment_color": dom_lower_c,
                    "upper_garment_type": dom_upper_t,
                    "lower_garment_type": dom_lower_t,
                    "headwear": dom_headwear,
                    "vehicle_color": dom_veh_c,
                    "vehicle_body_style": dom_veh_s,
                    "license_plate_number": license_plate,
                    "observation_count": trk.observation_count,
                    "associated_classes": associated_objects,
                    "carries_bag": carries_bag
                }
            )
            documents.append(doc)

        # -----------------------------------------------------------------
        # Part B: Generate Behavioral & Anomaly Event Documents
        # -----------------------------------------------------------------
        behavioral_events = EventEngine.detect_events(track_summaries_dicts, all_det_dicts)
        for ev in behavioral_events:
            ev_start_str = cls.format_timestamp(ev.get("start_time", 0.0))
            ev_end_str = cls.format_timestamp(ev.get("end_time", 0.0))
            event_type = ev.get("event_type", "BEHAVIORAL_EVENT")
            title = ev.get("title", f"Behavioral Event: {event_type}")
            desc = ev.get("description", "")
            inv_tracks = ev.get("involved_track_ids", [])

            content_text = (
                f"Behavioral anomaly ({event_type}) detected between {ev_start_str} and {ev_end_str}. "
                f"Details: {desc} Involved entities: Track IDs {inv_tracks}."
            )

            primary_track_id = inv_tracks[0] if inv_tracks else None

            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                track_id=primary_track_id,
                document_type=DocumentType.BEHAVIORAL_EVENT,
                source_type=DocumentSourceType.OBSERVATION,
                title=title,
                content=content_text,
                start_time=ev.get("start_time", 0.0),
                end_time=ev.get("end_time", 0.0),
                metadata_json={
                    "event_type": event_type,
                    "involved_track_ids": inv_tracks,
                    "confidence": ev.get("confidence", 0.90),
                    "event_metadata": ev.get("metadata", {})
                }
            )
            documents.append(doc)

        # -----------------------------------------------------------------
        # Part C: Generate Spatial Interaction Documents
        # -----------------------------------------------------------------
        for inter in interactions:
            s_str = cls.format_timestamp(inter.start_time)
            e_str = cls.format_timestamp(inter.end_time)
            inter_content = (
                f"Spatial proximity event detected between Track #{inter.entity_a_track_id} and Track #{inter.entity_b_track_id} "
                f"from {s_str} to {e_str}. The entities maintained close spatial proximity with confidence score of {inter.confidence_score:.2f}."
            )
            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                track_id=inter.entity_a_track_id,
                document_type=DocumentType.SPATIAL_RELATION,
                source_type=DocumentSourceType.OBSERVATION,
                title=f"Spatial Interaction: Track #{inter.entity_a_track_id} & #{inter.entity_b_track_id}",
                content=inter_content,
                start_time=inter.start_time,
                end_time=inter.end_time,
                metadata_json={
                    "entity_a": inter.entity_a_track_id,
                    "entity_b": inter.entity_b_track_id,
                    "distance": inter.min_distance_or_overlap
                }
            )
            documents.append(doc)

        # -----------------------------------------------------------------
        # Part D: Generate Keyframe Extraction Documents
        # -----------------------------------------------------------------
        for kf in keyframes:
            kf_time_str = cls.format_timestamp(kf.timestamp)
            visible_tracks = kf.track_ids.get("tracks", []) if kf.track_ids else []
            visible_classes = kf.detection_ids.get("classes", []) if kf.detection_ids else []
            reason_str = kf.selection_reason.value if hasattr(kf.selection_reason, 'value') else str(kf.selection_reason)
            
            content_desc = (
                f"Forensic Keyframe #{kf.frame_number} recorded at {kf_time_str}. "
                f"Reason for capture: {reason_str}. "
                f"Visible entities: {', '.join(visible_classes) if visible_classes else 'None'}. "
                f"Active track IDs: {visible_tracks}."
            )
            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                keyframe_id=kf.id,
                document_type=DocumentType.KEYFRAME,
                source_type=DocumentSourceType.OBSERVATION,
                title=f"Keyframe #{kf.frame_number} at {kf_time_str}",
                content=content_desc,
                start_time=kf.timestamp,
                end_time=kf.timestamp,
                metadata_json={
                    "frame_number": kf.frame_number,
                    "image_path": kf.image_path,
                    "sha256_hash": kf.sha256_hash,
                    "selection_reason": reason_str
                }
            )
            documents.append(doc)

        # -----------------------------------------------------------------
        # Part E: Generate Activity Interval Documents
        # -----------------------------------------------------------------
        for act in activity_intervals:
            start_str = cls.format_timestamp(act.start_time)
            end_str = cls.format_timestamp(act.end_time)
            act_level_str = act.activity_level.value if hasattr(act.activity_level, 'value') else str(act.activity_level)
            
            act_content = (
                f"Visual motion activity interval detected ({act_level_str}) between {start_str} and {end_str} "
                f"(Frames {act.start_frame}–{act.end_frame}) with motion score {act.motion_score:.2f}."
            )
            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                document_type=DocumentType.ACTIVITY_INTERVAL,
                source_type=DocumentSourceType.OBSERVATION,
                title=f"Activity Interval {start_str}–{end_str}",
                content=act_content,
                start_time=act.start_time,
                end_time=act.end_time,
                metadata_json={
                    "start_frame": act.start_frame,
                    "end_frame": act.end_frame,
                    "activity_level": act_level_str,
                    "motion_score": act.motion_score
                }
            )
            documents.append(doc)

        # Save all generated documents to DB
        db.add_all(documents)
        await db.commit()

        logger.info(f"[SEMANTIC] Generated {len(documents)} structured narrative forensic documents for job #{analysis_job_id}.")
        return documents
