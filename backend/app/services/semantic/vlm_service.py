import logging
import os
import re
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.analysis import Keyframe, Detection, Track
from app.models.semantic import VLMObservation, ForensicDocument, DocumentType, DocumentSourceType

logger = logging.getLogger(__name__)

# Sensitive terms safety filter list
UNSAFE_SUBJECTIVE_PATTERNS = [
    r"\bsuspicious\b", r"\bguilty\b", r"\bcriminal\b", r"\bangry\b", r"\bhostile\b",
    r"\bterrorist\b", r"\bshady\b", r"\bdangerous\b", r"\bevil\b", r"\bmalicious\b",
    r"\brace\b", r"\bethnicity\b", r"\breligion\b", r"\bpolitical\b"
]

class VisionLanguageService:
    """
    Vision-Language Model Service for analyzing forensic keyframes.
    Enforces forensic safety rules (no subjective or sensitive claims) and outputs VLM_OBSERVATION records.
    """

    _vlm_pipeline = None
    _vlm_loaded = False

    @classmethod
    def _initialize_model(cls):
        """Attempts to load the configured open-source VLM (BLIP/Transformers) if enabled."""
        if cls._vlm_loaded:
            return
        if not settings.VLM_ENABLED:
            logger.info("VLM is disabled in settings (VLM_ENABLED=False).")
            return

        try:
            from transformers import pipeline
            logger.info(f"Loading VLM model '{settings.VLM_MODEL_NAME}' on device '{settings.DEVICE}'...")
            cls._vlm_pipeline = pipeline(
                "image-to-text", 
                model=settings.VLM_MODEL_NAME, 
                device=-1 if settings.DEVICE == "cpu" else 0
            )
            cls._vlm_loaded = True
            logger.info("VLM model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize VLM pipeline '{settings.VLM_MODEL_NAME}': {e}. Using heuristic VLM fallback.")
            cls._vlm_pipeline = None
            cls._vlm_loaded = False

    @classmethod
    def sanitize_vlm_description(cls, text: str) -> str:
        """
        Sanitizes text according to Forensic VLM Safety Rules.
        Removes subjective intent, emotional state, or sensitive personal attributes.
        """
        sanitized = text
        for pattern in UNSAFE_SUBJECTIVE_PATTERNS:
            sanitized = re.sub(pattern, "observed entity", sanitized, flags=re.IGNORECASE)
        
        # Clean up double spaces
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    @classmethod
    async def analyze_keyframes_for_job(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        analysis_job_id: int
    ) -> List[VLMObservation]:
        """
        Analyzes keyframes for a given analysis job using VLM or heuristic fallback.
        """
        cls._initialize_model()

        # Retrieve keyframes (capped by settings.MAX_KEYFRAMES_FOR_VLM)
        kf_res = await db.execute(
            select(Keyframe)
            .where(Keyframe.analysis_job_id == analysis_job_id)
            .order_by(Keyframe.timestamp)
            .limit(settings.MAX_KEYFRAMES_FOR_VLM)
        )
        keyframes = kf_res.scalars().all()

        vlm_observations: List[VLMObservation] = []

        for kf in keyframes:
            # Check associated frame detections for fallback context
            det_res = await db.execute(
                select(Detection).where(
                    Detection.analysis_job_id == analysis_job_id,
                    Detection.frame_number == kf.frame_number
                )
            )
            frame_dets = det_res.scalars().all()
            det_classes = [d.class_name for d in frame_dets]
            track_ids = list(set([d.track_id for d in frame_dets if d.track_id is not None]))

            raw_description = ""
            confidence = 0.85

            # If VLM is loaded and image file exists, run inference
            if cls._vlm_pipeline and os.path.exists(kf.image_path):
                try:
                    from PIL import Image
                    image = Image.open(kf.image_path).convert("RGB")
                    vlm_out = cls._vlm_pipeline(image)
                    if vlm_out and len(vlm_out) > 0 and "generated_text" in vlm_out[0]:
                        raw_description = vlm_out[0]["generated_text"]
                except Exception as e:
                    logger.warning(f"VLM inference failed for keyframe {kf.id}: {e}")

            # Fallback visual description generator if VLM is unavailable or returned empty
            if not raw_description:
                counts = {}
                for c in det_classes:
                    counts[c] = counts.get(c, 0) + 1
                det_summary = ", ".join([f"{count} {cls_name}(s)" for cls_name, count in counts.items()]) or "objects"
                
                track_str = f" Associated Track(s): {', '.join([str(t) for t in track_ids])}." if track_ids else ""
                raw_description = (
                    f"A scene containing {det_summary} recorded at timestamp {kf.timestamp:.2f}s.{track_str}"
                )
                confidence = 0.80

            sanitized_description = cls.sanitize_vlm_description(raw_description)

            # Store VLM Observation
            vlm_obs = VLMObservation(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                keyframe_id=kf.id,
                track_id=track_ids[0] if track_ids else None,
                model_name=settings.VLM_MODEL_NAME if cls._vlm_pipeline else "VLM_Heuristic_Fallback",
                model_version="1.0",
                prompt_version="1.0",
                description=sanitized_description,
                confidence=confidence,
                is_safe=True
            )
            db.add(vlm_obs)
            vlm_observations.append(vlm_obs)

            # Create corresponding ForensicDocument for semantic search index
            doc_content = (
                f"[VLM Visual Observation] At {kf.timestamp:.2f}s (Keyframe KF-{kf.id}): {sanitized_description}"
            )
            doc = ForensicDocument(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                track_id=track_ids[0] if track_ids else None,
                keyframe_id=kf.id,
                document_type=DocumentType.KEYFRAME,
                source_type=DocumentSourceType.VLM_OBSERVATION,
                title=f"VLM Description KF-{kf.id}",
                content=doc_content,
                start_time=kf.timestamp,
                end_time=kf.timestamp,
                metadata_json={
                    "keyframe_id": kf.id,
                    "frame_number": kf.frame_number,
                    "vlm_model": settings.VLM_MODEL_NAME if cls._vlm_pipeline else "Fallback",
                    "confidence": confidence,
                    "is_vlm": True
                }
            )
            db.add(doc)

        await db.commit()
        logger.info(f"Generated {len(vlm_observations)} VLM observations for evidence {evidence_id}.")
        return vlm_observations
