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

    _vlm_model = None
    _vlm_processor = None
    _load_attempted = False
    _unavailable_reason: Optional[str] = None

    @classmethod
    def _initialize_model(cls):
        """
        Load the captioning model via its model classes rather than
        transformers.pipeline(): the "image-to-text" task name was removed in
        newer transformers releases, which silently disabled the VLM.
        """
        if cls._load_attempted:
            return
        cls._load_attempted = True
        if not settings.VLM_ENABLED:
            cls._unavailable_reason = "VLM disabled by configuration (VLM_ENABLED=false)."
            logger.info(f"[VLM] {cls._unavailable_reason}")
            return
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            logger.info(f"[VLM] Loading '{settings.VLM_MODEL_NAME}' on '{settings.DEVICE}'...")
            cls._vlm_processor = BlipProcessor.from_pretrained(settings.VLM_MODEL_NAME)
            cls._vlm_model = BlipForConditionalGeneration.from_pretrained(
                settings.VLM_MODEL_NAME
            ).to(settings.DEVICE)
            cls._vlm_model.eval()
            logger.info("[VLM] Captioning model loaded.")
        except Exception as e:
            cls._vlm_model = None
            cls._vlm_processor = None
            cls._unavailable_reason = f"Could not load '{settings.VLM_MODEL_NAME}': {type(e).__name__}: {e}"
            logger.warning(f"[VLM] {cls._unavailable_reason}. Keyframes will get detection summaries, not VLM descriptions.")

    @classmethod
    def is_available(cls) -> bool:
        cls._initialize_model()
        return cls._vlm_model is not None

    @classmethod
    def _caption(cls, image_path: str) -> str:
        import torch
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        inputs = cls._vlm_processor(images=image, return_tensors="pt").to(settings.DEVICE)
        with torch.no_grad():
            out = cls._vlm_model.generate(**inputs, max_new_tokens=40)
        return cls._vlm_processor.decode(out[0], skip_special_tokens=True).strip()

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
    def _resolve_keyframe_path(cls, api_path: str, evidence_id: int) -> str:
        """
        Resolves the actual filesystem path from a stored API URL path.
        Example: /api/v1/evidence/1/derived/keyframes/keyframe_ev1_fn0.jpg
              -> storage/derived/keyframes/1/keyframe_ev1_fn0.jpg
        """
        if not api_path:
            return ""
        try:
            # Extract filename from the API path
            filename = os.path.basename(api_path)
            resolved = os.path.join(
                settings.DERIVED_STORAGE_DIR,
                "keyframes",
                str(evidence_id),
                filename
            )
            return resolved
        except Exception as e:
            logger.warning(f"Could not resolve keyframe path '{api_path}': {e}")
            return ""

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


            actual_image_path = cls._resolve_keyframe_path(kf.image_path, evidence_id)

            vlm_description = ""
            if cls._vlm_model is not None and actual_image_path and os.path.exists(actual_image_path):
                try:
                    vlm_description = cls._caption(actual_image_path)
                except Exception as e:
                    logger.warning(f"VLM inference failed for keyframe {kf.id}: {e}")

            if vlm_description:
                sanitized_description = cls.sanitize_vlm_description(vlm_description)
                vlm_obs = VLMObservation(
                    evidence_id=evidence_id,
                    analysis_job_id=analysis_job_id,
                    keyframe_id=kf.id,
                    track_id=track_ids[0] if track_ids else None,
                    model_name=settings.VLM_MODEL_NAME,
                    model_version="1.0",
                    prompt_version="1.0",
                    description=sanitized_description,
                    # Captioning models expose no calibrated confidence; this is
                    # a fixed prior, labelled as such in metadata below.
                    confidence=0.85,
                    is_safe=True
                )
                db.add(vlm_obs)
                vlm_observations.append(vlm_obs)
                db.add(ForensicDocument(
                    evidence_id=evidence_id,
                    analysis_job_id=analysis_job_id,
                    track_id=track_ids[0] if track_ids else None,
                    keyframe_id=kf.id,
                    document_type=DocumentType.KEYFRAME,
                    source_type=DocumentSourceType.VLM_OBSERVATION,
                    title=f"VLM Description KF-{kf.id}",
                    content=f"[VLM Visual Observation] At {kf.timestamp:.2f}s (Keyframe KF-{kf.id}): {sanitized_description}",
                    start_time=kf.timestamp,
                    end_time=kf.timestamp,
                    metadata_json={
                        "keyframe_id": kf.id,
                        "frame_number": kf.frame_number,
                        "vlm_model": settings.VLM_MODEL_NAME,
                        "confidence": 0.85,
                        "confidence_kind": "FIXED_PRIOR",
                        "is_vlm": True,
                    }
                ))
            else:
                # No VLM output. Summarise what the detector stored for this
                # frame, and say so — this is NOT a visual description and must
                # never be recorded as a VLM observation.
                counts: Dict[str, int] = {}
                for c in det_classes:
                    counts[c] = counts.get(c, 0) + 1
                det_summary = ", ".join(f"{n} {c}(s)" for c, n in counts.items()) or "no detected objects"
                track_str = f" Tracks: {', '.join(str(t) for t in track_ids)}." if track_ids else ""
                db.add(ForensicDocument(
                    evidence_id=evidence_id,
                    analysis_job_id=analysis_job_id,
                    track_id=track_ids[0] if track_ids else None,
                    keyframe_id=kf.id,
                    document_type=DocumentType.KEYFRAME,
                    source_type=DocumentSourceType.SYSTEM_GENERATED,
                    title=f"Detection Summary KF-{kf.id}",
                    content=(
                        f"[Detection Summary — not a VLM description] At {kf.timestamp:.2f}s "
                        f"(Keyframe KF-{kf.id}) the detector recorded {det_summary}.{track_str}"
                    ),
                    start_time=kf.timestamp,
                    end_time=kf.timestamp,
                    metadata_json={
                        "keyframe_id": kf.id,
                        "frame_number": kf.frame_number,
                        "generator": "DETECTION_SUMMARY_TEMPLATE",
                        "vlm_unavailable_reason": cls._unavailable_reason,
                        "is_vlm": False,
                    }
                ))

        await db.commit()
        logger.info(
            f"[VLM] job={analysis_job_id} real VLM observations={len(vlm_observations)} "
            f"of {len(keyframes)} keyframes"
            + ("" if cls._vlm_model is not None else f" (VLM unavailable: {cls._unavailable_reason})")
        )
        return vlm_observations
