import logging
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.analysis import Track, Keyframe, Detection
from app.models.semantic import ForensicDocument

logger = logging.getLogger(__name__)

class EvidenceVerifier:
    """
    Evidence Verification Engine evaluating finding claims against raw DB observations.
    Status output: SUPPORTED | PARTIALLY_SUPPORTED | UNVERIFIED | CONTRADICTED | WITHHELD
    Never allows an unverified AI hallucination to be output as fact.
    """

    @classmethod
    async def verify_finding(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        finding_statement: str,
        track_id: Optional[int] = None,
        keyframe_id: Optional[int] = None,
        relevance_score: float = 0.0
    ) -> Dict[str, Any]:
        
        verification = {
            "status": "UNVERIFIED",
            "confidence_score": 0.0,
            "evidence_id": evidence_id,
            "track_id": track_id,
            "keyframe_id": keyframe_id,
            "support_reason": "No matching database record found for claim.",
            "limitations": []
        }

        # Check Track existence
        track_obj = None
        if track_id is not None:
            t_res = await db.execute(
                select(Track).where(Track.evidence_id == evidence_id, Track.track_number == track_id)
            )
            track_obj = t_res.scalars().first()

        # Check Keyframe existence
        kf_obj = None
        if keyframe_id is not None:
            k_res = await db.execute(
                select(Keyframe).where(Keyframe.evidence_id == evidence_id, Keyframe.id == keyframe_id)
            )
            kf_obj = k_res.scalars().first()

        if not track_obj and not kf_obj:
            verification["status"] = "UNVERIFIED"
            verification["support_reason"] = "Claim references non-existent track and keyframe IDs."
            return verification

        # Check evidence groundings
        frame_count = track_obj.observation_count if track_obj else 1
        class_name = track_obj.class_name if track_obj else "object"

        # Contradiction Check
        statement_lower = finding_statement.lower()
        if track_obj:
            if "vehicle" in statement_lower or "car" in statement_lower:
                if class_name not in ["car", "truck", "bus", "motorcycle", "vehicle"]:
                    if "near" not in statement_lower and "toward" not in statement_lower:
                        verification["status"] = "CONTRADICTED"
                        verification["support_reason"] = f"Claim asserts entity is a vehicle, but DB Track #{track_id} is recorded as '{class_name}'."
                        return verification

        # Support Level Evaluation
        if relevance_score >= 0.70 and frame_count >= 5:
            verification["status"] = "SUPPORTED"
            verification["confidence_score"] = round(relevance_score, 2)
            verification["support_reason"] = (
                f"Fully supported by Track #{track_id} ({class_name}, {frame_count} observation frames, "
                f"duration: {track_obj.duration if track_obj else 'N/A'}s)."
            )
        elif relevance_score >= 0.40 or frame_count >= 1:
            verification["status"] = "PARTIALLY_SUPPORTED"
            verification["confidence_score"] = round(relevance_score, 2)
            verification["support_reason"] = (
                f"Partially supported by Track #{track_id or 'N/A'}. Visual association or proximity established, "
                f"but frame count or attribute certainty is moderate."
            )
            verification["limitations"].append("Visual attribute co-location is indirect or based on proximity heuristics.")
        else:
            verification["status"] = "UNVERIFIED"
            verification["confidence_score"] = round(relevance_score, 2)
            verification["support_reason"] = "Relevance score below required forensic verification threshold."
            verification["limitations"].append("Insufficient detection confidence or frame coverage.")

        return verification
