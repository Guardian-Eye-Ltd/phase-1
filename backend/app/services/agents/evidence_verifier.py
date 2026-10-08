import logging
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.analysis import Track, Keyframe, Detection, AnalysisJob, JobStatus
from app.models.semantic import ForensicDocument
from app.services.agents.taxonomy import is_known_entity

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
        relevance_score: float = 0.0,
        analysis_job_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Track numbers are only unique within one run; without a job id,
        # "Track #5" could resolve to a different run's entity.
        if analysis_job_id is None:
            from app.services.analysis_jobs import get_active_analysis_job_id
            analysis_job_id = await get_active_analysis_job_id(db, evidence_id)

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
                select(Track).where(
                    Track.evidence_id == evidence_id,
                    Track.analysis_job_id == analysis_job_id,
                    Track.track_number == track_id,
                )
            )
            track_obj = t_res.scalars().first()

        # Check Keyframe existence
        kf_obj = None
        if keyframe_id is not None:
            k_res = await db.execute(
                select(Keyframe).where(
                    Keyframe.evidence_id == evidence_id,
                    Keyframe.analysis_job_id == analysis_job_id,
                    Keyframe.id == keyframe_id,
                )
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

    @classmethod
    async def verify_count_result(
        cls,
        db: AsyncSession,
        evidence_id: int,
        count_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deterministic verification for aggregation results. The count has
        already come from a trusted SQL aggregation; this validates that the
        inputs are well-formed and that no track id was counted twice.
        """
        job_id = count_result.get("analysis_job_id")
        entity = count_result.get("entity")
        aggregation = count_result.get("aggregation")

        checks: Dict[str, bool] = {
            "analysis_job_id_present": job_id is not None,
            "entity_known": bool(entity) and is_known_entity(entity),
            "aggregation_known": aggregation in ("DISTINCT_TRACK_COUNT", "DISTINCT_ACTIVE_TRACK_COUNT"),
            "evidence_basis_tracks": count_result.get("evidence_basis") == "TRACKS",
        }

        if job_id is not None:
            res = await db.execute(
                select(AnalysisJob).where(
                    AnalysisJob.id == job_id,
                    AnalysisJob.evidence_id == evidence_id,
                )
            )
            job = res.scalars().first()
            checks["job_belongs_to_evidence"] = job is not None
            checks["job_completed"] = job is not None and job.status == JobStatus.COMPLETED
        else:
            checks["job_belongs_to_evidence"] = False
            checks["job_completed"] = False

        track_ids = count_result.get("track_ids") or []
        checks["no_duplicate_track_ids"] = len(track_ids) == len(set(track_ids))

        status = "SUPPORTED" if all(checks.values()) else "WITHHELD"
        return {
            "status": status,
            "verification_type": "DETERMINISTIC_AGGREGATION",
            "evidence_basis": "TRACKS",
            "analysis_job_id": job_id,
            "checks": checks,
        }
