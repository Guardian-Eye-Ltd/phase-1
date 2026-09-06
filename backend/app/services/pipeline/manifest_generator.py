import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple

class ManifestGenerator:
    """
    Generates a machine-readable, cryptographic analysis manifest for digital evidence provenance.
    """
    @staticmethod
    def generate_manifest(
        evidence_id: int,
        analysis_job_id: int,
        source_sha256: str,
        started_at: datetime,
        completed_at: datetime,
        model_name: str,
        model_version: str,
        tracker_algorithm: str,
        sampling_fps: float,
        confidence_threshold: float,
        stats: Dict[str, Any],
        keyframes: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        manifest_data = {
            "evidence_id": evidence_id,
            "analysis_job_id": analysis_job_id,
            "source_sha256": source_sha256,
            "analysis_started": started_at.isoformat() if started_at else None,
            "analysis_completed": completed_at.isoformat() if completed_at else None,
            "models": {
                "object_detector": model_name,
                "model_version": model_version,
                "tracker": tracker_algorithm
            },
            "sampling_configuration": {
                "fps": sampling_fps,
                "confidence_threshold": confidence_threshold
            },
            "statistics": stats,
            "derived_artifacts": [
                {
                    "frame_number": k["frame_number"],
                    "timestamp": k["timestamp"],
                    "selection_reason": k["selection_reason"],
                    "image_path": k["image_path"],
                    "sha256_hash": k["sha256_hash"]
                }
                for k in keyframes
            ]
        }

        # Compute SHA-256 hash of deterministic JSON manifest
        json_bytes = json.dumps(manifest_data, sort_keys=True).encode('utf-8')
        manifest_hash = hashlib.sha256(json_bytes).hexdigest()

        return manifest_data, manifest_hash
