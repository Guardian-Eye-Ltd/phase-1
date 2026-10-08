import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from app.core.config import settings

class ManifestGenerator:
    """
    Generates a comprehensive, evidence-grounded machine-readable forensic manifest.
    Contains full structured arrays for detections, tracks, attributes, relationships, events, and keyframes.
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
        keyframes: List[Dict[str, Any]],
        detections: Optional[List[Dict[str, Any]]] = None,
        tracks: Optional[List[Dict[str, Any]]] = None,
        attributes: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        
        # Prepare structured detection records (capped at top 200 for clean manifest JSON size)
        formatted_detections = []
        if detections:
            for d in detections[:200]:
                formatted_detections.append({
                    "frame_number": d.get("frame_number"),
                    "timestamp": d.get("timestamp"),
                    "class": d.get("class_name"),
                    "confidence": d.get("confidence"),
                    "track_id": d.get("track_id"),
                    "bbox": {
                        "x1": d.get("bbox_x1"),
                        "y1": d.get("bbox_y1"),
                        "x2": d.get("bbox_x2"),
                        "y2": d.get("bbox_y2")
                    },
                    "model": model_name
                })

        # Prepare track summary records
        formatted_tracks = []
        if tracks:
            for t in tracks:
                formatted_tracks.append({
                    "track_id": t.get("track_number"),
                    "class": t.get("class_name"),
                    "first_seen": t.get("first_seen_timestamp"),
                    "last_seen": t.get("last_seen_timestamp"),
                    "duration": t.get("duration"),
                    "observation_count": t.get("observation_count"),
                    "attributes": {
                        "upper_garment": t.get("dominant_upper_color", "unknown"),
                        "lower_garment": t.get("dominant_lower_color", "unknown"),
                        "carries_bag": t.get("carries_bag", False)
                    }
                })

        manifest_data = {
            "analysis_job_id": analysis_job_id,
            "evidence_id": evidence_id,
            "source_sha256": source_sha256,
            "analysis_started": started_at.isoformat() if started_at else None,
            "analysis_completed": completed_at.isoformat() if completed_at else None,

            "models": {
                "detector": {
                    "name": model_name,
                    "version": model_version,
                    "status": "USED",
                    "processed_frames": stats.get("processed_frames"),
                    "stored_detections": stats.get("stored_detections"),
                },
                "tracker": {
                    "name": tracker_algorithm,
                    "status": "USED",
                    "unique_tracks": stats.get("unique_tracks"),
                },
            },

            # What this run could actually do, resolved at run time. Replaces a
            # previously hard-coded claim that a VLM/rule engine had been used.
            "capabilities": capabilities or {},

            "sampling": {
                "sampling_fps": sampling_fps,
                "confidence_threshold": confidence_threshold,
                "sampled_frames": stats.get("sampled_frames"),
            },

            # Canonical AnalysisStatistics snapshot at completion. Semantic
            # counts (VLM, documents) are filled in by indexing *after* the
            # manifest is sealed; the live statistics endpoint has them.
            "statistics": stats,

            "detections": formatted_detections,
            "tracks": formatted_tracks,
            "attributes": attributes or [],
            "relationships": relationships or [],
            "events": events or [],
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

        manifest_hash = hashlib.sha256(ManifestGenerator.canonical_bytes(manifest_data)).hexdigest()
        return manifest_data, manifest_hash

    @staticmethod
    def canonical_bytes(manifest_data: Dict[str, Any]) -> bytes:
        """The exact byte representation that is hashed and stored."""
        return json.dumps(manifest_data, sort_keys=True, default=str).encode("utf-8")

    @staticmethod
    def manifest_path(evidence_id: int, analysis_job_id: int) -> Path:
        return Path(settings.DERIVED_STORAGE_DIR) / "manifests" / f"ev{evidence_id}_job{analysis_job_id}.json"

    @staticmethod
    def save_manifest(path: Path, manifest_data: Dict[str, Any]) -> None:
        """
        Persist the sealed manifest. Re-generating later from the database is not
        equivalent — timestamps, ordering and run-time counters would differ and
        the hash would never match the one recorded on the job.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(ManifestGenerator.canonical_bytes(manifest_data))

    @staticmethod
    def load_and_verify(path: Path, recorded_hash: Optional[str]) -> Tuple[Dict[str, Any], str, bool]:
        """Return (manifest, sha256 of stored bytes, matches recorded hash)."""
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        return json.loads(raw), actual, bool(recorded_hash) and actual == recorded_hash
