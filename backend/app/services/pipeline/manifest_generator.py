import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

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
        events: Optional[List[Dict[str, Any]]] = None
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
                    "frames_processed": stats.get("total_frames_sampled", 0),
                    "detections": stats.get("total_detections", 0)
                },
                "tracker": {
                    "name": tracker_algorithm,
                    "status": "USED",
                    "tracks": stats.get("total_tracks", 0)
                },
                "open_vocab_detector": {
                    "name": "OwlViT",
                    "status": "NOT_USED",
                    "reason": "No query-driven open-vocab concept specified"
                },
                "vlm_reasoning": {
                    "name": "Ollama / Rule-Engine",
                    "status": "RULE_ENGINE_USED"
                }
            },

            "sampling": {
                "sampling_fps": sampling_fps,
                "confidence_threshold": confidence_threshold,
                "total_analysis_frames": stats.get("total_frames_sampled", 0)
            },

            "statistics": {
                "frames_processed": stats.get("total_frames_sampled", 0),
                "detections": stats.get("total_detections", 0),
                "tracks": stats.get("total_tracks", 0),
                "keyframes": stats.get("total_keyframes", 0),
                "events": stats.get("total_events", 0),
                "interactions": stats.get("total_interactions", 0),
                "activity_intervals": stats.get("total_activity_intervals", 0)
            },

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

        # Compute SHA-256 hash of deterministic JSON manifest
        json_bytes = json.dumps(manifest_data, sort_keys=True).encode('utf-8')
        manifest_hash = hashlib.sha256(json_bytes).hexdigest()

        return manifest_data, manifest_hash
