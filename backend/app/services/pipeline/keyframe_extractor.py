import os
import cv2
import hashlib
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class KeyframeExtractor:
    """
    Selects keyframes based on forensic criteria (track entry/exit, motion spikes, object appearance, periodic interval),
    saves derived image files, populates track & detection references, and computes SHA-256 hashes.
    """
    @staticmethod
    def extract_keyframes(
        sampled_frames: List[Tuple[int, float, np.ndarray]],
        all_detections: List[Dict[str, Any]],
        track_summaries: List[Dict[str, Any]],
        evidence_id: int,
        derived_dir: str
    ) -> List[Dict[str, Any]]:
        if not sampled_frames:
            return []

        target_dir = os.path.join(derived_dir, "keyframes", str(evidence_id))
        os.makedirs(target_dir, exist_ok=True)

        frames_dict = {fn: (ts, frame) for fn, ts, frame in sampled_frames}
        
        # Group detections by frame number
        detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for det in all_detections:
            fn = det["frame_number"]
            detections_by_frame.setdefault(fn, []).append(det)

        keyframes_list = []
        selected_fns = set()

        # 1. Keyframes for track entry & exit
        for trk in track_summaries:
            first_fn = trk["first_seen_frame"]
            last_fn = trk["last_seen_frame"]
            cls_name = trk["class_name"]

            # First appearance keyframe
            if first_fn in frames_dict and first_fn not in selected_fns:
                selected_fns.add(first_fn)
                ts, frame = frames_dict[first_fn]
                reason = "PERSON_APPEARANCE" if cls_name == "person" else "OBJECT_DETECTION"
                kf = KeyframeExtractor._save_keyframe(
                    frame, first_fn, ts, reason, evidence_id, target_dir, detections_by_frame.get(first_fn, [])
                )
                keyframes_list.append(kf)

            # Last appearance / exit keyframe
            if last_fn in frames_dict and last_fn not in selected_fns:
                selected_fns.add(last_fn)
                ts, frame = frames_dict[last_fn]
                kf = KeyframeExtractor._save_keyframe(
                    frame, last_fn, ts, "ENTRY_EXIT", evidence_id, target_dir, detections_by_frame.get(last_fn, [])
                )
                keyframes_list.append(kf)

        # 2. Periodic keyframes to ensure coverage if gap exceeds KEYFRAME_INTERVAL
        if sampled_frames:
            sorted_sampled = sorted(sampled_frames, key=lambda x: x[1])
            last_kf_ts = -999.0
            
            for fn, ts, frame in sorted_sampled:
                if (ts - last_kf_ts) >= settings.KEYFRAME_INTERVAL:
                    if fn not in selected_fns:
                        selected_fns.add(fn)
                        kf = KeyframeExtractor._save_keyframe(
                            frame, fn, ts, "MOTION_CHANGE", evidence_id, target_dir, detections_by_frame.get(fn, [])
                        )
                        keyframes_list.append(kf)
                        last_kf_ts = ts
                elif fn in selected_fns:
                    last_kf_ts = ts

        # 3. Fallback middle frame if no keyframes selected
        if not keyframes_list and sampled_frames:
            mid_fn, mid_ts, mid_frame = sampled_frames[len(sampled_frames) // 2]
            kf = KeyframeExtractor._save_keyframe(
                mid_frame, mid_fn, mid_ts, "MOTION_CHANGE", evidence_id, target_dir, detections_by_frame.get(mid_fn, [])
            )
            keyframes_list.append(kf)

        # Deduplicate strictly on frame_number and sort keyframes chronologically
        seen_fns = set()
        unique_keyframes = []
        for kf in keyframes_list:
            if kf["frame_number"] not in seen_fns:
                seen_fns.add(kf["frame_number"])
                unique_keyframes.append(kf)

        unique_keyframes.sort(key=lambda x: x["timestamp"])

        logger.info(f"[KEYFRAME] Extracted {len(unique_keyframes)} unique forensic keyframes for evidence {evidence_id}.")
        return unique_keyframes

    @staticmethod
    def _save_keyframe(
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        reason: str,
        evidence_id: int,
        target_dir: str,
        frame_detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        file_basename = f"keyframe_ev{evidence_id}_fn{frame_number}.jpg"
        full_path = os.path.join(target_dir, file_basename)

        # Save clean keyframe image file
        cv2.imwrite(full_path, frame)

        # Compute SHA-256 hash of derived keyframe image
        sha256 = hashlib.sha256()
        with open(full_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        img_hash = sha256.hexdigest()

        rel_path = f"/api/v1/evidence/{evidence_id}/derived/keyframes/{file_basename}"

        # Collect track_ids and detection details visible in this frame
        track_ids = list(set([d["track_id"] for d in frame_detections if d.get("track_id") is not None]))
        detection_classes = [d["class_name"] for d in frame_detections]

        return {
            "keyframe_id": None, # assigned after DB save if needed
            "evidence_id": evidence_id,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "selection_reason": reason,
            "image_path": rel_path,
            "sha256_hash": img_hash,
            "track_ids": {"tracks": track_ids},
            "detection_ids": {"classes": detection_classes, "count": len(frame_detections)}
        }
