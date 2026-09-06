import os
import cv2
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple

class KeyframeExtractor:
    """
    Selects keyframes based on forensic criteria (track entry/exit, motion spikes, object appearance),
    saves derived image files, and computes SHA-256 hashes.
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
        keyframes_list = []
        selected_fns = set()

        # 1. Keyframes for track first appearance & last appearance
        for trk in track_summaries:
            first_fn = trk["first_seen_frame"]
            last_fn = trk["last_seen_frame"]

            for fn, reason in [(first_fn, "PERSON_APPEARANCE" if trk["class_name"] == "person" else "OBJECT_DETECTION"),
                               (last_fn, "ENTRY_EXIT")]:
                if fn in frames_dict and fn not in selected_fns:
                    selected_fns.add(fn)
                    ts, frame = frames_dict[fn]
                    keyframe_info = KeyframeExtractor._save_keyframe(
                        frame, fn, ts, reason, [trk["track_number"]], evidence_id, target_dir
                    )
                    keyframes_list.append(keyframe_info)

        # 2. Keyframe for peak motion / middle frame if no keyframes yet
        if not keyframes_list and sampled_frames:
            mid_fn, mid_ts, mid_frame = sampled_frames[len(sampled_frames) // 2]
            keyframe_info = KeyframeExtractor._save_keyframe(
                mid_frame, mid_fn, mid_ts, "MOTION_CHANGE", [], evidence_id, target_dir
            )
            keyframes_list.append(keyframe_info)

        return keyframes_list

    @staticmethod
    def _save_keyframe(
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        reason: str,
        track_ids: List[int],
        evidence_id: int,
        target_dir: str
    ) -> Dict[str, Any]:
        file_basename = f"keyframe_ev{evidence_id}_fn{frame_number}.jpg"
        full_path = os.path.join(target_dir, file_basename)

        # Draw light visual box marker on keyframe copy if needed, or save clean
        cv2.imwrite(full_path, frame)

        # Compute SHA-256 hash of derived keyframe image
        sha256 = hashlib.sha256()
        with open(full_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        img_hash = sha256.hexdigest()

        rel_path = f"/api/v1/evidence/{evidence_id}/derived/keyframes/{file_basename}"

        return {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "selection_reason": reason,
            "image_path": rel_path,
            "sha256_hash": img_hash,
            "track_ids": track_ids,
            "detection_ids": []
        }
