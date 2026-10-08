"""
Pipeline stage: extract face observations from person tracks.

Runs on a few well-chosen frames per person track (largest person box, spread
in time) and on the head region only, so CPU cost scales with the number of
people rather than the number of frames.
"""
import hashlib
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.models.face import FaceQuality
from app.services.face.crypto import encrypt_embedding
from app.services.face.engine import EMBEDDING_MODEL_ID, DetectedFace, FaceEngine, assess_quality

logger = logging.getLogger(__name__)

MIN_SECONDS_BETWEEN_SAMPLES = 0.5


def select_frames_for_track(
    detections: List[Dict[str, Any]], max_frames: int
) -> List[Dict[str, Any]]:
    """Largest person boxes first, at least MIN_SECONDS_BETWEEN_SAMPLES apart."""
    ranked = sorted(
        detections,
        key=lambda d: (d["bbox_y2"] - d["bbox_y1"]) * (d["bbox_x2"] - d["bbox_x1"]),
        reverse=True,
    )
    chosen: List[Dict[str, Any]] = []
    for d in ranked:
        if all(abs(d["timestamp"] - c["timestamp"]) >= MIN_SECONDS_BETWEEN_SAMPLES for c in chosen):
            chosen.append(d)
        if len(chosen) >= max_frames:
            break
    return sorted(chosen, key=lambda d: d["timestamp"])


def head_region(frame_shape: Tuple[int, int], det: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Upper ~45% of the person box plus a margin, in pixel coords."""
    h, w = frame_shape
    x1, y1 = det["bbox_x1"] * w, det["bbox_y1"] * h
    x2, y2 = det["bbox_x2"] * w, det["bbox_y2"] * h
    bw, bh = x2 - x1, y2 - y1
    rx1 = int(max(0, x1 - 0.15 * bw))
    rx2 = int(min(w, x2 + 0.15 * bw))
    ry1 = int(max(0, y1 - 0.10 * bh))
    ry2 = int(min(h, y1 + 0.45 * bh))
    return rx1, ry1, rx2, ry2


def _pick_face(faces: List[DetectedFace], region_w: int) -> Optional[DetectedFace]:
    """The face nearest the horizontal centre of the person's head region."""
    if not faces:
        return None
    cx = region_w / 2.0
    return min(faces, key=lambda f: (abs((f.bbox[0] + f.bbox[2]) / 2.0 - cx), -f.det_score))


def extract_faces(
    sampled_frames: List[Tuple[int, float, np.ndarray]],
    detections: List[Dict[str, Any]],
    evidence_id: int,
    analysis_job_id: int,
    derived_dir: str,
    detect_fn: Callable[[np.ndarray], List[DetectedFace]] = FaceEngine.detect,
) -> List[Dict[str, Any]]:
    """Return one observation dict per (person track, selected frame) with a face."""
    frames = {fn: frame for fn, _, frame in sampled_frames}
    by_track: Dict[int, List[Dict[str, Any]]] = {}
    for d in detections:
        if d.get("class_name") == "person" and d.get("track_id") is not None and d["frame_number"] in frames:
            by_track.setdefault(d["track_id"], []).append(d)

    out_dir = os.path.join(derived_dir, "faces", str(evidence_id))
    os.makedirs(out_dir, exist_ok=True)
    observations: List[Dict[str, Any]] = []

    for track_number, dets in by_track.items():
        for det in select_frames_for_track(dets, settings.FACE_MAX_FRAMES_PER_TRACK):
            frame = frames[det["frame_number"]]
            fh, fw = frame.shape[:2]
            rx1, ry1, rx2, ry2 = head_region((fh, fw), det)
            if rx2 - rx1 < 8 or ry2 - ry1 < 8:
                continue
            region = frame[ry1:ry2, rx1:rx2]

            face = _pick_face(detect_fn(region), rx2 - rx1)
            if face is None:
                continue

            # Back to source-frame pixel coordinates.
            fx1 = int(max(0, rx1 + face.bbox[0])); fy1 = int(max(0, ry1 + face.bbox[1]))
            fx2 = int(min(fw, rx1 + face.bbox[2])); fy2 = int(min(fh, ry1 + face.bbox[3]))
            width_px, height_px = fx2 - fx1, fy2 - fy1
            if width_px <= 0 or height_px <= 0:
                continue
            face_crop = frame[fy1:fy2, fx1:fx2]

            quality = assess_quality(face_crop, width_px, height_px, face.det_score)

            crop_name = f"face_ev{evidence_id}_job{analysis_job_id}_trk{track_number}_fn{det['frame_number']}.jpg"
            crop_path = os.path.join(out_dir, crop_name)
            cv2.imwrite(crop_path, face_crop)
            with open(crop_path, "rb") as f:
                crop_sha = hashlib.sha256(f.read()).hexdigest()

            observations.append({
                "track_number": track_number,
                "frame_number": det["frame_number"],
                "timestamp": det["timestamp"],
                "bbox": (fx1 / fw, fy1 / fh, fx2 / fw, fy2 / fh),
                "face_width_px": width_px,
                "face_height_px": height_px,
                "det_score": round(face.det_score, 4),
                "sharpness": quality.sharpness,
                "quality_status": quality.status,
                "quality_reasons": quality.reasons,
                # Low-quality faces are never compared, so their embeddings are not kept.
                "embedding_encrypted": (
                    encrypt_embedding(face.embedding) if quality.status == FaceQuality.USABLE else None
                ),
                "embedding_model": EMBEDDING_MODEL_ID,
                "embedding_dim": int(face.embedding.shape[0]),
                "crop_filename": crop_name,
                "crop_sha256": crop_sha,
            })

    usable = sum(1 for o in observations if o["quality_status"] == FaceQuality.USABLE)
    logger.info(
        f"[FACE] job={analysis_job_id} person_tracks={len(by_track)} "
        f"faces={len(observations)} usable={usable}"
    )
    return observations
