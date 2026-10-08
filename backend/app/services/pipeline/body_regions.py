"""
Where on a person (or vehicle) to read an attribute from.

Person regions come from YOLO-pose keypoints when the relevant joints are
actually visible; otherwise the region is reported as not visible instead of
falling back to a guess about where clothes "should" be. Vehicle colour is read
from the body band below the windscreen and above the wheels.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# COCO keypoint indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14
L_ANKLE, R_ANKLE = 15, 16
NOSE, L_EYE, R_EYE = 0, 1, 2
KP_MIN_CONF = 0.5

Box = Tuple[float, float, float, float]


@dataclass
class PersonRegions:
    upper: Optional[Box]
    lower: Optional[Box]
    head: Optional[Box]
    source: str               # "POSE" | "BOX"
    upper_reason: str = ""
    lower_reason: str = ""


def _visible(kps: np.ndarray, conf: Optional[np.ndarray], idx: int) -> bool:
    if conf is not None:
        return conf[idx] >= KP_MIN_CONF
    return kps[idx][0] > 0 and kps[idx][1] > 0


def _shrink(box: Box, fx: float, fy: float) -> Box:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * fx, (y2 - y1) * fy
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def person_regions(
    person_box_px: Box,
    keypoints_px: Optional[np.ndarray] = None,
    keypoint_conf: Optional[np.ndarray] = None,
) -> PersonRegions:
    """
    keypoints_px: (17, 2) pixel coords for this person, or None.
    """
    bx1, by1, bx2, by2 = person_box_px
    bw, bh = bx2 - bx1, by2 - by1

    if keypoints_px is None or len(keypoints_px) < 17:
        # No pose: torso band only. Lower body is not assumed to be visible —
        # a box cut by water, a desk or the frame edge would otherwise yield a
        # colour for legs that are not in view.
        upper = _shrink((bx1, by1 + 0.20 * bh, bx2, by1 + 0.50 * bh), 0.20, 0.05)
        head = (bx1 + 0.2 * bw, by1, bx2 - 0.2 * bw, by1 + 0.18 * bh)
        return PersonRegions(upper, None, head, "BOX", lower_reason="no pose keypoints; legs not verified as visible")

    vis = lambda i: _visible(keypoints_px, keypoint_conf, i)
    k = keypoints_px

    # Head: around nose/eyes when visible.
    head = None
    face_pts = [k[i] for i in (NOSE, L_EYE, R_EYE) if vis(i)]
    if face_pts:
        fx = np.mean([p[0] for p in face_pts]); fy = np.mean([p[1] for p in face_pts])
        hs = max(0.18 * bw, 12)
        head = (fx - hs, fy - 1.4 * hs, fx + hs, fy + 0.6 * hs)

    # Upper garment: shoulders down to hips (or a torso-length estimate).
    upper, upper_reason = None, ""
    if vis(L_SHOULDER) and vis(R_SHOULDER):
        sx1, sx2 = sorted([k[L_SHOULDER][0], k[R_SHOULDER][0]])
        sy = (k[L_SHOULDER][1] + k[R_SHOULDER][1]) / 2
        shoulder_w = max(sx2 - sx1, 0.25 * bw)
        if vis(L_HIP) and vis(R_HIP):
            hy = (k[L_HIP][1] + k[R_HIP][1]) / 2
        else:
            hy = sy + 1.3 * shoulder_w
        cx = (sx1 + sx2) / 2
        upper = _shrink((cx - shoulder_w / 2, sy, cx + shoulder_w / 2, min(hy, by2)), 0.12, 0.12)
    else:
        upper_reason = "shoulders not visible"

    # Lower garment: prefer knee-to-ankle (long tops and coats cover the thighs),
    # else hip-to-knee. Only when those joints are actually visible.
    lower, lower_reason = None, ""
    knees = [k[i] for i in (L_KNEE, R_KNEE) if vis(i)]
    ankles = [k[i] for i in (L_ANKLE, R_ANKLE) if vis(i)]
    if knees and ankles:
        xs = [p[0] for p in knees + ankles]
        ky, ay = float(np.mean([p[1] for p in knees])), float(np.mean([p[1] for p in ankles]))
        width = max(max(xs) - min(xs), 0.18 * bw)
        cx = (max(xs) + min(xs)) / 2
        if ay - ky > 4:
            lower = _shrink((cx - width / 2, ky, cx + width / 2, ay), 0.10, 0.10)
    if lower is None and vis(L_HIP) and vis(R_HIP) and knees:
        hx1, hx2 = sorted([k[L_HIP][0], k[R_HIP][0]])
        hy = (k[L_HIP][1] + k[R_HIP][1]) / 2
        ky = float(np.mean([p[1] for p in knees]))
        hip_w = max(hx2 - hx1, 0.2 * bw)
        cx = (hx1 + hx2) / 2
        if ky - hy > 4:
            # Lower half of the thigh, below where most tops end.
            lower = _shrink((cx - hip_w / 2, hy + 0.4 * (ky - hy), cx + hip_w / 2, ky), 0.10, 0.05)
    if lower is None:
        lower_reason = "knees/ankles not visible"

    return PersonRegions(upper, lower, head, "POSE", upper_reason, lower_reason)


def face_skin_patch(keypoints_px: Optional[np.ndarray], keypoint_conf: Optional[np.ndarray]) -> Optional[Box]:
    """Small cheek/nose patch for sampling the person's own skin tone."""
    if keypoints_px is None or len(keypoints_px) < 17:
        return None
    vis = lambda i: _visible(keypoints_px, keypoint_conf, i)
    if not (vis(NOSE) and vis(L_EYE) and vis(R_EYE)):
        return None
    nose, le, re = keypoints_px[NOSE], keypoints_px[L_EYE], keypoints_px[R_EYE]
    eye_d = float(np.hypot(le[0] - re[0], le[1] - re[1]))
    if eye_d < 6:
        return None
    return (nose[0] - 0.6 * eye_d, nose[1] - 0.4 * eye_d, nose[0] + 0.6 * eye_d, nose[1] + 0.3 * eye_d)


def vehicle_body_region(vehicle_box_px: Box) -> Box:
    """Below the windscreen, above the wheels/road, away from the side edges."""
    x1, y1, x2, y2 = vehicle_box_px
    w, h = x2 - x1, y2 - y1
    return (x1 + 0.20 * w, y1 + 0.55 * h, x2 - 0.20 * w, y1 + 0.85 * h)


def match_pose_to_boxes(
    person_boxes: List[Box], pose_boxes: List[Box], min_iou: float = 0.45
) -> List[Optional[int]]:
    """Greedy IoU assignment of pose detections to person detections."""
    def iou(a: Box, b: Box) -> float:
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    pairs = sorted(
        ((iou(p, q), i, j) for i, p in enumerate(person_boxes) for j, q in enumerate(pose_boxes)),
        reverse=True,
    )
    assignment: List[Optional[int]] = [None] * len(person_boxes)
    used = set()
    for score, i, j in pairs:
        if score < min_iou:
            break
        if assignment[i] is None and j not in used:
            assignment[i] = j
            used.add(j)
    return assignment
