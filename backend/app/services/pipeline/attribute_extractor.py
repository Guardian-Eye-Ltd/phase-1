"""
Per-detection visual attributes for people and vehicles.

Design rules (each fixes a failure seen on real footage):
  * A region that cannot be seen produces no claim. Shirtless swimmers were
    previously reported in "brown shirts" (skin) and "white trousers" (water).
  * Garment colour is read from pose-guided torso/leg regions with this
    person's own face tone as the skin reference, using deterministic colour
    naming rather than a closed-set classifier that must always pick a colour.
  * Vehicle colour combines CLIP with an object-aware prompt ("a photo of a
    black car") and the pixel colour of the body below the windscreen.
  * Every value carries a confidence and a source; nothing defaults to a value.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np

from app.core.config import settings
from app.services.pipeline import clip_attributes
from app.services.pipeline.body_regions import face_skin_patch, person_regions, vehicle_body_region
from app.services.pipeline.color_naming import COLOR_NAMES, crop, read_color, skin_reference

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "van"}
MIN_PERSON_HEIGHT_PX = 60
MIN_VEHICLE_WIDTH_PX = 40
BARE_SKIN_FRACTION_WITH_REF = 0.45
BARE_SKIN_FRACTION_NO_REF = 0.60   # generic skin range is less trustworthy
# Between this and the bare-skin threshold the torso is partly skin-coloured
# (e.g. mixed with water or a dark float): no claim either way.
UNCERTAIN_SKIN_FRACTION = 0.25
BAG_CLASSES = ("backpack", "handbag", "suitcase")

UPPER_TYPES = ["t-shirt", "shirt", "jacket", "coat", "hoodie", "sweater"]
HEADWEAR = ["hat", "cap", "helmet", "none"]
BODY_STYLES = ["sedan", "SUV", "pickup truck", "hatchback", "van", "minivan", "bus", "truck", "motorcycle"]

_UPPER_TYPE_Q = (UPPER_TYPES, [f"a photo of a person wearing a {t}" for t in UPPER_TYPES])
_HEADWEAR_Q = (HEADWEAR, ["a photo of a person wearing a hat", "a photo of a person wearing a cap",
                          "a photo of a person wearing a helmet", "a photo of a person with nothing on their head"])
_BODY_Q = (BODY_STYLES, [f"a photo of a {b}" for b in BODY_STYLES])

POSE_SOURCE = "POSE_CROP_COLOR_MODEL"
BOX_SOURCE = "COLOR_HEURISTIC"


def _set(det: Dict[str, Any], attr: str, value: Any, confidence: float, source: str) -> None:
    det[attr] = value
    det[f"{attr}_confidence"] = round(float(confidence), 4)
    det[f"{attr}_source"] = source


def enrich_person(
    frame: np.ndarray, det: Dict[str, Any], box_px,
    keypoints_px: Optional[np.ndarray], keypoint_conf: Optional[np.ndarray],
) -> None:
    height = box_px[3] - box_px[1]
    if height < MIN_PERSON_HEIGHT_PX:
        det["attributes_skipped"] = f"person too small ({int(height)}px < {MIN_PERSON_HEIGHT_PX}px)"
        return

    regions = person_regions(box_px, keypoints_px, keypoint_conf)
    face_patch = face_skin_patch(keypoints_px, keypoint_conf)
    ref = skin_reference(crop(frame, face_patch)) if face_patch else None
    color_source = POSE_SOURCE if regions.source == "POSE" else BOX_SOURCE
    bare_threshold = BARE_SKIN_FRACTION_WITH_REF if ref is not None else BARE_SKIN_FRACTION_NO_REF
    det["attribute_region_source"] = regions.source

    upper_crop = crop(frame, regions.upper) if regions.upper else None
    if upper_crop is not None:
        rd = read_color(upper_crop, skin_ref_lab=ref)
        if rd.skin_fraction >= bare_threshold:
            # Discounted when judged without the person's own skin tone.
            _set(det, "upper_garment_presence", "absent",
                 min(0.95, rd.skin_fraction) * (1.0 if ref is not None else 0.7), color_source)
        elif rd.skin_fraction >= UNCERTAIN_SKIN_FRACTION:
            det["upper_garment_not_visible"] = (
                f"torso partly skin-coloured ({rd.skin_fraction:.0%}); garment presence uncertain"
            )
        else:
            _set(det, "upper_garment_presence", "present", min(0.95, 1.0 - rd.skin_fraction), color_source)
            if rd.color:
                _set(det, "upper_garment_color", rd.color, rd.share, color_source)
            if min(upper_crop.shape[:2]) >= 24:
                res = clip_attributes.classify(upper_crop, {"type": _UPPER_TYPE_Q})
                label, p, _ = clip_attributes.top(res.get("type", {}))
                if label:
                    _set(det, "upper_garment_type", label, p, "CLIP_ZERO_SHOT")
    else:
        det["upper_garment_not_visible"] = regions.upper_reason

    lower_crop = crop(frame, regions.lower) if regions.lower else None
    if lower_crop is not None:
        rd = read_color(lower_crop, skin_ref_lab=ref)
        # Mostly skin below the knee means bare legs (e.g. shorts): no trouser colour.
        if rd.skin_fraction < bare_threshold and rd.color:
            _set(det, "lower_garment_color", rd.color, rd.share, color_source)
    else:
        det["lower_garment_not_visible"] = regions.lower_reason

    head_crop = crop(frame, regions.head) if regions.head else None
    if head_crop is not None and min(head_crop.shape[:2]) >= 16:
        res = clip_attributes.classify(head_crop, {"headwear": _HEADWEAR_Q})
        label, p, _ = clip_attributes.top(res.get("headwear", {}))
        if label:
            _set(det, "headwear", "bare head" if label == "none" else label, p, "CLIP_ZERO_SHOT")


def associate_carried_items(frame_dets) -> None:
    """
    A person carries a bag only when the detector found a bag (backpack /
    handbag / suitcase) whose centre lies within that person's box. CLIP was
    tried for this and reported handbags on people holding phones, so carried
    items are grounded in detected objects. No detected bag means no claim —
    absence of a detection is not evidence of absence.
    """
    persons = [d for d in frame_dets if d["class_name"] == "person"]
    bags = [d for d in frame_dets if d["class_name"] in BAG_CLASSES]
    for bag in bags:
        cx = (bag["bbox_x1"] + bag["bbox_x2"]) / 2
        cy = (bag["bbox_y1"] + bag["bbox_y2"]) / 2
        holders = [p for p in persons
                   if p["bbox_x1"] <= cx <= p["bbox_x2"] and p["bbox_y1"] <= cy <= p["bbox_y2"]]
        if not holders:
            continue
        # Closest person centre if several boxes contain the bag.
        holder = min(holders, key=lambda p: abs((p["bbox_x1"] + p["bbox_x2"]) / 2 - cx))
        if bag["confidence"] > holder.get("carried_item_confidence", 0.0):
            _set(holder, "carried_item", bag["class_name"], bag["confidence"], "YOLO_DETECTOR")
            _set(holder, "carries_bag", True, bag["confidence"], "YOLO_DETECTOR")


def enrich_vehicle(frame: np.ndarray, det: Dict[str, Any], box_px) -> None:
    width = box_px[2] - box_px[0]
    if width < MIN_VEHICLE_WIDTH_PX:
        det["attributes_skipped"] = f"vehicle too small ({int(width)}px < {MIN_VEHICLE_WIDTH_PX}px)"
        return

    vehicle_crop = crop(frame, box_px)
    cls = det["class_name"]
    clip_res = clip_attributes.classify(vehicle_crop, {
        "color": (COLOR_NAMES, [f"a photo of a {c} {cls}" for c in COLOR_NAMES]),
        "body": _BODY_Q,
    })
    pixel = read_color(crop(frame, vehicle_body_region(box_px)), exclude_skin=False, min_share=0.0)

    clip_color = clip_res.get("color", {})
    if clip_color or pixel.distribution:
        # Fixed weights: CLIP understands "the car" despite glass and road;
        # pixels anchor the hue when a large graphic misleads CLIP.
        scores = {c: 0.6 * clip_color.get(c, 0.0) + 0.4 * pixel.distribution.get(c, 0.0) for c in COLOR_NAMES}
        if not clip_color:
            scores = {c: pixel.distribution.get(c, 0.0) for c in COLOR_NAMES}
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            _set(det, "vehicle_color", best, scores[best], "COMBINED" if clip_color else "COLOR_HEURISTIC")

    label, p, _ = clip_attributes.top(clip_res.get("body", {}))
    if label:
        _set(det, "vehicle_body_style", label, p, "CLIP_ZERO_SHOT")


def enrich_detection(frame: np.ndarray, det: Dict[str, Any],
                     keypoints_px: Optional[np.ndarray] = None,
                     keypoint_conf: Optional[np.ndarray] = None) -> None:
    if not settings.ENABLE_ATTRIBUTE_CLASSIFICATION:
        return
    h, w = frame.shape[:2]
    box_px = (det["bbox_x1"] * w, det["bbox_y1"] * h, det["bbox_x2"] * w, det["bbox_y2"] * h)
    try:
        if det["class_name"] == "person":
            enrich_person(frame, det, box_px, keypoints_px, keypoint_conf)
        elif det["class_name"] in VEHICLE_CLASSES:
            enrich_vehicle(frame, det, box_px)
    except Exception as e:  # one bad crop must not lose the detection
        logger.warning(f"[ATTR] attribute extraction failed for a {det['class_name']}: {e}")
        det["attributes_skipped"] = f"error: {type(e).__name__}"
