import math
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes IoU between two normalized bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compute_centroid_distance(boxA: List[float], boxB: List[float]) -> float:
    """Computes normalized Euclidean distance between centroids of two boxes."""
    cxA = (boxA[0] + boxA[2]) / 2.0
    cyA = (boxA[1] + boxA[3]) / 2.0
    cxB = (boxB[0] + boxB[2]) / 2.0
    cyB = (boxB[1] + boxB[3]) / 2.0
    return math.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)

class MultiObjectTracker:
    """
    Robust Multi-Object Tracker maintaining stable track IDs across video frames
    using hybrid IoU + Centroid Proximity matching with lost-track memory.
    """
    def __init__(
        self, 
        iou_threshold: float = 0.15,
        max_time_lost: float = 3.0,
        max_centroid_dist: float = 0.35
    ):
        self.iou_threshold = iou_threshold
        self.max_time_lost = max_time_lost
        self.max_centroid_dist = max_centroid_dist
        self.next_track_id = 1
        # track_id -> track_state dict
        self.active_tracks: Dict[int, Dict[str, Any]] = {}

    def process_frame_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []

        frame_ts = detections[0]["timestamp"]
        frame_num = detections[0]["frame_number"]

        # Prune stale tracks exceeding max_time_lost
        stale_ids = [
            tid for tid, info in self.active_tracks.items()
            if (frame_ts - info["last_timestamp"]) > self.max_time_lost
        ]
        for tid in stale_ids:
            del self.active_tracks[tid]

        updated_detections = []
        assigned_track_ids = set()

        for det in detections:
            bbox = [det["bbox_x1"], det["bbox_y1"], det["bbox_x2"], det["bbox_y2"]]
            cls_name = det["class_name"]

            best_match_id = None
            best_score = -1.0

            for track_id, last_info in self.active_tracks.items():
                if track_id in assigned_track_ids:
                    continue
                if last_info["class_name"] != cls_name:
                    continue

                iou = compute_iou(bbox, last_info["bbox"])
                dist = compute_centroid_distance(bbox, last_info["bbox"])

                # Hybrid match score combining IoU and normalized centroid proximity
                if iou >= self.iou_threshold:
                    match_score = 0.6 * iou + 0.4 * max(0.0, 1.0 - (dist / self.max_centroid_dist))
                elif dist <= self.max_centroid_dist:
                    match_score = 0.4 * max(0.0, 1.0 - (dist / self.max_centroid_dist))
                else:
                    match_score = 0.0

                if match_score > best_score and match_score > 0.20:
                    best_score = match_score
                    best_match_id = track_id

            if best_match_id is not None:
                track_id = best_match_id
            else:
                track_id = self.next_track_id
                self.next_track_id += 1

            assigned_track_ids.add(track_id)
            
            # Update track state in memory
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    "track_id": track_id,
                    "class_name": cls_name,
                    "bbox": bbox,
                    "last_timestamp": frame_ts,
                    "last_frame_number": frame_num,
                    "confidences": [det["confidence"]]
                }
            else:
                info = self.active_tracks[track_id]
                info["bbox"] = bbox
                info["last_timestamp"] = frame_ts
                info["last_frame_number"] = frame_num
                info["confidences"].append(det["confidence"])

            det_copy = dict(det)
            det_copy["track_id"] = track_id
            updated_detections.append(det_copy)

        return updated_detections

    @staticmethod
    def generate_track_summaries(all_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregates frame detections into stable track summary records with statistics and visual attributes.
        """
        from collections import Counter
        tracks_map: Dict[int, Dict[str, Any]] = {}

        for det in all_detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            ts = det["timestamp"]
            fn = det["frame_number"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            bbox = [det["bbox_x1"], det["bbox_y1"], det["bbox_x2"], det["bbox_y2"]]

            if track_id not in tracks_map:
                tracks_map[track_id] = {
                    "track_number": track_id,
                    "class_name": cls_name,
                    "first_seen_timestamp": ts,
                    "last_seen_timestamp": ts,
                    "first_seen_frame": fn,
                    "last_seen_frame": fn,
                    "duration": 0.0,
                    "observation_count": 1,
                    "keyframe_count": 0,
                    "confidences": [conf],
                    "bboxes": [bbox],
                    "upper_colors": [det["upper_garment_color"]] if det.get("upper_garment_color") and det["upper_garment_color"] != "unknown" else [],
                    "lower_colors": [det["lower_garment_color"]] if det.get("lower_garment_color") and det["lower_garment_color"] != "unknown" else [],
                    "vehicle_colors": [det["vehicle_color"]] if det.get("vehicle_color") and det["vehicle_color"] != "unknown" else [],
                    "carries_bag_count": 1 if det.get("carries_bag") else 0
                }
            else:
                t = tracks_map[track_id]
                t["last_seen_timestamp"] = ts
                t["last_seen_frame"] = fn
                t["duration"] = round(ts - t["first_seen_timestamp"], 2)
                t["observation_count"] += 1
                t["confidences"].append(conf)
                t["bboxes"].append(bbox)
                if det.get("upper_garment_color") and det["upper_garment_color"] != "unknown":
                    t["upper_colors"].append(det["upper_garment_color"])
                if det.get("lower_garment_color") and det["lower_garment_color"] != "unknown":
                    t["lower_colors"].append(det["lower_garment_color"])
                if det.get("vehicle_color") and det["vehicle_color"] != "unknown":
                    t["vehicle_colors"].append(det["vehicle_color"])
                if det.get("carries_bag"):
                    t["carries_bag_count"] += 1

        # Compute summary statistics per track
        summaries = []
        for t in tracks_map.values():
            confs = t.pop("confidences", [0.8])
            bboxes = t.pop("bboxes", [])
            u_colors = t.pop("upper_colors", [])
            l_colors = t.pop("lower_colors", [])
            v_colors = t.pop("vehicle_colors", [])
            bag_count = t.pop("carries_bag_count", 0)

            t["min_confidence"] = round(min(confs), 2)
            t["max_confidence"] = round(max(confs), 2)
            t["avg_confidence"] = round(sum(confs) / float(len(confs)), 2)
            
            t["dominant_upper_color"] = Counter(u_colors).most_common(1)[0][0] if u_colors else None
            t["dominant_lower_color"] = Counter(l_colors).most_common(1)[0][0] if l_colors else None
            t["dominant_vehicle_color"] = Counter(v_colors).most_common(1)[0][0] if v_colors else None
            t["carries_bag"] = bag_count > 0

            summaries.append(t)

        logger.info(f"[TRACK] Generated {len(summaries)} unique track summaries.")
        return summaries
