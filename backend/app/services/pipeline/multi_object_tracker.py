from typing import List, Dict, Any

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    # box format [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class MultiObjectTracker:
    """
    IoU-based Multi-Object Tracker assigning stable temporary track numbers across frames.
    """
    def __init__(self, iou_threshold: float = 0.25):
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}  # track_id -> last_detection_info

    def process_frame_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []

        updated_detections = []
        assigned_track_ids = set()

        for det in detections:
            bbox = [det["bbox_x1"], det["bbox_y1"], det["bbox_x2"], det["bbox_y2"]]
            cls_name = det["class_name"]

            best_match_id = None
            best_iou = 0.0

            for track_id, last_info in self.active_tracks.items():
                if track_id in assigned_track_ids:
                    continue
                if last_info["class_name"] != cls_name:
                    continue

                iou = compute_iou(bbox, last_info["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_match_id = track_id

            if best_match_id is not None:
                track_id = best_match_id
            else:
                track_id = self.next_track_id
                self.next_track_id += 1

            assigned_track_ids.add(track_id)
            self.active_tracks[track_id] = {
                "bbox": bbox,
                "class_name": cls_name,
                "timestamp": det["timestamp"],
                "frame_number": det["frame_number"]
            }

            det_copy = dict(det)
            det_copy["track_id"] = track_id
            updated_detections.append(det_copy)

        return updated_detections

    @staticmethod
    def generate_track_summaries(all_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tracks_map: Dict[int, Dict[str, Any]] = {}

        for det in all_detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            ts = det["timestamp"]
            fn = det["frame_number"]
            cls_name = det["class_name"]

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
                    "keyframe_count": 0
                }
            else:
                t = tracks_map[track_id]
                t["last_seen_timestamp"] = ts
                t["last_seen_frame"] = fn
                t["duration"] = round(ts - t["first_seen_timestamp"], 2)
                t["observation_count"] += 1

        return list(tracks_map.values())
