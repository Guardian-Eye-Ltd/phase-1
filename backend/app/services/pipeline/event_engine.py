import logging
import math
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EventEngine:
    """
    Deterministic rule-based Event Engine for detecting forensic events:
    - LOITERING (track remaining in localized zone > threshold)
    - PERSON_VEHICLE_PROXIMITY (spatial proximity between person and vehicle)
    - OBJECT_CARRIED (co-location of person and bag tracks)
    - SUDDEN_MOVEMENT (abrupt velocity spikes)
    """

    @classmethod
    def detect_events(
        cls, 
        tracks: List[Dict[str, Any]], 
        all_detections: List[Dict[str, Any]],
        loitering_threshold_seconds: float = 10.0
    ) -> List[Dict[str, Any]]:
        events = []
        
        # Group detections by track_id
        track_dets: Dict[int, List[Dict[str, Any]]] = {}
        for d in all_detections:
            tid = d.get("track_id")
            if tid is not None:
                track_dets.setdefault(tid, []).append(d)

        # 1. Detect Loitering & Sudden Movement per Track
        for trk in tracks:
            tid = trk["track_number"]
            cls_name = trk["class_name"]
            duration = trk.get("duration", 0.0)
            dets = track_dets.get(tid, [])

            if not dets:
                continue

            # Loitering Rule: Person track remaining in area for > loitering_threshold_seconds
            if cls_name == "person" and duration >= loitering_threshold_seconds:
                centroids = [
                    ((d["bbox_x1"] + d["bbox_x2"]) / 2.0, (d["bbox_y1"] + d["bbox_y2"]) / 2.0)
                    for d in dets
                ]
                min_x = min(c[0] for c in centroids)
                max_x = max(c[0] for c in centroids)
                min_y = min(c[1] for c in centroids)
                max_y = max(c[1] for c in centroids)
                
                spatial_spread = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
                if spatial_spread < 0.30:  # Stayed in small localized region
                    events.append({
                        "event_type": "LOITERING",
                        "title": f"Loitering Detected (Track {tid})",
                        "description": f"Person Track {tid} remained in localized area for {duration:.1f}s.",
                        "start_time": trk["first_seen_timestamp"],
                        "end_time": trk["last_seen_timestamp"],
                        "involved_track_ids": [tid],
                        "confidence": 0.90,
                        "metadata": {
                            "track_id": tid,
                            "duration": duration,
                            "spatial_spread": round(spatial_spread, 3)
                        }
                    })

            # Sudden Movement Rule: Velocity spike between consecutive detections
            if len(dets) >= 2:
                for idx in range(1, len(dets)):
                    prev_d = dets[idx-1]
                    curr_d = dets[idx]
                    dt = curr_d["timestamp"] - prev_d["timestamp"]
                    if 0.05 <= dt <= 2.0:
                        pcx = (prev_d["bbox_x1"] + prev_d["bbox_x2"]) / 2.0
                        pcy = (prev_d["bbox_y1"] + prev_d["bbox_y2"]) / 2.0
                        ccx = (curr_d["bbox_x1"] + curr_d["bbox_x2"]) / 2.0
                        ccy = (curr_d["bbox_y1"] + curr_d["bbox_y2"]) / 2.0
                        dist = math.sqrt((ccx - pcx)**2 + (ccy - pcy)**2)
                        velocity = dist / dt
                        if velocity > 0.40:  # Normalized screen distance per second
                            events.append({
                                "event_type": "SUDDEN_MOVEMENT",
                                "title": f"Sudden Movement (Track {tid})",
                                "description": f"{cls_name.capitalize()} Track {tid} exhibited rapid movement (speed: {velocity:.2f}/s) at {curr_d['timestamp']:.2f}s.",
                                "start_time": prev_d["timestamp"],
                                "end_time": curr_d["timestamp"],
                                "involved_track_ids": [tid],
                                "confidence": 0.85,
                                "metadata": {
                                    "track_id": tid,
                                    "velocity": round(velocity, 2)
                                }
                            })
                            break  # Record max one sudden movement per track

        # 2. Detect Person-Vehicle Proximity & Object Carried across tracks
        person_tracks = [t for t in tracks if t["class_name"] == "person"]
        vehicle_tracks = [t for t in tracks if t["class_name"] in ["car", "truck", "bus", "motorcycle"]]
        bag_tracks = [t for t in tracks if t["class_name"] in ["backpack", "handbag", "suitcase"]]

        for p_trk in person_tracks:
            pid = p_trk["track_number"]
            p_dets = track_dets.get(pid, [])
            
            # Check proximity to vehicles
            for v_trk in vehicle_tracks:
                vid = v_trk["track_number"]
                v_dets = track_dets.get(vid, [])
                
                # Check overlapping timestamps
                overlap_start = max(p_trk["first_seen_timestamp"], v_trk["first_seen_timestamp"])
                overlap_end = min(p_trk["last_seen_timestamp"], v_trk["last_seen_timestamp"])
                
                if overlap_end >= overlap_start:
                    min_dist = 999.0
                    for pd in p_dets:
                        pcx = (pd["bbox_x1"] + pd["bbox_x2"]) / 2.0
                        pcy = (pd["bbox_y1"] + pd["bbox_y2"]) / 2.0
                        for vd in v_dets:
                            if abs(pd["timestamp"] - vd["timestamp"]) < 1.0:
                                vcx = (vd["bbox_x1"] + vd["bbox_x2"]) / 2.0
                                vcy = (vd["bbox_y1"] + vd["bbox_y2"]) / 2.0
                                dist = math.sqrt((pcx - vcx)**2 + (pcy - vcy)**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    
                    if min_dist < 0.25:
                        events.append({
                            "event_type": "PERSON_VEHICLE_PROXIMITY",
                            "title": f"Person-Vehicle Proximity (Track {pid} & Vehicle Track {vid})",
                            "description": f"Person Track {pid} approached Vehicle Track {vid} between {overlap_start:.1f}s and {overlap_end:.1f}s (Min Distance: {min_dist:.2f}).",
                            "start_time": overlap_start,
                            "end_time": overlap_end,
                            "involved_track_ids": [pid, vid],
                            "confidence": 0.88,
                            "metadata": {
                                "person_track_id": pid,
                                "vehicle_track_id": vid,
                                "min_distance": round(min_dist, 3)
                            }
                        })

            # Check carried bag association across tracks
            for b_trk in bag_tracks:
                bid = b_trk["track_number"]
                b_dets = track_dets.get(bid, [])
                overlap_start = max(p_trk["first_seen_timestamp"], b_trk["first_seen_timestamp"])
                overlap_end = min(p_trk["last_seen_timestamp"], b_trk["last_seen_timestamp"])
                if overlap_end >= overlap_start:
                    events.append({
                        "event_type": "OBJECT_CARRIED",
                        "title": f"Carried Bag Association (Track {pid} & Bag Track {bid})",
                        "description": f"Person Track {pid} was co-located with {b_trk['class_name'].capitalize()} Track {bid} between {overlap_start:.1f}s and {overlap_end:.1f}s.",
                        "start_time": overlap_start,
                        "end_time": overlap_end,
                        "involved_track_ids": [pid, bid],
                        "confidence": 0.92,
                        "metadata": {
                            "person_track_id": pid,
                            "bag_track_id": bid
                        }
                    })

        logger.info(f"[EVENTS] EventEngine detected {len(events)} structured forensic events.")
        return events
