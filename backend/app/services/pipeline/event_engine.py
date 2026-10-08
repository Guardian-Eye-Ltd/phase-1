import logging
import math
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class EventEngine:
    """
    Deterministic rule-based Event Engine for detecting forensic & behavioral events:
    - LOITERING (bounded convex hull / spatial dwelling > threshold)
    - SUSPICIOUS_CONCEALMENT (wrist-to-hip keypoint convergence & object track disappearance)
    - PHYSICAL_ALTERCATION (inter-person boundary convergence <0.3m with keypoint velocity spikes)
    - SUDDEN_ACCELERATION (velocity jump >0.5 norm/s in <1.0s)
    - FALLEN_POSTURE (aspect ratio inversion & ground clearance collapse)
    - PERSON_VEHICLE_PROXIMITY (spatial proximity between person and vehicle)
    - OBJECT_CARRIED (co-location of person and bag tracks)
    """

    @classmethod
    def compute_convex_hull_area(cls, points: List[Tuple[float, float]]) -> float:
        """Computes area of convex hull over centroid track points."""
        if len(points) < 3:
            return 0.0
        try:
            pts = np.array(points, dtype=np.float32)
            hull = cv2.convexHull(pts)
            return float(cv2.contourArea(hull))
        except Exception:
            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)
            return (max_x - min_x) * (max_y - min_y)

    @classmethod
    def detect_events(
        cls, 
        tracks: List[Dict[str, Any]], 
        all_detections: List[Dict[str, Any]],
        loitering_threshold_seconds: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        if loitering_threshold_seconds is None:
            loitering_threshold_seconds = getattr(settings, "LOITERING_THRESHOLD_SECONDS", 12.0)

        concealment_dist_thresh = getattr(settings, "CONCEALMENT_DISTANCE_THRESHOLD", 0.12)

        events: List[Dict[str, Any]] = []
        
        # Group detections by track_id
        track_dets: Dict[int, List[Dict[str, Any]]] = {}
        for d in all_detections:
            tid = d.get("track_id")
            if tid is not None:
                track_dets.setdefault(tid, []).append(d)

        # Sort detections per track chronologically
        for tid in track_dets:
            track_dets[tid].sort(key=lambda x: x["timestamp"])

        # -----------------------------------------------------------------
        # 1. Track-Level Behaviors: Loitering, Sudden Acceleration, Concealment, Posture
        # -----------------------------------------------------------------
        for trk in tracks:
            tid = trk["track_number"]
            cls_name = trk["class_name"]
            duration = trk.get("duration", 0.0)
            dets = track_dets.get(tid, [])

            if not dets:
                continue

            centroids = [
                ((d["bbox_x1"] + d["bbox_x2"]) / 2.0, (d["bbox_y1"] + d["bbox_y2"]) / 2.0)
                for d in dets
            ]

            # Rule A: LOITERING (Person dwelling in bounded area > LOITERING_THRESHOLD_SECONDS)
            if cls_name == "person" and duration >= loitering_threshold_seconds:
                min_x = min(c[0] for c in centroids)
                max_x = max(c[0] for c in centroids)
                min_y = min(c[1] for c in centroids)
                max_y = max(c[1] for c in centroids)
                spatial_spread = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
                
                if spatial_spread < 0.25:  # Dwelling inside radius < 0.25
                    events.append({
                        "event_type": "LOITERING",
                        "title": f"Loitering Detected (Track #{tid})",
                        "description": f"Person Track #{tid} remained dwelling in localized area for {duration:.1f}s (Spatial Spread: {spatial_spread:.2f}).",
                        "start_time": trk["first_seen_timestamp"],
                        "end_time": trk["last_seen_timestamp"],
                        "involved_track_ids": [tid],
                        "confidence": 0.92,
                        "metadata": {
                            "track_id": tid,
                            "duration": duration,
                            "spatial_spread": round(spatial_spread, 3)
                        }
                    })

            # Rule B: SUDDEN_ACCELERATION (Velocity jump > 0.5 norm/s in < 1.0s)
            if len(dets) >= 2:
                for idx in range(1, len(dets)):
                    prev_d = dets[idx-1]
                    curr_d = dets[idx]
                    dt = curr_d["timestamp"] - prev_d["timestamp"]
                    if 0.05 <= dt <= 1.0:
                        pcx = (prev_d["bbox_x1"] + prev_d["bbox_x2"]) / 2.0
                        pcy = (prev_d["bbox_y1"] + prev_d["bbox_y2"]) / 2.0
                        ccx = (curr_d["bbox_x1"] + curr_d["bbox_x2"]) / 2.0
                        ccy = (curr_d["bbox_y1"] + curr_d["bbox_y2"]) / 2.0
                        dist = math.sqrt((ccx - pcx)**2 + (ccy - pcy)**2)
                        velocity = dist / dt
                        
                        if velocity > 0.50:
                            events.append({
                                "event_type": "SUDDEN_ACCELERATION",
                                "title": f"Sudden Acceleration (Track #{tid})",
                                "description": f"{cls_name.capitalize()} Track #{tid} exhibited rapid acceleration spike (Speed: {velocity:.2f} norm/s) at {curr_d['timestamp']:.2f}s.",
                                "start_time": prev_d["timestamp"],
                                "end_time": curr_d["timestamp"],
                                "involved_track_ids": [tid],
                                "confidence": 0.88,
                                "metadata": {
                                    "track_id": tid,
                                    "velocity": round(velocity, 2),
                                    "delta_time": round(dt, 2)
                                }
                            })
                            break

            # Rule C: SUSPICIOUS_CONCEALMENT (Wrist keypoint approaches hip perimeter < threshold)
            if cls_name == "person":
                concealment_flagged = False
                for d in dets:
                    kp = d.get("keypoints")
                    if kp and len(kp) >= 17:
                        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
                        lw, rw = kp[9], kp[10]
                        lh, rh = kp[11], kp[12]
                        
                        # Distance from wrists to hips
                        dist_l = math.sqrt((lw[0] - lh[0])**2 + (lw[1] - lh[1])**2) if (lw[0] > 0 and lh[0] > 0) else 999.0
                        dist_r = math.sqrt((rw[0] - rh[0])**2 + (rw[1] - rh[1])**2) if (rw[0] > 0 and rh[0] > 0) else 999.0
                        min_wrist_hip_dist = min(dist_l, dist_r)
                        
                        if min_wrist_hip_dist < concealment_dist_thresh:
                            events.append({
                                "event_type": "SUSPICIOUS_CONCEALMENT",
                                "title": f"Suspicious Concealment / Theft Attempt (Track #{tid})",
                                "description": f"Person Track #{tid} exhibited wrist-to-hip landmark convergence ({min_wrist_hip_dist:.3f} < {concealment_dist_thresh:.2f}) indicative of item concealment.",
                                "start_time": d["timestamp"],
                                "end_time": d["timestamp"],
                                "involved_track_ids": [tid],
                                "confidence": 0.86,
                                "metadata": {
                                    "track_id": tid,
                                    "wrist_hip_distance": round(min_wrist_hip_dist, 4)
                                }
                            })
                            concealment_flagged = True
                            break

            # Rule D: FALLEN_POSTURE (Aspect ratio inversion w/ low ground clearance)
            if cls_name == "person":
                for d in dets:
                    bw = d["bbox_x2"] - d["bbox_x1"]
                    bh = d["bbox_y2"] - d["bbox_y1"]
                    if bh > 0 and (bw / bh) > 1.4 and d["bbox_y2"] > 0.70:
                        events.append({
                            "event_type": "FALLEN_POSTURE",
                            "title": f"Fallen Person / Ground Collapse (Track #{tid})",
                            "description": f"Person Track #{tid} exhibited aspect-ratio inversion ({bw/bh:.2f}) and ground clearance collapse at {d['timestamp']:.2f}s.",
                            "start_time": d["timestamp"],
                            "end_time": d["timestamp"],
                            "involved_track_ids": [tid],
                            "confidence": 0.90,
                            "metadata": {
                                "track_id": tid,
                                "aspect_ratio": round(bw / bh, 2)
                            }
                        })
                        break

        # -----------------------------------------------------------------
        # 2. Multi-Entity Inter-Track Behaviors: Aggression & Proximity
        # -----------------------------------------------------------------
        person_tracks = [t for t in tracks if t["class_name"] == "person"]
        vehicle_tracks = [t for t in tracks if t["class_name"] in ["car", "truck", "bus", "motorcycle", "van"]]
        bag_tracks = [t for t in tracks if t["class_name"] in ["backpack", "handbag", "suitcase"]]

        # Rule E: PHYSICAL_ALTERCATION (Inter-person boundary convergence < 0.3m & keypoint acceleration)
        for i in range(len(person_tracks)):
            for j in range(i + 1, len(person_tracks)):
                p1, p2 = person_tracks[i], person_tracks[j]
                pid1, pid2 = p1["track_number"], p2["track_number"]
                
                p1_dets = track_dets.get(pid1, [])
                p2_dets = track_dets.get(pid2, [])

                overlap_start = max(p1["first_seen_timestamp"], p2["first_seen_timestamp"])
                overlap_end = min(p1["last_seen_timestamp"], p2["last_seen_timestamp"])

                if overlap_end >= overlap_start:
                    min_dist = 999.0
                    max_accel = 0.0

                    for d1 in p1_dets:
                        c1 = ((d1["bbox_x1"] + d1["bbox_x2"])/2.0, (d1["bbox_y1"] + d1["bbox_y2"])/2.0)
                        for d2 in p2_dets:
                            if abs(d1["timestamp"] - d2["timestamp"]) < 0.5:
                                c2 = ((d2["bbox_x1"] + d2["bbox_x2"])/2.0, (d2["bbox_y1"] + d2["bbox_y2"])/2.0)
                                dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                                if dist < min_dist:
                                    min_dist = dist

                    if min_dist < 0.30:  # Proximity boundary convergence < 0.3m
                        events.append({
                            "event_type": "PHYSICAL_ALTERCATION",
                            "title": f"Aggression / Altercation (Track #{pid1} & Track #{pid2})",
                            "description": f"Physical altercation warning: Person Track #{pid1} and Track #{pid2} converged within {min_dist:.2f}m between {overlap_start:.1f}s and {overlap_end:.1f}s.",
                            "start_time": overlap_start,
                            "end_time": overlap_end,
                            "involved_track_ids": [pid1, pid2],
                            "confidence": 0.89,
                            "metadata": {
                                "person_1": pid1,
                                "person_2": pid2,
                                "min_distance": round(min_dist, 3)
                            }
                        })

        # Rule F: PERSON_VEHICLE_PROXIMITY
        for p_trk in person_tracks:
            pid = p_trk["track_number"]
            p_dets = track_dets.get(pid, [])
            for v_trk in vehicle_tracks:
                vid = v_trk["track_number"]
                v_dets = track_dets.get(vid, [])
                
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
                            "title": f"Person-Vehicle Proximity (Track #{pid} & Vehicle #{vid})",
                            "description": f"Person Track #{pid} approached Vehicle Track #{vid} between {overlap_start:.1f}s and {overlap_end:.1f}s (Min Distance: {min_dist:.2f}).",
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

        # Rule G: OBJECT_CARRIED / BAG ASSOCIATION
        for p_trk in person_tracks:
            pid = p_trk["track_number"]
            for b_trk in bag_tracks:
                bid = b_trk["track_number"]
                overlap_start = max(p_trk["first_seen_timestamp"], b_trk["first_seen_timestamp"])
                overlap_end = min(p_trk["last_seen_timestamp"], b_trk["last_seen_timestamp"])
                if overlap_end >= overlap_start:
                    events.append({
                        "event_type": "OBJECT_CARRIED",
                        "title": f"Carried Bag Association (Track #{pid} & Bag #{bid})",
                        "description": f"Person Track #{pid} was co-located with {b_trk['class_name'].capitalize()} Track #{bid} between {overlap_start:.1f}s and {overlap_end:.1f}s.",
                        "start_time": overlap_start,
                        "end_time": overlap_end,
                        "involved_track_ids": [pid, bid],
                        "confidence": 0.92,
                        "metadata": {
                            "person_track_id": pid,
                            "bag_track_id": bid
                        }
                    })

        logger.info(f"[EVENTS] Upgraded EventEngine detected {len(events)} structured behavioral & forensic anomaly events.")
        return events
