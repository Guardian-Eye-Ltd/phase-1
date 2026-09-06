from typing import List, Dict, Any

class InteractionDetector:
    """
    Detects spatial proximity and bounding box overlap between tracked entities
    over time, carefully labeling results as POSSIBLE_INTERACTION.
    """
    @staticmethod
    def detect_interactions(all_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not all_detections:
            return []

        # Group detections by frame_number
        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for det in all_detections:
            fn = det["frame_number"]
            if fn not in by_frame:
                by_frame[fn] = []
            by_frame[fn].append(det)

        interactions = []
        interaction_pairs = {}  # (trackA, trackB) -> list of timestamps

        for fn, frame_dets in by_frame.items():
            if len(frame_dets) < 2:
                continue

            for i in range(len(frame_dets)):
                for j in range(i + 1, len(frame_dets)):
                    d1, d2 = frame_dets[i], frame_dets[j]
                    t1, t2 = d1.get("track_id"), d2.get("track_id")
                    if t1 is None or t2 is None or t1 == t2:
                        continue

                    # Calculate centroid distance between bounding boxes
                    c1_x = (d1["bbox_x1"] + d1["bbox_x2"]) / 2.0
                    c1_y = (d1["bbox_y1"] + d1["bbox_y2"]) / 2.0
                    c2_x = (d2["bbox_x1"] + d2["bbox_x2"]) / 2.0
                    c2_y = (d2["bbox_y1"] + d2["bbox_y2"]) / 2.0

                    dist = ((c1_x - c2_x)**2 + (c1_y - c2_y)**2)**0.5

                    if dist <= 0.35:  # Spatial proximity threshold
                        pair_key = (min(t1, t2), max(t1, t2))
                        if pair_key not in interaction_pairs:
                            interaction_pairs[pair_key] = []
                        interaction_pairs[pair_key].append((d1["timestamp"], dist))

        for (t1, t2), ts_dist_list in interaction_pairs.items():
            timestamps = [item[0] for item in ts_dist_list]
            distances = [item[1] for item in ts_dist_list]

            start_time = min(timestamps)
            end_time = max(timestamps)
            min_dist = round(min(distances), 4)

            interactions.append({
                "entity_a_track_id": t1,
                "entity_b_track_id": t2,
                "start_time": start_time,
                "end_time": end_time,
                "min_distance_or_overlap": min_dist,
                "confidence_score": 0.85,
                "label": "POSSIBLE_INTERACTION"
            })

        return interactions
