import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

class MotionFilter:
    """
    Computes frame differencing to identify activity intervals (LOW ACTIVITY vs HIGH ACTIVITY)
    without modifying original video data.
    """
    @staticmethod
    def analyze_motion(sampled_frames: List[Tuple[int, float, np.ndarray]], threshold: float = 15.0) -> List[Dict[str, Any]]:
        if not sampled_frames:
            return []

        intervals = []
        scores = []

        prev_gray = None
        for frame_num, timestamp, frame in sampled_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is None:
                score = 0.0
            else:
                frame_delta = cv2.absdiff(prev_gray, gray)
                score = float(np.mean(frame_delta))

            scores.append((frame_num, timestamp, score))
            prev_gray = gray

        if not scores:
            return []

        # Determine motion threshold dynamically based on median/mean score
        all_scores = [s[2] for s in scores]
        mean_score = float(np.mean(all_scores)) if all_scores else 0.0
        active_thresh = max(5.0, mean_score * 0.8)

        current_level = "LOW ACTIVITY"
        start_fn, start_ts = scores[0][0], scores[0][1]
        interval_scores = [scores[0][2]]

        for i in range(1, len(scores)):
            fn, ts, score = scores[i]
            level = "HIGH ACTIVITY" if score >= active_thresh else "LOW ACTIVITY"

            if level == current_level:
                interval_scores.append(score)
            else:
                end_fn, end_ts = scores[i-1][0], scores[i-1][1]
                intervals.append({
                    "start_frame": start_fn,
                    "end_frame": end_fn,
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "activity_level": current_level,
                    "motion_score": round(float(np.mean(interval_scores)), 2)
                })
                current_level = level
                start_fn, start_ts = fn, ts
                interval_scores = [score]

        # Final interval
        last_fn, last_ts = scores[-1][0], scores[-1][1]
        intervals.append({
            "start_frame": start_fn,
            "end_frame": last_fn,
            "start_time": start_ts,
            "end_time": last_ts,
            "activity_level": current_level,
            "motion_score": round(float(np.mean(interval_scores)), 2)
        })

        return intervals
