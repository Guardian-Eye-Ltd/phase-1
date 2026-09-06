import os
import cv2
from typing import Dict, Any

class VideoValidationError(Exception):
    pass

class VideoValidator:
    """
    Validates CCTV video file existence, format readability, and basic technical parameters.
    """
    @staticmethod
    def validate_video(file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise VideoValidationError(f"Video file does not exist at path: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise VideoValidationError(f"Video file is empty (0 bytes): {file_path}")

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise VideoValidationError(f"OpenCV failed to open video file. Corrupted or unsupported codec: {file_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if frame_count <= 0:
            raise VideoValidationError(f"Video file contains no valid decodable frames (frame_count={frame_count}).")

        calc_fps = fps if (fps and fps > 0 and fps < 240) else 30.0
        duration = frame_count / calc_fps

        return {
            "valid": True,
            "width": width,
            "height": height,
            "fps": calc_fps,
            "frame_count": frame_count,
            "duration": round(duration, 2),
            "file_size": os.path.getsize(file_path)
        }
