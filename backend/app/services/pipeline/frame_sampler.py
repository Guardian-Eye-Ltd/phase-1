import cv2
import numpy as np
from typing import Generator, Tuple

class FrameSampler:
    """
    Configurable frame sampling service for extracting frames from CCTV video footage.
    """
    @staticmethod
    def sample_frames(
        file_path: str, 
        target_fps: float = 2.0
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for frame sampling: {file_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0:
            video_fps = 30.0

        # Determine frame interval step
        step = max(1, int(round(video_fps / target_fps)))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                timestamp = round(frame_idx / video_fps, 2)
                yield (frame_idx, timestamp, frame)

            frame_idx += 1

        cap.release()
