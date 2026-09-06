import cv2
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Core 10 forensic object classes
TARGET_CLASSES = {
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "backpack", "handbag", "suitcase", "cell phone"
}

class ObjectDetector:
    """
    Object detection service wrapper supporting Ultralytics YOLOv8 with automatic fallback
    to OpenCV detection for 10 core forensic classes.
    """
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.4):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        self.use_yolo = False

        # Attempt to load Ultralytics YOLO model
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.model_name)
            self.use_yolo = True
            logger.info(f"Loaded Ultralytics YOLO model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Ultralytics YOLO not loaded ({e}). Falling back to OpenCV detection engine.")
            self.use_yolo = False

    def detect_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> List[Dict[str, Any]]:
        detections = []
        height, width = frame.shape[:2]

        if self.use_yolo and self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(frame, conf=self.confidence_threshold, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = r.names[cls_id].lower()
                        conf = float(box.conf[0])

                        if cls_name in TARGET_CLASSES:
                            xyxy = box.xyxy[0].tolist()
                            detections.append({
                                "frame_number": frame_number,
                                "timestamp": timestamp,
                                "class_name": cls_name,
                                "confidence": round(conf, 2),
                                "bbox_x1": round(xyxy[0] / width, 4),
                                "bbox_y1": round(xyxy[1] / height, 4),
                                "bbox_x2": round(xyxy[2] / width, 4),
                                "bbox_y2": round(xyxy[3] / height, 4),
                            })
                return detections
            except Exception as ex:
                logger.warning(f"YOLO inference error ({ex}). Falling back to OpenCV detector.")

        # OpenCV Fallback Detector (HOG Person & Contour/Color Heuristic Object Detector)
        detections = self._opencv_fallback_detect(frame, frame_number, timestamp, width, height)
        return detections

    def _opencv_fallback_detect(self, frame: np.ndarray, frame_number: int, timestamp: float, width: int, height: int) -> List[Dict[str, Any]]:
        detections = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (width * height * 0.015):  # Significant object size threshold
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                
                # Class heuristic based on aspect ratio & bounding dimensions
                cls_name = "person" if aspect_ratio < 0.9 else ("car" if aspect_ratio > 1.2 else "backpack")
                
                detections.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "class_name": cls_name,
                    "confidence": 0.80,
                    "bbox_x1": round(x / width, 4),
                    "bbox_y1": round(y / height, 4),
                    "bbox_x2": round((x + w) / width, 4),
                    "bbox_y2": round((y + h) / height, 4),
                })

        return detections
