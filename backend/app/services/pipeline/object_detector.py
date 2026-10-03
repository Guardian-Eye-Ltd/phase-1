import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# Core 10 forensic object classes
TARGET_CLASSES = {
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "backpack", "handbag", "suitcase", "cell phone"
}

# Global singleton model cache to prevent re-loading YOLO model per frame/job
_MODEL_CACHE: Dict[str, Any] = {}

def get_yolo_model(model_name: str):
    """Loads and caches YOLO model ONCE in memory."""
    if model_name not in _MODEL_CACHE:
        try:
            from ultralytics import YOLO
            logger.info(f"[DETECTION] Loading Ultralytics YOLO model '{model_name}' on device '{settings.DEVICE}'...")
            model = YOLO(model_name)
            _MODEL_CACHE[model_name] = model
            logger.info(f"[DETECTION] Ultralytics YOLO model '{model_name}' successfully cached.")
        except Exception as e:
            logger.warning(f"[DETECTION] Could not load Ultralytics YOLO '{model_name}': {e}. Fallback enabled.")
            _MODEL_CACHE[model_name] = None
    return _MODEL_CACHE.get(model_name)

def extract_dominant_color_hsv(crop: np.ndarray) -> str:
    """Classifies crop dominant color using HSV color histogramming."""
    if crop is None or crop.size == 0:
        return "unknown"
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        mean_v = float(np.mean(v))
        mean_s = float(np.mean(s))
        
        if mean_v < 50:
            return "black"
        if mean_s < 35 and mean_v > 185:
            return "white"
        if mean_s < 40 and 50 <= mean_v <= 185:
            return "grey"
            
        mean_h = float(np.mean(h))
        if mean_h < 10 or mean_h > 170:
            return "red"
        elif 10 <= mean_h < 25:
            return "brown" if mean_v < 120 else "orange"
        elif 25 <= mean_h < 35:
            return "yellow"
        elif 35 <= mean_h < 85:
            return "green"
        elif 85 <= mean_h < 130:
            return "blue"
        elif 130 <= mean_h < 150:
            return "purple"
        elif 150 <= mean_h < 170:
            return "pink"
        return "dark" if mean_v < 100 else "light"
    except Exception:
        return "unknown"

def enrich_detections_with_attributes(frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enriches detections with upper/lower garment color, vehicle color, and carried bag proximity."""
    if not detections or frame is None:
        return detections

    h, w = frame.shape[:2]
    bag_boxes = [
        d for d in detections 
        if d["class_name"] in ["backpack", "handbag", "suitcase"]
    ]

    for det in detections:
        x1 = int(max(0, det["bbox_x1"] * w))
        y1 = int(max(0, det["bbox_y1"] * h))
        x2 = int(min(w, det["bbox_x2"] * w))
        y2 = int(min(h, det["bbox_y2"] * h))
        
        crop_h = max(0, y2 - y1)
        crop_w = max(0, x2 - x1)
        
        if crop_h > 5 and crop_w > 5:
            crop = frame[y1:y2, x1:x2]
            
            if det["class_name"] == "person":
                upper_y1, upper_y2 = int(crop_h * 0.15), int(crop_h * 0.55)
                lower_y1, lower_y2 = int(crop_h * 0.55), int(crop_h * 0.95)
                
                upper_crop = crop[upper_y1:upper_y2, :]
                lower_crop = crop[lower_y1:lower_y2, :]
                
                det["upper_garment_color"] = extract_dominant_color_hsv(upper_crop)
                det["lower_garment_color"] = extract_dominant_color_hsv(lower_crop)
                
                # Check bag co-location (backpack/handbag within proximity)
                cx = (det["bbox_x1"] + det["bbox_x2"]) / 2.0
                cy = (det["bbox_y1"] + det["bbox_y2"]) / 2.0
                
                carries_bag = False
                for bag in bag_boxes:
                    bcx = (bag["bbox_x1"] + bag["bbox_x2"]) / 2.0
                    bcy = (bag["bbox_y1"] + bag["bbox_y2"]) / 2.0
                    dist = np.sqrt((cx - bcx)**2 + (cy - bcy)**2)
                    if dist < 0.25:
                        carries_bag = True
                        break
                det["carries_bag"] = carries_bag
                
            elif det["class_name"] in ["car", "motorcycle", "bicycle", "bus", "truck"]:
                det["vehicle_color"] = extract_dominant_color_hsv(crop)
                
    return detections

class ObjectDetector:
    """
    Object detection service supporting Ultralytics YOLOv8 with single-instance model caching,
    frame downscaling for performance, batched inference, and OpenCV fallback detection.
    """
    def __init__(
        self, 
        model_name: str = settings.YOLO_MODEL_NAME, 
        confidence_threshold: float = settings.DETECTION_CONFIDENCE_THRESHOLD,
        iou_threshold: float = settings.DETECTION_IOU_THRESHOLD,
        max_processing_dim: int = settings.MAX_PROCESSING_RESOLUTION
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_processing_dim = max_processing_dim
        self.yolo_model = get_yolo_model(self.model_name)
        self.use_yolo = self.yolo_model is not None

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int, float]:
        """
        Resizes frame for detection if larger than MAX_PROCESSING_RESOLUTION,
        returning (processed_frame, orig_width, orig_height, scale).
        """
        orig_height, orig_width = frame.shape[:2]
        max_dim = max(orig_width, orig_height)

        if max_dim > self.max_processing_dim:
            scale = self.max_processing_dim / float(max_dim)
            new_w = int(orig_width * scale)
            new_h = int(orig_height * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized, orig_width, orig_height, scale
        
        return frame, orig_width, orig_height, 1.0

    def detect_batch(
        self, 
        sampled_frames: List[Tuple[int, float, np.ndarray]], 
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Runs batched object detection over a list of (frame_number, timestamp, frame) tuples.
        """
        if not sampled_frames:
            return []

        all_detections = []
        num_frames = len(sampled_frames)

        logger.info(f"[DETECTION] Starting object detection on {num_frames} sampled frames (batch_size={batch_size})...")

        for i in range(0, num_frames, batch_size):
            batch = sampled_frames[i : i + batch_size]
            
            if self.use_yolo:
                try:
                    processed_frames = []
                    metadata_list = []
                    for fn, ts, frame in batch:
                        p_frame, orig_w, orig_h, scale = self.preprocess_frame(frame)
                        processed_frames.append(p_frame)
                        metadata_list.append((fn, ts, orig_w, orig_h))

                    # Run inference in no_grad mode if PyTorch available
                    try:
                        import torch
                        with torch.no_grad():
                            results = self.yolo_model.predict(
                                processed_frames, 
                                conf=self.confidence_threshold, 
                                iou=self.iou_threshold,
                                device=settings.DEVICE,
                                verbose=False
                            )
                    except Exception:
                        results = self.yolo_model.predict(
                            processed_frames, 
                            conf=self.confidence_threshold, 
                            iou=self.iou_threshold,
                            device=settings.DEVICE,
                            verbose=False
                        )

                    frame_map = {fn: frame for fn, ts, frame in batch}
                    frame_dets_map = {}

                    for r, (fn, ts, orig_w, orig_h) in zip(results, metadata_list):
                        boxes = r.boxes
                        p_h, p_w = r.orig_shape[:2]
                        fdets = []
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            cls_name = r.names[cls_id].lower()
                            conf = float(box.conf[0])

                            if cls_name in TARGET_CLASSES:
                                xyxy = box.xyxy[0].tolist()
                                fdets.append({
                                    "frame_number": fn,
                                    "timestamp": ts,
                                    "class_name": cls_name,
                                    "confidence": round(conf, 2),
                                    "bbox_x1": round(xyxy[0] / p_w, 4),
                                    "bbox_y1": round(xyxy[1] / p_h, 4),
                                    "bbox_x2": round(xyxy[2] / p_w, 4),
                                    "bbox_y2": round(xyxy[3] / p_h, 4),
                                })
                        
                        enriched = enrich_detections_with_attributes(frame_map.get(fn), fdets)
                        all_detections.extend(enriched)
                    continue
                except Exception as ex:
                    logger.warning(f"[DETECTION] YOLO batch inference failed: {ex}. Falling back to single-frame OpenCV detect.")

            # Fallback per frame in batch
            for fn, ts, frame in batch:
                dets = self.detect_frame(frame, fn, ts)
                all_detections.extend(dets)

        logger.info(f"[DETECTION] Object detection complete. Found {len(all_detections)} total detections.")
        return all_detections

    def detect_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> List[Dict[str, Any]]:
        """Detects objects in a single frame with resolution scaling."""
        p_frame, orig_w, orig_h, scale = self.preprocess_frame(frame)
        p_h, p_w = p_frame.shape[:2]

        if self.use_yolo:
            try:
                try:
                    import torch
                    with torch.no_grad():
                        results = self.yolo_model.predict(
                            p_frame, 
                            conf=self.confidence_threshold, 
                            iou=self.iou_threshold,
                            device=settings.DEVICE,
                            verbose=False
                        )
                except Exception:
                    results = self.yolo_model.predict(
                        p_frame, 
                        conf=self.confidence_threshold, 
                        iou=self.iou_threshold,
                        device=settings.DEVICE,
                        verbose=False
                    )

                detections = []
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
                                "bbox_x1": round(xyxy[0] / p_w, 4),
                                "bbox_y1": round(xyxy[1] / p_h, 4),
                                "bbox_x2": round(xyxy[2] / p_w, 4),
                                "bbox_y2": round(xyxy[3] / p_h, 4),
                            })
                return detections
            except Exception as ex:
                logger.warning(f"[DETECTION] YOLO single-frame inference error ({ex}). Falling back to OpenCV detector.")

        # OpenCV Fallback Detector
        return self._opencv_fallback_detect(p_frame, frame_number, timestamp, p_w, p_h)

    def _opencv_fallback_detect(self, frame: np.ndarray, frame_number: int, timestamp: float, width: int, height: int) -> List[Dict[str, Any]]:
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = width * height * 0.001
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if w >= 12 and h >= 20:
                    aspect_ratio = w / float(h)
                    cls_name = "person" if aspect_ratio < 0.9 else ("car" if aspect_ratio > 1.2 else "backpack")
                    
                    detections.append({
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "class_name": cls_name,
                        "confidence": 0.75,
                        "bbox_x1": round(x / float(width), 4),
                        "bbox_y1": round(y / float(height), 4),
                        "bbox_x2": round((x + w) / float(width), 4),
                        "bbox_y2": round((y + h) / float(height), 4),
                    })

        return detections
