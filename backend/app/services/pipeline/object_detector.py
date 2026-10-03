import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)

# COCO target classes for forensic surveillance
TARGET_CLASSES = {
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "backpack", "handbag", "suitcase", "cell phone"
}

_MODEL_CACHE: Dict[str, Any] = {}

def get_yolo_model(model_name: str):
    """Loads and caches the YOLO model once in memory. Fails loudly on error."""
    clean_name = model_name.strip().lower()
    if not clean_name.endswith(".pt") and not clean_name.endswith(".yaml") and not clean_name.endswith(".onnx") and not clean_name.endswith(".engine"):
        clean_name += ".pt"

    if model_name not in _MODEL_CACHE or _MODEL_CACHE[model_name] is None:
        try:
            from ultralytics import YOLO
            logger.info(f"[DETECTION] Loading YOLO model '{clean_name}' (requested: '{model_name}') on device '{settings.DEVICE}'...")
            model = YOLO(clean_name)
            # Warm-up pass to allocate CUDA/CPU graph
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy_frame, verbose=False, device=settings.DEVICE)
            _MODEL_CACHE[model_name] = model
            _MODEL_CACHE[clean_name] = model
            logger.info(f"[DETECTION] Successfully loaded and cached YOLO model '{clean_name}'.")
        except Exception as e:
            logger.critical(f"[DETECTION] FATAL: Failed to initialize YOLO model '{clean_name}': {e}", exc_info=True)
            raise RuntimeError(f"YOLO model initialization failed for '{model_name}': {e}")
    return _MODEL_CACHE[model_name]

def extract_dominant_color_hsv(crop: np.ndarray) -> str:
    """Classifies dominant color using HSV thresholds."""
    if crop is None or crop.size == 0:
        return "unknown"
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        mean_v = float(np.mean(v))
        mean_s = float(np.mean(s))
        
        if mean_v < 45:
            return "black"
        if mean_s < 30 and mean_v > 190:
            return "white"
        if mean_s < 35 and 45 <= mean_v <= 190:
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
    """Enriches detected bounding boxes with HSV color attributes and bag proximity."""
    if not detections or frame is None:
        return detections

    h, w = frame.shape[:2]
    bag_boxes = [
        d for d in detections 
        if d["class_name"] in ["backpack", "handbag", "suitcase"]
    ]

    for det in detections:
        x1 = max(0, int(det["bbox_x1"] * w))
        y1 = max(0, int(det["bbox_y1"] * h))
        x2 = min(w, int(det["bbox_x2"] * w))
        y2 = min(h, int(det["bbox_y2"] * h))
        
        crop_h = y2 - y1
        crop_w = x2 - x1
        
        if crop_h > 10 and crop_w > 10:
            crop = frame[y1:y2, x1:x2]
            
            if det["class_name"] == "person":
                upper_crop = crop[int(crop_h * 0.15):int(crop_h * 0.55), :]
                lower_crop = crop[int(crop_h * 0.55):int(crop_h * 0.95), :]
                
                det["upper_garment_color"] = extract_dominant_color_hsv(upper_crop)
                det["lower_garment_color"] = extract_dominant_color_hsv(lower_crop)
                
                # Check bag proximity association
                cx = (det["bbox_x1"] + det["bbox_x2"]) / 2.0
                cy = (det["bbox_y1"] + det["bbox_y2"]) / 2.0
                
                carries_bag = False
                for bag in bag_boxes:
                    bcx = (bag["bbox_x1"] + bag["bbox_x2"]) / 2.0
                    bcy = (bag["bbox_y1"] + bag["bbox_y2"]) / 2.0
                    if np.sqrt((cx - bcx)**2 + (cy - bcy)**2) < 0.20:
                        carries_bag = True
                        break
                det["carries_bag"] = carries_bag
                
            elif det["class_name"] in ["car", "motorcycle", "bicycle", "bus", "truck"]:
                det["vehicle_color"] = extract_dominant_color_hsv(crop)
                
    return detections

class ObjectDetector:
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

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        orig_h, orig_w = frame.shape[:2]
        max_dim = max(orig_w, orig_h)

        if max_dim > self.max_processing_dim:
            scale = self.max_processing_dim / float(max_dim)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized, orig_w, orig_h
        
        return frame, orig_w, orig_h

    def detect_batch(
        self,
        sampled_frames: List[Tuple[int, float, np.ndarray]],
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        if not sampled_frames:
            return []

        all_detections = []
        num_frames = len(sampled_frames)
        logger.info(f"[DETECTION] Executing YOLO batch detection on {num_frames} frames (batch_size={batch_size})...")

        for i in range(0, num_frames, batch_size):
            batch = sampled_frames[i : i + batch_size]
            processed_frames = []
            frame_meta = []

            for fn, ts, frame in batch:
                p_frame, orig_w, orig_h = self.preprocess_frame(frame)
                processed_frames.append(p_frame)
                frame_meta.append((fn, ts, orig_w, orig_h, frame))

            results = self.yolo_model.predict(
                processed_frames,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=settings.DEVICE,
                verbose=False
            )

            for result, (fn, ts, orig_w, orig_h, raw_frame) in zip(results, frame_meta):
                boxes = result.boxes
                p_h, p_w = result.orig_shape[:2]
                frame_dets = []

                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id].lower().strip()
                    conf = float(box.conf[0])

                    if cls_name in TARGET_CLASSES:
                        xyxy = box.xyxy[0].tolist()
                        frame_dets.append({
                            "frame_number": fn,
                            "timestamp": ts,
                            "class_name": cls_name,
                            "confidence": round(conf, 3),
                            "bbox_x1": round(max(0.0, min(1.0, xyxy[0] / p_w)), 4),
                            "bbox_y1": round(max(0.0, min(1.0, xyxy[1] / p_h)), 4),
                            "bbox_x2": round(max(0.0, min(1.0, xyxy[2] / p_w)), 4),
                            "bbox_y2": round(max(0.0, min(1.0, xyxy[3] / p_h)), 4),
                            "upper_garment_color": "unknown",
                            "lower_garment_color": "unknown",
                            "vehicle_color": "unknown",
                            "carries_bag": False
                        })

                enriched = enrich_detections_with_attributes(raw_frame, frame_dets)
                all_detections.extend(enriched)

        logger.info(f"[DETECTION] Completed detection: {len(all_detections)} validated forensic entities identified.")
        return all_detections
