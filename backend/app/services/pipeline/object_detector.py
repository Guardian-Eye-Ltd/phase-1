import cv2
import numpy as np
import torch
import logging
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel
from app.core.config import settings
from app.services.pipeline.alpr_service import ALPRService

logger = logging.getLogger(__name__)

TARGET_CLASSES = {
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "backpack", "handbag", "suitcase"
}

_DET_MODEL = None
_POSE_MODEL = None
_CLIP_MODEL = None
_CLIP_PROCESSOR = None

COLOR_LABELS = ["black", "white", "silver", "grey", "red", "blue", "green", "yellow", "brown"]
HEADWEAR_LABELS = ["hat", "cap", "helmet", "bare head"]
CARRIED_LABELS = ["backpack", "handbag", "suitcase", "box", "none"]
VEHICLE_BODY_LABELS = ["sedan", "SUV", "pickup truck", "hatchback", "truck", "van", "motorcycle"]
UPPER_GARMENT_LABELS = ["t-shirt", "jacket", "coat", "hoodie", "shirt", "sweater"]
LOWER_GARMENT_LABELS = ["jeans", "pants", "shorts", "skirt"]

def get_models():
    """Lazily loads and caches YOLO detection, YOLO pose, and OpenCLIP models."""
    global _DET_MODEL, _POSE_MODEL, _CLIP_MODEL, _CLIP_PROCESSOR
    from ultralytics import YOLO
    
    if _DET_MODEL is None:
        logger.info(f"[DETECTION] Loading YOLO detection model '{settings.YOLO_MODEL_NAME}' on device '{settings.DEVICE}'...")
        _DET_MODEL = YOLO(settings.YOLO_MODEL_NAME)
        
    if _POSE_MODEL is None:
        try:
            logger.info(f"[POSE] Loading YOLO pose model '{settings.YOLO_POSE_MODEL_NAME}' on device '{settings.DEVICE}'...")
            _POSE_MODEL = YOLO(settings.YOLO_POSE_MODEL_NAME)
        except Exception as e:
            logger.warning(f"[POSE] Pose model initialization warning ({e}). Proceeding without pose estimation.")
            _POSE_MODEL = None

    if _CLIP_MODEL is None and settings.ENABLE_ATTRIBUTE_CLASSIFICATION:
        try:
            logger.info(f"[CLIP] Loading OpenCLIP model '{settings.CLIP_MODEL_NAME}' on device '{settings.DEVICE}'...")
            _CLIP_MODEL = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(settings.DEVICE)
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        except Exception as e:
            logger.warning(f"[CLIP] OpenCLIP model initialization warning ({e}). Attribute classification will degrade safely.")
            _CLIP_MODEL = None
            _CLIP_PROCESSOR = None

    return _DET_MODEL, _POSE_MODEL, _CLIP_MODEL, _CLIP_PROCESSOR

def classify_crop_clip(crop_bgr: np.ndarray, labels: List[str]) -> Tuple[str, float]:
    """Zero-shot neural classification over image crops using OpenCLIP."""
    if crop_bgr is None or crop_bgr.size == 0 or not labels:
        return "unknown", 0.0
    try:
        _, _, clip_model, clip_proc = get_models()
        if clip_model is None or clip_proc is None:
            return "unknown", 0.0

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Format labels into structured prompts for enhanced CLIP accuracy
        text_prompts = [f"a photo of a {label}" for label in labels]
        
        inputs = clip_proc(text=text_prompts, images=pil_img, return_tensors="pt", padding=True).to(settings.DEVICE)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            best_idx = int(np.argmax(probs))
            return labels[best_idx], float(probs[best_idx])
    except Exception as e:
        logger.warning(f"[CLIP] Attribute inference error: {e}")
        return "unknown", 0.0

def enrich_detections_with_attributes(
    frame: np.ndarray, 
    detections: List[Dict[str, Any]],
    pose_keypoints_by_box: Optional[Dict[int, np.ndarray]] = None
) -> List[Dict[str, Any]]:
    """
    Enriches detected bounding boxes with OpenCLIP fine-grained entity attributes,
    PaddleOCR ALPR license plate extraction, and pose keypoint landmarks.
    """
    if not detections or frame is None:
        return detections

    h, w = frame.shape[:2]

    for idx, det in enumerate(detections):
        x1 = max(0, int(det["bbox_x1"] * w))
        y1 = max(0, int(det["bbox_y1"] * h))
        x2 = min(w, int(det["bbox_x2"] * w))
        y2 = min(h, int(det["bbox_y2"] * h))
        
        crop_h = y2 - y1
        crop_w = x2 - x1
        
        if crop_h > 15 and crop_w > 15:
            crop = frame[y1:y2, x1:x2]
            
            if det["class_name"] == "person":
                upper_crop = crop[int(crop_h * 0.10):int(crop_h * 0.50), :]
                lower_crop = crop[int(crop_h * 0.50):int(crop_h * 0.90), :]
                head_crop = crop[0:int(crop_h * 0.25), :]
                
                # Zero-shot OpenCLIP attribute classification
                if settings.ENABLE_ATTRIBUTE_CLASSIFICATION:
                    upper_color, _ = classify_crop_clip(upper_crop, COLOR_LABELS)
                    lower_color, _ = classify_crop_clip(lower_crop, COLOR_LABELS)
                    upper_type, _ = classify_crop_clip(upper_crop, UPPER_GARMENT_LABELS)
                    lower_type, _ = classify_crop_clip(lower_crop, LOWER_GARMENT_LABELS)
                    headwear, _ = classify_crop_clip(head_crop, HEADWEAR_LABELS)
                    carried_item, _ = classify_crop_clip(crop, CARRIED_LABELS)
                    
                    det["upper_garment_color"] = upper_color
                    det["lower_garment_color"] = lower_color
                    det["upper_garment_type"] = upper_type
                    det["lower_garment_type"] = lower_type
                    det["headwear"] = headwear
                    det["carries_bag"] = carried_item in ["backpack", "handbag", "suitcase", "box"]
                    det["carried_item"] = carried_item

                # Attach Pose Keypoints if available
                if pose_keypoints_by_box and idx in pose_keypoints_by_box:
                    det["keypoints"] = pose_keypoints_by_box[idx].tolist()
                
            elif det["class_name"] in ["car", "motorcycle", "bicycle", "bus", "truck"]:
                if settings.ENABLE_ATTRIBUTE_CLASSIFICATION:
                    v_color, _ = classify_crop_clip(crop, COLOR_LABELS)
                    v_body, _ = classify_crop_clip(crop, VEHICLE_BODY_LABELS)
                    det["vehicle_color"] = v_color
                    det["vehicle_body_style"] = v_body

                # ALPR License Plate Extraction via PaddleOCR / ALPRService
                if settings.ENABLE_ALPR:
                    plate_num = ALPRService.extract_license_plate(crop)
                    if plate_num:
                        det["license_plate_number"] = plate_num

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
        det_model, pose_model, _, _ = get_models()
        self.yolo_model = det_model
        self.pose_model = pose_model

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

            # Run Primary Bounding Box Detection
            results = self.yolo_model.predict(
                processed_frames,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=settings.DEVICE,
                verbose=False
            )

            # Optional Pose Estimation pass for keypoint extraction
            pose_results = None
            if self.pose_model is not None:
                try:
                    pose_results = self.pose_model.predict(
                        processed_frames,
                        conf=self.confidence_threshold,
                        device=settings.DEVICE,
                        verbose=False
                    )
                except Exception as pe:
                    logger.warning(f"[POSE] Pose prediction pass skipped: {pe}")

            for b_idx, (result, (fn, ts, orig_w, orig_h, raw_frame)) in enumerate(zip(results, frame_meta)):
                boxes = result.boxes
                p_h, p_w = result.orig_shape[:2]
                frame_dets = []
                pose_map: Dict[int, np.ndarray] = {}

                # Match pose keypoints to person boxes in this frame
                if pose_results and b_idx < len(pose_results):
                    pose_res = pose_results[b_idx]
                    if hasattr(pose_res, "keypoints") and pose_res.keypoints is not None:
                        try:
                            kp_data = pose_res.keypoints.xyn.cpu().numpy() # (N, 17, 2)
                            for k_idx, kp in enumerate(kp_data):
                                pose_map[k_idx] = kp
                        except Exception:
                            pass

                person_det_counter = 0

                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id].lower().strip()
                    conf = float(box.conf[0])

                    if cls_name in TARGET_CLASSES:
                        xyxy = box.xyxy[0].tolist()
                        
                        # Preserve forensic precision by mapping normalized coordinates
                        # directly to original image dimensions
                        orig_x1 = max(0.0, min(float(orig_w), (xyxy[0] / float(p_w)) * orig_w))
                        orig_y1 = max(0.0, min(float(orig_h), (xyxy[1] / float(p_h)) * orig_h))
                        orig_x2 = max(0.0, min(float(orig_w), (xyxy[2] / float(p_w)) * orig_w))
                        orig_y2 = max(0.0, min(float(orig_h), (xyxy[3] / float(p_h)) * orig_h))

                        det_entry = {
                            "frame_number": fn,
                            "timestamp": ts,
                            "class_name": cls_name,
                            "confidence": round(conf, 3),
                            "bbox_x1": round(orig_x1 / float(orig_w), 4),
                            "bbox_y1": round(orig_y1 / float(orig_h), 4),
                            "bbox_x2": round(orig_x2 / float(orig_w), 4),
                            "bbox_y2": round(orig_y2 / float(orig_h), 4),
                            "upper_garment_color": "unknown",
                            "lower_garment_color": "unknown",
                            "upper_garment_type": "unknown",
                            "lower_garment_type": "unknown",
                            "headwear": "unknown",
                            "vehicle_color": "unknown",
                            "vehicle_body_style": "unknown",
                            "license_plate_number": None,
                            "carries_bag": False,
                            "keypoints": None
                        }

                        if cls_name == "person":
                            if person_det_counter in pose_map:
                                det_entry["keypoints"] = pose_map[person_det_counter].tolist()
                            person_det_counter += 1

                        frame_dets.append(det_entry)

                enriched = enrich_detections_with_attributes(raw_frame, frame_dets)
                all_detections.extend(enriched)

        logger.info(f"[DETECTION] Completed detection: {len(all_detections)} validated forensic entities identified.")
        return all_detections
