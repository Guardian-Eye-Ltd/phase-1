import cv2
import numpy as np
import torch
import logging
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel
from app.core.config import settings
from app.services.pipeline import clip_attributes
from app.services.pipeline.attribute_extractor import associate_carried_items, enrich_detection
from app.services.pipeline.body_regions import match_pose_to_boxes

logger = logging.getLogger(__name__)

TARGET_CLASSES = {
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "backpack", "handbag", "suitcase"
}

_DET_MODEL = None
_POSE_MODEL = None
_CLIP_MODEL = None
_CLIP_PROCESSOR = None

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
        self.last_diagnostics: Dict[str, int] = {}

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

        # The model runs at a low floor and the job threshold is applied here, so
        # every filtering stage is countable. NMS keeps higher-confidence boxes
        # first, so the set of boxes above the threshold is unchanged.
        model_conf = min(settings.DETECTOR_RAW_CONFIDENCE_FLOOR, self.confidence_threshold)
        diag = {
            "processed_frames": 0,
            "raw_model_detections": 0,
            "confidence_filtered_detections": 0,
            "class_filtered_detections": 0,
        }

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
                conf=model_conf,
                iou=self.iou_threshold,
                device=settings.DEVICE,
                verbose=False
            )
            diag["processed_frames"] += len(processed_frames)

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

            pending = []      # (frame, detection, keypoints, keypoint conf) awaiting attributes
            batch_frame_dets = []
            for b_idx, (result, (fn, ts, orig_w, orig_h, raw_frame)) in enumerate(zip(results, frame_meta)):
                boxes = result.boxes
                p_h, p_w = result.orig_shape[:2]
                sx, sy = orig_w / float(p_w), orig_h / float(p_h)
                frame_dets = []

                # Pose detections in original-frame pixels. They are matched to
                # person boxes by IoU below — the two models do not list people
                # in the same order, so index-based pairing attached one
                # person's keypoints to another.
                pose_boxes, pose_kps, pose_conf = [], None, None
                if pose_results and b_idx < len(pose_results):
                    pose_res = pose_results[b_idx]
                    try:
                        if pose_res.boxes is not None and pose_res.keypoints is not None and len(pose_res.boxes):
                            pb = pose_res.boxes.xyxy.cpu().numpy()
                            pose_boxes = [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in pb]
                            pose_kps = pose_res.keypoints.xy.cpu().numpy() * np.array([sx, sy])
                            if pose_res.keypoints.conf is not None:
                                pose_conf = pose_res.keypoints.conf.cpu().numpy()
                    except Exception as pe:
                        logger.warning(f"[POSE] Could not read pose output: {pe}")

                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id].lower().strip()
                    conf = float(box.conf[0])

                    diag["raw_model_detections"] += 1
                    if conf < self.confidence_threshold:
                        continue
                    diag["confidence_filtered_detections"] += 1

                    if cls_name in TARGET_CLASSES:
                        diag["class_filtered_detections"] += 1
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
                            # Attributes are only ever set by the extractor;
                            # absence means "not determined", never a default.
                            "carries_bag": None,
                            "keypoints": None,
                            "_box_px": (orig_x1, orig_y1, orig_x2, orig_y2),
                        }
                        frame_dets.append(det_entry)

                persons = [d for d in frame_dets if d["class_name"] == "person"]
                assignment = match_pose_to_boxes([d["_box_px"] for d in persons], pose_boxes)
                pose_for = {id(d): j for d, j in zip(persons, assignment)}

                for det_entry in frame_dets:
                    j = pose_for.get(id(det_entry))
                    kps = pose_kps[j] if (j is not None and pose_kps is not None) else None
                    kconf = pose_conf[j] if (j is not None and pose_conf is not None) else None
                    if kps is not None:
                        # Normalised keypoints are what EventEngine consumes.
                        det_entry["keypoints"] = (kps / np.array([orig_w, orig_h])).tolist()
                    pending.append((raw_frame, det_entry, kps, kconf))
                batch_frame_dets.append(frame_dets)

            # Attributes. A dry run on copies records every crop CLIP will be
            # asked about; they are encoded in one batched pass, then the real
            # run reads those features. Same attribute logic, ~1.6x faster.
            with clip_attributes.recording() as crops:
                for raw_frame, det_entry, kps, kconf in pending:
                    enrich_detection(raw_frame, dict(det_entry), kps, kconf)
            with clip_attributes.serving(clip_attributes.encode_batch(crops)):
                for raw_frame, det_entry, kps, kconf in pending:
                    enrich_detection(raw_frame, det_entry, kps, kconf)
                    del det_entry["_box_px"]
            for frame_dets in batch_frame_dets:
                associate_carried_items(frame_dets)
                all_detections.extend(frame_dets)

        self.last_diagnostics = diag
        logger.info(
            "[DETECTION] frames=%d raw=%d conf_filtered=%d class_filtered=%d (threshold=%.2f, model_conf=%.2f)",
            diag["processed_frames"], diag["raw_model_detections"],
            diag["confidence_filtered_detections"], diag["class_filtered_detections"],
            self.confidence_threshold, model_conf,
        )
        return all_detections
