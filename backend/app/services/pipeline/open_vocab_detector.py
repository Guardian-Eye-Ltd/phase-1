import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class OpenVocabDetector:
    """
    Open-Vocabulary Object Detector supporting query-driven natural language object grounding.
    Uses OWL-ViT / Grounding DINO or open-vocabulary heuristic matching over frame visual embeddings.
    """

    _model = None
    _processor = None
    _is_loaded = False

    @classmethod
    def _init_model(cls):
        if cls._is_loaded:
            return
        
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            logger.info("[OPEN-VOCAB] Initializing OwlViT open-vocabulary detector...")
            cls._processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            cls._model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            cls._model.eval()
            cls._is_loaded = True
            logger.info("[OPEN-VOCAB] OwlViT open-vocabulary detector loaded successfully.")
        except Exception as e:
            logger.warning(f"[OPEN-VOCAB] Could not load HuggingFace OwlViT ({e}). Using heuristic open-vocab matcher.")
            cls._is_loaded = False

    @classmethod
    def detect_concepts(
        cls,
        frame: np.ndarray,
        concepts: List[str],
        confidence_threshold: float = 0.20,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Detects user-specified text concepts (e.g. ['red shirt', 'umbrella', 'backpack']) in a frame.
        """
        if not concepts or frame is None:
            return []

        h, w = frame.shape[:2]
        cls._init_model()

        detections = []

        if cls._is_loaded and cls._model is not None and cls._processor is not None:
            try:
                import torch
                from PIL import Image

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                inputs = cls._processor(text=[concepts], images=pil_img, return_tensors="pt")
                with torch.no_grad():
                    outputs = cls._model(**inputs)

                target_sizes = torch.Tensor([pil_img.size[::-1]])
                results = cls._processor.post_process_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=confidence_threshold
                )[0]

                boxes = results["boxes"]
                scores = results["scores"]
                labels = results["labels"]

                for box, score, label in zip(boxes, scores, labels):
                    box_list = [round(i, 2) for i in box.tolist()]
                    concept_name = concepts[label.item()]
                    
                    x1, y1, x2, y2 = box_list
                    detections.append({
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "class_name": concept_name,
                        "confidence": round(float(score), 2),
                        "bbox_x1": round(x1 / float(w), 4),
                        "bbox_y1": round(y1 / float(h), 4),
                        "bbox_x2": round(x2 / float(w), 4),
                        "bbox_y2": round(y2 / float(h), 4),
                        "detector": "owl_vit_open_vocab"
                    })
                return detections
            except Exception as ex:
                logger.warning(f"[OPEN-VOCAB] OwlViT inference error ({ex}). Falling back to heuristic text concept search.")

        # Fallback heuristic open-vocab search based on HSV color and bounding box contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for concept in concepts:
            concept_lower = concept.lower()
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > (w * h * 0.002):
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect = bw / float(bh)

                    # Quick heuristic class assignment
                    matched = False
                    if "car" in concept_lower or "vehicle" in concept_lower or "motorcycle" in concept_lower:
                        if aspect > 1.1:
                            matched = True
                    elif "person" in concept_lower or "shirt" in concept_lower:
                        if aspect < 0.95:
                            matched = True
                    elif "bag" in concept_lower or "backpack" in concept_lower or "umbrella" in concept_lower:
                        matched = True

                    if matched:
                        detections.append({
                            "frame_number": frame_number,
                            "timestamp": timestamp,
                            "class_name": concept,
                            "confidence": 0.65,
                            "bbox_x1": round(bx / float(w), 4),
                            "bbox_y1": round(by / float(h), 4),
                            "bbox_x2": round((bx + bw) / float(w), 4),
                            "bbox_y2": round((by + bh) / float(h), 4),
                            "detector": "heuristic_open_vocab"
                        })

        return detections
