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
                logger.error(f"[OPEN-VOCAB] OwlViT inference error ({ex}). Zero heuristic fallbacks allowed.")
                return []

        logger.warning("[OPEN-VOCAB] Open-vocabulary neural detector unavailable. Zero heuristic contour fallbacks executed.")
        return []

