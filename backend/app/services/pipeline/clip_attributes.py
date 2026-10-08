"""
CLIP zero-shot classification with cached text embeddings.

Two fixes over the original per-call implementation:
  * Prompts describe the object ("a photo of a black car"), not a bare word
    ("a photo of a black"), which measurably improved vehicle-colour accuracy.
  * Text embeddings for a label set are computed once, and one image embedding
    is shared across every label set asked of the same crop (previously each
    question re-encoded both the text and the image).

The result is mathematically identical to a full CLIP forward pass: softmax of
logit_scale * cosine(image, text).
"""
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


def _model():
    from app.services.pipeline.object_detector import get_models
    _, _, clip_model, clip_proc = get_models()
    return clip_model, clip_proc


def _as_tensor(x) -> torch.Tensor:
    # transformers >= 5 returns an output object from get_*_features.
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"Unexpected CLIP output type {type(x)}")


@lru_cache(maxsize=64)
def _text_features(prompts: Tuple[str, ...]) -> Optional[torch.Tensor]:
    clip_model, clip_proc = _model()
    if clip_model is None:
        return None
    inputs = clip_proc(text=list(prompts), return_tensors="pt", padding=True).to(settings.DEVICE)
    with torch.no_grad():
        feats = _as_tensor(clip_model.get_text_features(**inputs))
    return feats / feats.norm(dim=-1, keepdim=True)


def _image_features(crop_bgr: np.ndarray) -> Optional[torch.Tensor]:
    clip_model, clip_proc = _model()
    if clip_model is None or crop_bgr is None or crop_bgr.size == 0:
        return None
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    inputs = clip_proc(images=img, return_tensors="pt").to(settings.DEVICE)
    with torch.no_grad():
        feats = _as_tensor(clip_model.get_image_features(**inputs))
    return feats / feats.norm(dim=-1, keepdim=True)


def classify(
    crop_bgr: np.ndarray, label_sets: Dict[str, Tuple[Sequence[str], Sequence[str]]]
) -> Dict[str, Dict[str, float]]:
    """
    label_sets: {question: (labels, prompts)}. Returns {question: {label: prob}}
    using a single image encode. Empty dict when CLIP is unavailable.
    """
    if not settings.ENABLE_ATTRIBUTE_CLASSIFICATION:
        return {}
    try:
        image = _image_features(crop_bgr)
        if image is None:
            return {}
        clip_model, _ = _model()
        scale = clip_model.logit_scale.exp()
        out: Dict[str, Dict[str, float]] = {}
        for question, (labels, prompts) in label_sets.items():
            text = _text_features(tuple(prompts))
            if text is None:
                continue
            with torch.no_grad():
                probs = (scale * image @ text.T).softmax(dim=-1)[0].cpu().numpy()
            out[question] = {label: float(p) for label, p in zip(labels, probs)}
        return out
    except Exception as e:
        logger.warning(f"[CLIP] Attribute inference error: {e}")
        return {}


def top(probs: Dict[str, float]) -> Tuple[Optional[str], float, float]:
    """(best label, its probability, margin over the runner-up)."""
    if not probs:
        return None, 0.0, 0.0
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][0], ranked[0][1], ranked[0][1] - second
