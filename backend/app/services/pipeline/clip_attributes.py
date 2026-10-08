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

Batching: CLIP is ~90% of the detection stage, and one crop at a time is the
slowest way to run it on CPU. The detector therefore enriches each batch of
frames twice: once inside recording() on throw-away copies, which notes every
crop classify() is asked about without running CLIP; then inside serving(),
with all those crops encoded in one batched forward pass by encode_batch().
The attribute logic itself is unchanged, and batched features equal per-crop
features to ~1e-7.
"""
import hashlib
import logging
import threading
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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


_state = threading.local()   # .recording: list of crops | None, .features: dict | None

ENCODE_BATCH_SIZE = 32


def _key(crop_bgr: np.ndarray) -> bytes:
    """Exact content key for a crop (shape + pixels)."""
    h = hashlib.blake2b(np.ascontiguousarray(crop_bgr).tobytes(), digest_size=16)
    h.update(repr(crop_bgr.shape).encode())
    return h.digest()


def _encode(crops_bgr: Sequence[np.ndarray]) -> Optional[torch.Tensor]:
    clip_model, clip_proc = _model()
    if clip_model is None:
        return None
    imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
    inputs = clip_proc(images=imgs, return_tensors="pt").to(settings.DEVICE)
    with torch.no_grad():
        feats = _as_tensor(clip_model.get_image_features(**inputs))
    return feats / feats.norm(dim=-1, keepdim=True)


def _image_features(crop_bgr: np.ndarray) -> Optional[torch.Tensor]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    served = getattr(_state, "features", None)
    if served is not None:
        hit = served.get(_key(crop_bgr))
        if hit is not None:
            return hit
    return _encode([crop_bgr])


@contextmanager
def recording() -> Iterator[List[np.ndarray]]:
    """classify() records each crop it is asked about and returns {} without running CLIP."""
    crops: List[np.ndarray] = []
    _state.recording = crops
    try:
        yield crops
    finally:
        _state.recording = None


@contextmanager
def serving(features: Dict[bytes, torch.Tensor]) -> Iterator[None]:
    """classify() takes image features from encode_batch() output (per-crop fallback on a miss)."""
    _state.features = features
    try:
        yield
    finally:
        _state.features = None


def encode_batch(crops_bgr: Sequence[np.ndarray]) -> Dict[bytes, torch.Tensor]:
    """Encode distinct crops in batches of ENCODE_BATCH_SIZE; {} when CLIP is unavailable."""
    unique: Dict[bytes, np.ndarray] = {}
    for c in crops_bgr:
        if c is not None and c.size:
            unique.setdefault(_key(c), c)
    keys = list(unique)
    out: Dict[bytes, torch.Tensor] = {}
    try:
        for i in range(0, len(keys), ENCODE_BATCH_SIZE):
            chunk = keys[i:i + ENCODE_BATCH_SIZE]
            feats = _encode([unique[k] for k in chunk])
            if feats is None:
                return {}
            for j, k in enumerate(chunk):
                out[k] = feats[j:j + 1]
    except Exception as e:
        logger.warning(f"[CLIP] Batched encode failed, falling back to per-crop: {e}")
        return {}
    return out


def classify(
    crop_bgr: np.ndarray, label_sets: Dict[str, Tuple[Sequence[str], Sequence[str]]]
) -> Dict[str, Dict[str, float]]:
    """
    label_sets: {question: (labels, prompts)}. Returns {question: {label: prob}}
    using a single image encode. Empty dict when CLIP is unavailable.
    """
    if not settings.ENABLE_ATTRIBUTE_CLASSIFICATION:
        return {}
    recorder = getattr(_state, "recording", None)
    if recorder is not None:
        if crop_bgr is not None and crop_bgr.size:
            recorder.append(crop_bgr)
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
