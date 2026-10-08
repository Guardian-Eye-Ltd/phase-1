"""
Face detection + embedding via InsightFace (SCRFD detector, ArcFace recogniser).

Only the detection and recognition modules are loaded. The buffalo_l pack also
ships a gender/age estimator; it is deliberately excluded — GuardianEye does not
infer sensitive personal characteristics.
"""
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.models.face import FaceQuality

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_ID = f"insightface/{settings.FACE_MODEL_PACK}:arcface"
EMBEDDING_DIM = 512   # ArcFace output size; recorded even for faces that are never embedded


@dataclass
class DetectedFace:
    bbox: Tuple[float, float, float, float]   # pixel coords in the image passed to detect()
    det_score: float
    embedding: Optional[np.ndarray] = None     # L2-normalised; see vector()
    embed: Optional[Callable[[], np.ndarray]] = field(default=None, repr=False, compare=False)

    def vector(self) -> np.ndarray:
        """
        The L2-normalised embedding, computed on first use. Recognition costs
        ~0.26 s per face on CPU and a head crop often contains a neighbour's
        face too; callers pick a face by box and score first, so only the face
        actually kept pays for recognition.
        """
        if self.embedding is None:
            self.embedding = self.embed()
        return self.embedding


@dataclass
class QualityAssessment:
    status: str
    sharpness: float
    reasons: List[str] = field(default_factory=list)


def assess_quality(face_crop_bgr: np.ndarray, width_px: int, height_px: int, det_score: float) -> QualityAssessment:
    """
    Decide whether a face is good enough to compare. Size is measured in
    source-frame pixels: upscaling a tiny face does not make it identifiable.
    """
    sharpness = 0.0
    if face_crop_bgr is not None and face_crop_bgr.size:
        gray = cv2.cvtColor(cv2.resize(face_crop_bgr, (112, 112)), cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons = []
    if min(width_px, height_px) < settings.FACE_MIN_SIZE_PX:
        reasons.append(f"face too small ({width_px}x{height_px}px < {settings.FACE_MIN_SIZE_PX}px)")
    if det_score < settings.FACE_MIN_DET_SCORE:
        reasons.append(f"low detection confidence ({det_score:.2f} < {settings.FACE_MIN_DET_SCORE:.2f})")
    if sharpness < settings.FACE_MIN_SHARPNESS:
        reasons.append(f"too blurry (sharpness {sharpness:.1f} < {settings.FACE_MIN_SHARPNESS:.1f})")

    return QualityAssessment(
        status=FaceQuality.LOW_QUALITY if reasons else FaceQuality.USABLE,
        sharpness=round(sharpness, 2),
        reasons=reasons,
    )


class FaceEngine:
    _app = None
    _load_error: Optional[str] = None
    _lock = threading.Lock()

    @classmethod
    def load(cls) -> bool:
        """Lazily load the model. Returns False (with load_error set) if unavailable."""
        if cls._app is not None:
            return True
        if cls._load_error is not None:
            return False
        with cls._lock:
            if cls._app is not None:
                return True
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name=settings.FACE_MODEL_PACK,
                    allowed_modules=["detection", "recognition"],
                )
                app.prepare(ctx_id=-1 if settings.DEVICE == "cpu" else 0,
                            det_size=(settings.FACE_DET_SIZE, settings.FACE_DET_SIZE))
                cls._app = app
                logger.info(f"[FACE] Loaded {settings.FACE_MODEL_PACK} (detection + recognition only).")
                return True
            except Exception as e:
                cls._load_error = f"{type(e).__name__}: {e}"
                logger.warning(f"[FACE] Face model unavailable: {cls._load_error}")
                return False

    @classmethod
    def load_error(cls) -> Optional[str]:
        return cls._load_error

    @classmethod
    def detect(cls, image_bgr: np.ndarray) -> List[DetectedFace]:
        if not cls.load():
            raise RuntimeError(f"Face model unavailable: {cls._load_error}")
        # Same steps as FaceAnalysis.get(), with recognition deferred to
        # DetectedFace.vector() (no add-ons are loaded, so nothing else runs).
        from insightface.app.common import Face
        bboxes, kpss = cls._app.det_model.detect(image_bgr, max_num=0, metric="default")
        recognizer = cls._app.models["recognition"]

        def _embed(face):
            recognizer.get(image_bgr, face)
            return np.asarray(face.normed_embedding, dtype=np.float32)

        out = []
        for i in range(bboxes.shape[0]):
            face = Face(bbox=bboxes[i, 0:4], kps=kpss[i] if kpss is not None else None,
                        det_score=bboxes[i, 4])
            out.append(DetectedFace(
                bbox=tuple(float(v) for v in face.bbox),
                det_score=float(face.det_score),
                embed=lambda face=face: _embed(face),
            ))
        return out
