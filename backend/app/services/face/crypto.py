"""Encryption of face embeddings at rest (Fernet: AES-128-CBC + HMAC-SHA256)."""
import base64
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    if settings.FACE_EMBEDDING_KEY:
        return Fernet(settings.FACE_EMBEDDING_KEY.encode())
    logger.warning(
        "[FACE] FACE_EMBEDDING_KEY is not set; deriving the embedding key from SECRET_KEY. "
        "Set FACE_EMBEDDING_KEY explicitly in production — rotating SECRET_KEY would "
        "otherwise make every stored face embedding unreadable."
    )
    derived = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None,
        info=b"guardianeye-face-embeddings-v1",
    ).derive(settings.SECRET_KEY.encode())
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_embedding(embedding: np.ndarray) -> bytes:
    return _fernet().encrypt(np.asarray(embedding, dtype="<f4").tobytes())


def decrypt_embedding(token: bytes) -> Optional[np.ndarray]:
    """None if the token cannot be decrypted (e.g. the key changed)."""
    try:
        return np.frombuffer(_fernet().decrypt(token), dtype="<f4")
    except InvalidToken:
        return None
