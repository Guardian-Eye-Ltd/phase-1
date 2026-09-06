import os
import hashlib
import uuid
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, BinaryIO
from fastapi import UploadFile
from app.core.config import settings

class BaseStorageService(ABC):
    """
    Abstract Storage Interface enabling seamless transitions between Local, MinIO, S3, or Azure Blob storage.
    """
    
    @abstractmethod
    async def save_evidence(self, file: UploadFile) -> Tuple[str, str, int, str]:
        """
        Saves uploaded file safely, returning (stored_filename, absolute_file_path, file_size, sha256_hash).
        """
        pass

    @abstractmethod
    def get_file_path(self, stored_filename: str) -> Path:
        """
        Retrieves absolute safe path to stored file.
        """
        pass

    @abstractmethod
    def verify_integrity(self, stored_filename: str, expected_sha256: str) -> bool:
        """
        Verifies evidence SHA-256 hash against saved content on disk.
        """
        pass


class LocalStorageService(BaseStorageService):
    """
    Local filesystem implementation of digital forensics evidence storage with path traversal protection.
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or settings.STORAGE_DIR).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize input filename to prevent directory traversal and special character injection attacks.
        """
        clean_name = os.path.basename(filename)
        # Strip path traversal symbols
        clean_name = clean_name.replace("..", "").replace("/", "").replace("\\", "")
        if not clean_name:
            clean_name = "evidence_file.mp4"
        return clean_name

    async def save_evidence(self, file: UploadFile) -> Tuple[str, str, int, str]:
        """
        Reads file in 64KB chunks, writes to isolated storage, and computes SHA-256 hash in a single streaming pass.
        """
        safe_original_name = self._sanitize_filename(file.filename or "evidence.mp4")
        unique_prefix = uuid.uuid4().hex[:16]
        stored_filename = f"{unique_prefix}_{safe_original_name}"
        
        target_path = (self.base_dir / stored_filename).resolve()
        
        # Enforce path traversal check: target path must be inside base_dir
        if not str(target_path).startswith(str(self.base_dir)):
            raise ValueError("Path traversal attempt detected in filename.")

        hasher = hashlib.sha256()
        file_size = 0

        # Stream file to disk while calculating SHA-256
        with open(target_path, "wb") as destination:
            while chunk := await file.read(64 * 1024):
                hasher.update(chunk)
                destination.write(chunk)
                file_size += len(chunk)

        # Reset file cursor position
        await file.seek(0)
        sha256_hash = hasher.hexdigest()

        return stored_filename, str(target_path), file_size, sha256_hash

    def get_file_path(self, stored_filename: str) -> Path:
        safe_name = os.path.basename(stored_filename)
        target_path = (self.base_dir / safe_name).resolve()
        if not str(target_path).startswith(str(self.base_dir)) or not target_path.exists():
            raise FileNotFoundError(f"Evidence file '{stored_filename}' not found or access denied.")
        return target_path

    def verify_integrity(self, stored_filename: str, expected_sha256: str) -> bool:
        try:
            file_path = self.get_file_path(stored_filename)
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(64 * 1024):
                    hasher.update(chunk)
            return hasher.hexdigest().lower() == expected_sha256.lower()
        except Exception:
            return False

# Global instance for injection
storage_service = LocalStorageService()
