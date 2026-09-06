import asyncio
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
from fastapi import UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, or_
from app.models.evidence import Evidence, EvidenceStatus
from app.services.storage_service import storage_service
from app.services.metadata_service import VideoMetadataService
from app.services.audit_service import AuditService
from app.database.session import SessionLocal

class EvidenceService:
    """
    Service layer handling CCTV video evidence upload, forensic integrity checks, duplicate detection, and streaming.
    """

    @staticmethod
    async def process_background_pipeline(evidence_id: int):
        """
        Simulates background video processing pipeline (Celery / Redis / RQ mock).
        Transitions state: UPLOADED -> QUEUED -> PROCESSING -> COMPLETED.
        """
        await asyncio.sleep(1)
        async with SessionLocal() as db:
            result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
            evidence = result.scalar_one_or_none()
            if evidence:
                evidence.status = EvidenceStatus.QUEUED
                await db.commit()

        await asyncio.sleep(2)
        async with SessionLocal() as db:
            result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
            evidence = result.scalar_one_or_none()
            if evidence:
                evidence.status = EvidenceStatus.PROCESSING
                await db.commit()

        await asyncio.sleep(3)
        async with SessionLocal() as db:
            result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
            evidence = result.scalar_one_or_none()
            if evidence:
                evidence.status = EvidenceStatus.COMPLETED
                await db.commit()

    @staticmethod
    async def upload_evidence(
        db: AsyncSession,
        file: UploadFile,
        uploaded_by_user_id: int
    ) -> Tuple[Evidence, bool]:
        """
        Upload CCTV video file, calculate SHA-256 hash, detect duplicates, extract metadata, and save record.
        Returns (evidence_model, is_duplicate).
        """
        # Validate filename extension
        filename = file.filename or "evidence.mp4"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        allowed_exts = ["mp4", "webm", "avi", "mov", "mkv"]
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '.{ext}'. Allowed formats: {', '.join(allowed_exts)}"
            )

        # Stream file to storage & compute SHA-256
        stored_filename, file_path, file_size, sha256_hash = await storage_service.save_evidence(file)

        # Check duplicate hash in DB
        existing_result = await db.execute(
            select(Evidence).where(Evidence.sha256_hash == sha256_hash)
        )
        existing_evidence = existing_result.scalar_one_or_none()
        if existing_evidence:
            # Delete duplicate saved file from storage to avoid waste
            try:
                os.remove(file_path)
            except Exception:
                pass

            # Log audit for duplicate attempt
            await AuditService.log_action(
                db=db,
                user_id=uploaded_by_user_id,
                action="UPLOAD_EVIDENCE_DUPLICATE",
                resource_type="evidence",
                resource_id=str(existing_evidence.id),
                metadata={"sha256_hash": sha256_hash, "original_filename": filename}
            )
            return existing_evidence, True

        # Extract technical video metadata
        meta = VideoMetadataService.extract_metadata(file_path)

        # Create new Evidence DB Record
        evidence = Evidence(
            original_filename=filename,
            stored_filename=stored_filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=file.content_type or "video/mp4",
            duration=meta.get("duration"),
            resolution=meta.get("resolution"),
            fps=meta.get("fps"),
            status=EvidenceStatus.UPLOADED,
            sha256_hash=sha256_hash,
            uploaded_by=uploaded_by_user_id
        )

        db.add(evidence)
        await db.commit()
        await db.refresh(evidence)

        # Record Audit Log
        await AuditService.log_action(
            db=db,
            user_id=uploaded_by_user_id,
            action="UPLOAD_EVIDENCE",
            resource_type="evidence",
            resource_id=str(evidence.id),
            metadata={
                "original_filename": filename,
                "sha256_hash": sha256_hash,
                "file_size": file_size,
                "mime_type": evidence.mime_type
            }
        )

        return evidence, False

    @staticmethod
    async def get_evidence_list(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        status_filter: str | None = None,
        search_query: str | None = None
    ) -> Tuple[List[Evidence], int]:
        """
        List evidence with pagination, status filters, and search by filename/hash.
        """
        query = select(Evidence)

        if status_filter:
            try:
                enum_status = EvidenceStatus(status_filter.upper())
                query = query.where(Evidence.status == enum_status)
            except ValueError:
                pass

        if search_query:
            term = f"%{search_query}%"
            query = query.where(
                or_(
                    Evidence.original_filename.ilike(term),
                    Evidence.sha256_hash.ilike(term)
                )
            )

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()

        query = query.order_by(desc(Evidence.uploaded_at)).offset(skip).limit(limit)
        results = await db.execute(query)
        evidence_list = list(results.scalars().all())

        return evidence_list, total

    @staticmethod
    async def get_evidence_by_id(db: AsyncSession, evidence_id: int) -> Evidence | None:
        result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
        return result.scalar_one_or_none()

    @staticmethod
    def stream_evidence_video(file_path: str, range_header: str | None = None) -> StreamingResponse:
        """
        Provides HTTP Range streaming for HTML5 video player playback.
        """
        path = Path(file_path)
        if not path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence video file missing on server.")

        file_size = path.stat().st_size
        start = 0
        end = file_size - 1

        if range_header:
            bytes_range = range_header.replace("bytes=", "").split("-")
            if bytes_range[0]:
                start = int(bytes_range[0])
            if len(bytes_range) > 1 and bytes_range[1]:
                end = int(bytes_range[1])

        chunk_size = (end - start) + 1

        # Detect MIME type from extension
        ext = path.suffix.lower()
        mime_map = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
        }
        content_type = mime_map.get(ext, "video/mp4")

        def iter_file():
            with open(path, "rb") as f:
                f.seek(start)
                bytes_left = chunk_size
                while bytes_left > 0:
                    read_len = min(64 * 1024, bytes_left)
                    data = f.read(read_len)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": content_type,
            # Explicit CORS headers — CORSMiddleware doesn't reliably inject these on StreamingResponse
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Range, Authorization, Content-Type",
            "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length, Content-Type",
        }

        status_code = status.HTTP_206_PARTIAL_CONTENT if range_header else status.HTTP_200_OK
        return StreamingResponse(iter_file(), headers=headers, status_code=status_code)

