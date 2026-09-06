from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, BackgroundTasks, Header
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.database.session import get_db
from app.schemas.evidence import EvidenceResponse, EvidenceListResponse, EvidenceUploadResponse
from app.services.evidence_service import EvidenceService
from app.services.audit_service import AuditService
from app.api.dependencies.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/evidence", tags=["Evidence Forensics"])

@router.post(
    "/upload",
    response_model=EvidenceUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload CCTV video evidence and calculate SHA-256 hash"
)
async def upload_evidence(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Ingests CCTV video file, checks SHA-256 duplicate status, extracts metadata,
    writes to isolated evidence storage, and enqueues background analysis pipeline.
    """
    evidence, is_duplicate = await EvidenceService.upload_evidence(
        db=db,
        file=file,
        uploaded_by_user_id=current_user.id
    )

    if not is_duplicate:
        # Enqueue background pipeline simulation (UPLOADED -> QUEUED -> PROCESSING -> COMPLETED)
        background_tasks.add_task(EvidenceService.process_background_pipeline, evidence.id)
        msg = "Evidence video successfully ingested and integrity hash recorded. Background pipeline enqueued."
    else:
        msg = "Identical evidence hash detected in database. Ingested record matched existing forensic record."

    return EvidenceUploadResponse(
        evidence=EvidenceResponse.model_validate(evidence),
        is_duplicate=is_duplicate,
        message=msg
    )

@router.get(
    "",
    response_model=EvidenceListResponse,
    summary="List CCTV evidence library with pagination and filters"
)
async def get_evidence_list(
    page: int = Query(1, ge=1, description="Page index (1-based)"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status (UPLOADED, QUEUED, PROCESSING, COMPLETED, FAILED)"),
    search: Optional[str] = Query(None, description="Search by filename or SHA-256 hash"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    skip = (page - 1) * size
    items, total = await EvidenceService.get_evidence_list(
        db=db,
        skip=skip,
        limit=size,
        status_filter=status,
        search_query=search
    )

    # Log search audit if search term provided
    if search:
        await AuditService.log_action(
            db=db,
            user_id=current_user.id,
            action="SEARCH_EVIDENCE",
            resource_type="evidence",
            metadata={"query": search, "status_filter": status}
        )

    return EvidenceListResponse(
        items=[EvidenceResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        size=size
    )

@router.get(
    "/{evidence_id}",
    response_model=EvidenceResponse,
    summary="Retrieve single CCTV evidence record metadata"
)
async def get_evidence_details(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    evidence = await EvidenceService.get_evidence_by_id(db, evidence_id)
    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID {evidence_id} not found."
        )

    # Log investigator view action
    await AuditService.log_action(
        db=db,
        user_id=current_user.id,
        action="VIEW_EVIDENCE",
        resource_type="evidence",
        resource_id=str(evidence_id),
        metadata={"filename": evidence.original_filename, "sha256_hash": evidence.sha256_hash}
    )

    return EvidenceResponse.model_validate(evidence)

@router.get(
    "/{evidence_id}/stream",
    summary="Stream CCTV evidence video with HTTP Range support"
)
async def stream_evidence_video(
    evidence_id: int,
    range_header: Optional[str] = Header(None, alias="Range"),
    token: Optional[str] = Query(None, description="JWT token (query param fallback for <video> src tags)"),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    # If get_current_user returned None (no header token), try the query param token
    if current_user is None:
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        from app.authentication.jwt import verify_token, TokenExpiredError, TokenInvalidError
        from app.services.auth_service import AuthService
        try:
            payload = verify_token(token, expected_type="access")
        except (TokenExpiredError, TokenInvalidError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_id = payload.get("user_id")
        current_user = await AuthService.get_user_by_id(db, user_id)
        if not current_user or not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive.",
            )

    evidence = await EvidenceService.get_evidence_by_id(db, evidence_id)
    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID {evidence_id} not found."
        )

    return EvidenceService.stream_evidence_video(evidence.file_path, range_header)

@router.get(
    "/{evidence_id}/metadata",
    summary="Fetch technical forensic metadata of evidence file"
)
async def get_evidence_technical_metadata(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    evidence = await EvidenceService.get_evidence_by_id(db, evidence_id)
    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID {evidence_id} not found."
        )

    return {
        "evidence_id": evidence.id,
        "original_filename": evidence.original_filename,
        "stored_filename": evidence.stored_filename,
        "file_size": evidence.file_size,
        "mime_type": evidence.mime_type,
        "duration_seconds": evidence.duration,
        "resolution": evidence.resolution,
        "fps": evidence.fps,
        "sha256_hash": evidence.sha256_hash,
        "status": evidence.status.value,
        "uploaded_at": evidence.uploaded_at.isoformat(),
        "uploaded_by": evidence.uploaded_by
    }

@router.post(
    "/{evidence_id}/reprocess",
    response_model=EvidenceResponse,
    summary="Trigger background state reprocessing simulation"
)
async def reprocess_evidence(
    evidence_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    evidence = await EvidenceService.get_evidence_by_id(db, evidence_id)
    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence with ID {evidence_id} not found."
        )

    background_tasks.add_task(EvidenceService.process_background_pipeline, evidence.id)
    return EvidenceResponse.model_validate(evidence)
