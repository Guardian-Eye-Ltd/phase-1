"""
Full-system evidence reset.

Wipes every piece of evidence and everything derived from it — DB rows, stored
video files, derived artifacts (keyframes, manifests) and the vector index — so
the platform can start fresh.

Deliberately PRESERVED:
  - users and roles           (so the operator can still log in)
  - cameras, alerts, incidents (configuration / operational records, not evidence)
  - the audit log             (chain of custody: a record of who erased the
                               evidence must outlive the evidence itself)

The reset itself is written to the audit log in the same transaction as the
deletion, so there is never a wipe without a record of it.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.analysis import (
    ActivityInterval, AnalysisJob, Detection, FrameObservation, Keyframe,
    PossibleInteraction, Track,
)
from app.models.audit import AuditLog
from app.models.evidence import Evidence, EvidenceStatus
from app.models.face import FaceObservation
from app.models.observation import TrackAttributeAggregate, VisualAttributeObservation
from app.models.semantic import (
    AIModelExecution, EmbeddingRecord, ForensicDocument, SearchQuery, SearchResult,
    VLMObservation,
)

logger = logging.getLogger(__name__)

RESET_CONFIRMATION_PHRASE = "DELETE ALL EVIDENCE"

# Children before parents. SQLite runs without PRAGMA foreign_keys=ON here, so
# ondelete="CASCADE" never fires — deleting only `evidence` would orphan every
# derived row. Explicit ordering works on both SQLite and Postgres.
_WIPE_ORDER = [
    SearchResult,
    SearchQuery,
    EmbeddingRecord,
    VLMObservation,
    ForensicDocument,
    AIModelExecution,
    TrackAttributeAggregate,
    VisualAttributeObservation,
    FaceObservation,
    PossibleInteraction,
    Keyframe,
    Detection,
    Track,
    FrameObservation,
    ActivityInterval,
    AnalysisJob,
    Evidence,
]


class ResetBlockedError(Exception):
    """Raised when a reset cannot safely run right now."""


def _derived_dir() -> Path:
    return Path(settings.DERIVED_STORAGE_DIR).resolve()


def _storage_dirs() -> List[Path]:
    """Directories whose *contents* are wiped. Resolved the same way StorageService does."""
    return [Path(settings.STORAGE_DIR).resolve(), _derived_dir()]


def _is_safe_to_wipe(path: Path) -> bool:
    """
    Refuse obviously catastrophic targets in case STORAGE_DIR is misconfigured
    (e.g. set to "" or "/"), which would otherwise rmtree the working directory.
    """
    resolved = path.resolve()
    forbidden = {Path(resolved.anchor), Path.home().resolve(), Path.cwd().resolve()}
    return resolved not in forbidden and len(resolved.parts) >= 3


def _running_job_ids() -> List[int]:
    from app.services.analysis_runner import running_analysis_jobs
    return sorted(running_analysis_jobs)


def _dir_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "files": 0, "bytes": 0}
    files = [p for p in path.rglob("*") if p.is_file()]
    return {
        "path": str(path),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


def _list_vector_collections() -> List[str]:
    from app.services.semantic.indexer import get_chroma_client
    client = get_chroma_client()
    # chromadb >= 0.6 returns names; older versions return Collection objects.
    return [c if isinstance(c, str) else c.name for c in client.list_collections()]


def _wipe_vector_store() -> Dict[str, Any]:
    from app.services.semantic.indexer import get_chroma_client
    from app.services.semantic.vector_service import VectorService

    client = get_chroma_client()
    deleted, errors = 0, []
    for name in _list_vector_collections():
        try:
            client.delete_collection(name=name)
            deleted += 1
        except Exception as e:
            errors.append(f"collection '{name}': {e}")

    # VectorService caches a handle to its collection; drop it so the next use
    # re-creates the collection instead of writing to a deleted one.
    VectorService._chroma_client = None
    VectorService._chroma_collection = None
    VectorService._engine_type = "fallback"

    return {"collections_deleted": deleted, "errors": errors}


def _wipe_vector_collections_with_prefix(prefix: str) -> Dict[str, Any]:
    from app.services.semantic.indexer import get_chroma_client

    client = get_chroma_client()
    deleted, errors = 0, []
    for name in _list_vector_collections():
        if not name.startswith(prefix):
            continue
        try:
            client.delete_collection(name=name)
            deleted += 1
        except Exception as e:
            errors.append(f"collection '{name}': {e}")
    return {"collections_deleted": deleted, "errors": errors}


def _wipe_evidence_derived_files(evidence_id: int) -> Dict[str, Any]:
    """Keyframes and sealed manifests for one evidence file; the video is kept."""
    derived = _derived_dir()
    result = {"files_deleted": 0, "errors": []}
    if not _is_safe_to_wipe(derived):
        result["errors"].append(f"Refused to wipe unsafe path: {derived}")
        return result

    # Keyframes and face crops (biometric) for this evidence.
    for sub in ("keyframes", "faces"):
        target = derived / sub / str(evidence_id)
        if not target.is_dir():
            continue
        try:
            result["files_deleted"] += sum(1 for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target)
        except Exception as e:
            result["errors"].append(f"{sub}/{target.name}: {e}")

    for manifest in (derived / "manifests").glob(f"ev{evidence_id}_job*.json"):
        try:
            manifest.unlink()
            result["files_deleted"] += 1
        except Exception as e:
            result["errors"].append(f"{manifest.name}: {e}")
    return result


def _wipe_directory_contents(path: Path) -> Dict[str, Any]:
    """Delete everything inside `path` but keep the directory itself."""
    result = {"path": str(path), "files_deleted": 0, "errors": []}
    if not path.exists():
        return result
    if not _is_safe_to_wipe(path):
        result["errors"].append(f"Refused to wipe unsafe path: {path}")
        return result

    for child in path.iterdir():
        try:
            if child.is_dir():
                result["files_deleted"] += sum(1 for p in child.rglob("*") if p.is_file())
                shutil.rmtree(child)
            else:
                child.unlink()
                result["files_deleted"] += 1
        except Exception as e:
            # Typically a Windows file lock, e.g. a video still being streamed.
            result["errors"].append(f"{child.name}: {e}")
    return result


class SystemResetService:

    @classmethod
    async def preview(cls, db: AsyncSession) -> Dict[str, Any]:
        """What a reset would delete, without deleting anything."""
        tables = {}
        for model in _WIPE_ORDER:
            count = (await db.execute(select(func.count()).select_from(model))).scalar() or 0
            tables[model.__tablename__] = count

        try:
            vector_collections = len(_list_vector_collections())
        except Exception as e:
            logger.warning(f"[RESET] Could not inspect vector store: {e}")
            vector_collections = None

        running = _running_job_ids()
        return {
            "tables": tables,
            "total_rows": sum(tables.values()),
            "storage": [_dir_stats(p) for p in _storage_dirs()],
            "vector_collections": vector_collections,
            "running_jobs": running,
            "can_reset": not running,
            "confirmation_phrase": RESET_CONFIRMATION_PHRASE,
            "preserved": ["users", "roles", "cameras", "alerts", "incidents", "audit_logs"],
        }

    @classmethod
    async def reset(cls, db: AsyncSession, user_id: int) -> Dict[str, Any]:
        running = _running_job_ids()
        if running:
            raise ResetBlockedError(
                f"Analysis job(s) {running} are still running. Wait for them to "
                "finish or cancel them before resetting."
            )

        before = await cls.preview(db)

        # 1. Database — one transaction, audit record included, so the wipe and
        #    its record either both happen or neither does.
        deleted_rows: Dict[str, int] = {}
        for model in _WIPE_ORDER:
            res = await db.execute(delete(model))
            deleted_rows[model.__tablename__] = res.rowcount or 0

        db.add(AuditLog(
            user_id=user_id,
            action="SYSTEM_RESET",
            resource_type="SYSTEM",
            resource_id=None,
            metadata_json=json.dumps({
                "deleted_rows": deleted_rows,
                "storage_before": before["storage"],
                "vector_collections_before": before["vector_collections"],
                "note": "All evidence and derived data erased. Evidence/job IDs "
                        "created after this entry are not the same objects as "
                        "IDs referenced by earlier audit entries.",
            }),
        ))
        await db.commit()
        logger.warning(f"[RESET] user={user_id} erased database rows: {deleted_rows}")

        # 2. Vector index and files. Done after the commit: if a file is locked,
        #    leaving an orphaned file is harmless, whereas DB rows pointing at
        #    already-deleted files would not be.
        errors: List[str] = []
        try:
            vector = _wipe_vector_store()
            errors.extend(vector["errors"])
        except Exception as e:
            vector = {"collections_deleted": 0, "errors": [str(e)]}
            errors.append(f"vector store: {e}")

        storage = []
        for path in _storage_dirs():
            r = _wipe_directory_contents(path)
            storage.append(r)
            errors.extend(r["errors"])

        status = "COMPLETED" if not errors else "COMPLETED_WITH_ERRORS"
        logger.warning(
            f"[RESET] status={status} vector={vector['collections_deleted']} "
            f"files={sum(s['files_deleted'] for s in storage)} errors={len(errors)}"
        )
        return {
            "status": status,
            "deleted_rows": deleted_rows,
            "total_rows_deleted": sum(deleted_rows.values()),
            "vector_collections_deleted": vector["collections_deleted"],
            "storage": storage,
            "files_deleted": sum(s["files_deleted"] for s in storage),
            "errors": errors,
        }

    @classmethod
    async def reset_evidence_analysis(
        cls, db: AsyncSession, evidence_id: int, user_id: int
    ) -> Dict[str, Any]:
        """
        Remove every analysis run of ONE evidence file and everything derived
        from them. The evidence record and its original video are kept so it
        can be re-analysed.
        """
        evidence = (await db.execute(
            select(Evidence).where(Evidence.id == evidence_id)
        )).scalars().first()
        if evidence is None:
            raise LookupError(f"Evidence #{evidence_id} not found.")

        job_ids = set((await db.execute(
            select(AnalysisJob.id).where(AnalysisJob.evidence_id == evidence_id)
        )).scalars().all())
        running = sorted(job_ids & set(_running_job_ids()))
        if running:
            raise ResetBlockedError(
                f"Analysis job(s) {running} for evidence #{evidence_id} are still running."
            )

        deleted_rows: Dict[str, int] = {}
        for model in _WIPE_ORDER:
            if model is Evidence:
                continue
            res = await db.execute(delete(model).where(model.evidence_id == evidence_id))
            deleted_rows[model.__tablename__] = res.rowcount or 0

        evidence.status = EvidenceStatus.UPLOADED
        db.add(AuditLog(
            user_id=user_id,
            action="ANALYSIS_RESET",
            resource_type="EVIDENCE",
            resource_id=str(evidence_id),
            metadata_json=json.dumps({
                "deleted_rows": deleted_rows,
                "analysis_job_ids": sorted(job_ids),
            }),
        ))
        await db.commit()

        errors: List[str] = []
        try:
            vector = _wipe_vector_collections_with_prefix(f"ev{evidence_id}_job")
            errors.extend(vector["errors"])
        except Exception as e:
            vector = {"collections_deleted": 0}
            errors.append(f"vector store: {e}")

        files = _wipe_evidence_derived_files(evidence_id)
        errors.extend(files["errors"])

        return {
            "status": "COMPLETED" if not errors else "COMPLETED_WITH_ERRORS",
            "evidence_id": evidence_id,
            "analysis_job_ids": sorted(job_ids),
            "deleted_rows": deleted_rows,
            "vector_collections_deleted": vector["collections_deleted"],
            "files_deleted": files["files_deleted"],
            "errors": errors,
        }
