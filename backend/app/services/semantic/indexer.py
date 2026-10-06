import os
import logging
import asyncio
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sentence_transformers import SentenceTransformer
import chromadb

from app.database.session import SessionLocal
from app.models.analysis import AnalysisJob, JobStatus
from app.models.semantic import ForensicDocument, EmbeddingRecord, VLMObservation
from app.services.semantic.document_generator import ForensicDocumentGenerator
from app.services.semantic.vlm_service import VisionLanguageService
from app.core.config import settings

logger = logging.getLogger(__name__)

# In-memory indexing status tracking (keyed by evidence_id)
_indexing_status: Dict[int, Dict[str, Any]] = {}

# Lazy-loaded embedding model & Chroma client cache
_EMBEDDING_MODEL = None
_CHROMA_CLIENT = None


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info(
            f"[EMBEDDING] Loading sentence-transformer '{settings.EMBEDDING_MODEL_NAME}' "
            f"on device '{settings.DEVICE}'..."
        )
        _EMBEDDING_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=settings.DEVICE)
    return _EMBEDDING_MODEL


def get_chroma_client():
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    return _CHROMA_CLIENT


def _chroma_collection_name(evidence_id: int, analysis_job_id: int) -> str:
    """
    Returns a deterministic, job-scoped ChromaDB collection name.
    Each analysis job gets its own isolated vector collection — no cross-job contamination.
    """
    return f"ev{evidence_id}_job{analysis_job_id}"


async def _get_latest_completed_job_id(evidence_id: int, db: AsyncSession) -> Optional[int]:
    """Fetch the analysis_job_id of the most recently completed job for this evidence."""
    res = await db.execute(
        select(AnalysisJob)
        .where(
            AnalysisJob.evidence_id == evidence_id,
            AnalysisJob.status == JobStatus.COMPLETED
        )
        .order_by(AnalysisJob.completed_at.desc())
    )
    job = res.scalars().first()
    return job.id if job else None


class SemanticIndexer:

    @classmethod
    def get_indexing_status(cls, evidence_id: int) -> Dict[str, Any]:
        return _indexing_status.get(evidence_id, {
            "evidence_id": evidence_id,
            "status": "NOT_STARTED",
            "progress": 0.0,
            "stage": "IDLE",
            "total_documents": 0,
            "total_vlm_observations": 0,
            "total_embeddings": 0
        })

    @classmethod
    async def index_evidence_async(cls, evidence_id: int, analysis_job_id: int):
        """
        Orchestrates document generation → sentence-transformer embedding → ChromaDB upsert.

        ISOLATION GUARANTEE:
        - Only ForensicDocuments and EmbeddingRecords for THIS analysis_job_id are deleted/recreated.
        - Every prior job's documents remain intact in both the SQL DB and a separate ChromaDB collection.
        - ChromaDB collection is named ev{evidence_id}_job{analysis_job_id} (per-job).
        """
        _indexing_status[evidence_id] = {
            "evidence_id": evidence_id,
            "analysis_job_id": analysis_job_id,
            "status": "PROCESSING",
            "progress": 10.0,
            "stage": "GENERATING_DOCUMENTS",
            "total_documents": 0,
            "total_vlm_observations": 0,
            "total_embeddings": 0
        }

        async with SessionLocal() as db:
            try:
                # 1. Clean up only THIS job's documents & VLM observations — other jobs are untouched
                await db.execute(
                    delete(EmbeddingRecord).where(
                        EmbeddingRecord.evidence_id == evidence_id,
                        EmbeddingRecord.analysis_job_id == analysis_job_id
                    )
                )
                await db.execute(
                    delete(ForensicDocument).where(
                        ForensicDocument.evidence_id == evidence_id,
                        ForensicDocument.analysis_job_id == analysis_job_id
                    )
                )
                await db.execute(
                    delete(VLMObservation).where(
                        VLMObservation.evidence_id == evidence_id,
                        VLMObservation.analysis_job_id == analysis_job_id
                    )
                )
                await db.commit()
                logger.info(
                    f"[INDEXER] Cleared existing documents and VLM observations for evidence #{evidence_id}, "
                    f"job #{analysis_job_id}. Other job documents preserved."
                )

                # 2. Run Vision-Language analysis on forensic keyframes
                vlm_obs_list = await VisionLanguageService.analyze_keyframes_for_job(
                    db=db,
                    evidence_id=evidence_id,
                    analysis_job_id=analysis_job_id
                )
                total_vlm = len(vlm_obs_list) if vlm_obs_list else 0

                # 2. Generate structured ForensicDocuments scoped to this job
                docs = await ForensicDocumentGenerator.generate_documents_for_job(
                    db=db,
                    evidence_id=evidence_id,
                    analysis_job_id=analysis_job_id
                )

                if not docs:
                    logger.warning(
                        f"[INDEXER] No documents generated for evidence #{evidence_id}, "
                        f"job #{analysis_job_id}. Pipeline may have produced no observations."
                    )
                    _indexing_status[evidence_id] = {
                        "evidence_id": evidence_id,
                        "analysis_job_id": analysis_job_id,
                        "status": "COMPLETED",
                        "progress": 100.0,
                        "stage": "READY",
                        "total_documents": 0,
                        "total_vlm_observations": 0,
                        "total_embeddings": 0
                    }
                    return

                # Re-query saved documents to get DB-assigned IDs
                res_docs = await db.execute(
                    select(ForensicDocument).where(
                        ForensicDocument.evidence_id == evidence_id,
                        ForensicDocument.analysis_job_id == analysis_job_id
                    )
                )
                saved_docs = res_docs.scalars().all()

                _indexing_status[evidence_id].update({
                    "progress": 50.0,
                    "stage": "GENERATING_EMBEDDINGS",
                    "total_documents": len(saved_docs)
                })
                logger.info(
                    f"[INDEXER] Generating embeddings for {len(saved_docs)} documents "
                    f"(evidence #{evidence_id}, job #{analysis_job_id})..."
                )

                # 3. Generate Vector Embeddings (off the event-loop thread)
                model = get_embedding_model()
                doc_contents = [d.content for d in saved_docs]

                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: model.encode(doc_contents, normalize_embeddings=True).tolist()
                )

                # 4. Push into job-scoped ChromaDB collection
                chroma_client = get_chroma_client()
                collection_name = _chroma_collection_name(evidence_id, analysis_job_id)

                # Delete and recreate for idempotency (retry-safe)
                try:
                    chroma_client.delete_collection(name=collection_name)
                    logger.info(f"[INDEXER] Deleted existing ChromaDB collection '{collection_name}'.")
                except Exception:
                    pass

                collection = chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"[INDEXER] Created ChromaDB collection '{collection_name}'.")

                vector_ids = [f"doc_{d.id}" for d in saved_docs]
                metadatas = [
                    {
                        "doc_id": d.id,
                        "evidence_id": d.evidence_id,
                        "analysis_job_id": analysis_job_id,
                        "track_id": d.track_id if d.track_id is not None else -1,
                        "document_type": d.document_type.value,
                        "start_time": float(d.start_time or 0.0),
                        "end_time": float(d.end_time or 0.0)
                    }
                    for d in saved_docs
                ]

                collection.add(
                    ids=vector_ids,
                    embeddings=embeddings,
                    documents=doc_contents,
                    metadatas=metadatas
                )
                logger.info(
                    f"[INDEXER] Upserted {len(saved_docs)} vectors into '{collection_name}'."
                )

                # 5. Persist EmbeddingRecords in DB (job-scoped)
                embed_records = [
                    EmbeddingRecord(
                        evidence_id=evidence_id,
                        analysis_job_id=analysis_job_id,
                        forensic_document_id=d.id,
                        vector_id=f"doc_{d.id}",
                        embedding_model=settings.EMBEDDING_MODEL_NAME,
                        document_type=d.document_type.value,
                        source_type=d.source_type.value
                    )
                    for d in saved_docs
                ]
                db.add_all(embed_records)
                await db.commit()

                _indexing_status[evidence_id] = {
                    "evidence_id": evidence_id,
                    "analysis_job_id": analysis_job_id,
                    "status": "COMPLETED",
                    "progress": 100.0,
                    "stage": "READY",
                    "total_documents": len(saved_docs),
                    "total_vlm_observations": total_vlm,
                    "total_embeddings": len(saved_docs)
                }
                logger.info(
                    f"[INDEXER] ✅ Successfully indexed {len(saved_docs)} documents "
                    f"for evidence #{evidence_id}, job #{analysis_job_id}."
                )

            except Exception as e:
                logger.exception(
                    f"[INDEXER] ❌ Failed to index evidence #{evidence_id}, job #{analysis_job_id}: {e}"
                )
                _indexing_status[evidence_id] = {
                    "evidence_id": evidence_id,
                    "analysis_job_id": analysis_job_id,
                    "status": "FAILED",
                    "progress": 0.0,
                    "stage": "ERROR",
                    "error": str(e),
                    "total_documents": 0,
                    "total_vlm_observations": 0,
                    "total_embeddings": 0
                }
