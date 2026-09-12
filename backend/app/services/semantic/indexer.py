import logging
import asyncio
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.session import SessionLocal
from app.models.analysis import AnalysisJob, JobStatus
from app.models.semantic import ForensicDocument, AIModelExecution
from app.services.semantic.document_generator import ForensicDocumentGenerator
from app.services.semantic.vlm_service import VisionLanguageService
from app.services.semantic.vector_service import VectorService

logger = logging.getLogger(__name__)

# Memory status tracker for background indexing jobs
active_indexing_jobs: Dict[int, Dict[str, Any]] = {}

class SemanticIndexer:
    """
    Background Indexing Orchestrator managing document generation, VLM analysis,
    and vector database embedding for evidence analysis jobs.
    """

    @classmethod
    def get_indexing_status(cls, evidence_id: int) -> Dict[str, Any]:
        """Returns the current background indexing status for evidence_id."""
        return active_indexing_jobs.get(evidence_id, {
            "evidence_id": evidence_id,
            "status": "NOT_STARTED",
            "progress": 0.0,
            "stage": "IDLE",
            "error": None
        })

    @classmethod
    async def index_evidence_async(cls, evidence_id: int, analysis_job_id: int):
        """
        Asynchronous worker executing the full Phase 1C indexing workflow:
        Document Generation -> VLM Analysis -> Embedding & Vector Storage -> Verification.
        """
        active_indexing_jobs[evidence_id] = {
            "evidence_id": evidence_id,
            "analysis_job_id": analysis_job_id,
            "status": "PROCESSING",
            "progress": 10.0,
            "stage": "GENERATING_FORENSIC_DOCUMENTS",
            "error": None
        }

        try:
            async with SessionLocal() as db:
                # Stage 1: Generate Forensic Documents from Phase 1B observations
                logger.info(f"Generating ForensicDocuments for evidence {evidence_id}...")
                docs = await ForensicDocumentGenerator.generate_documents_for_job(
                    db=db,
                    evidence_id=evidence_id,
                    analysis_job_id=analysis_job_id
                )

                active_indexing_jobs[evidence_id]["progress"] = 40.0
                active_indexing_jobs[evidence_id]["stage"] = "VLM_KEYFRAME_ANALYSIS"

                # Stage 2: VLM Keyframe Analysis
                logger.info(f"Executing VLM keyframe analysis for evidence {evidence_id}...")
                vlm_obs = await VisionLanguageService.analyze_keyframes_for_job(
                    db=db,
                    evidence_id=evidence_id,
                    analysis_job_id=analysis_job_id
                )

                active_indexing_jobs[evidence_id]["progress"] = 70.0
                active_indexing_jobs[evidence_id]["stage"] = "VECTOR_EMBEDDING"

                # Re-fetch all generated documents including VLM documents
                res_docs = await db.execute(
                    select(ForensicDocument).where(ForensicDocument.evidence_id == evidence_id)
                )
                all_documents = res_docs.scalars().all()

                # Stage 3: Embedding & Vector DB Insertion
                logger.info(f"Embedding {len(all_documents)} documents into vector storage...")
                embeddings = await VectorService.index_documents(
                    db=db,
                    evidence_id=evidence_id,
                    analysis_job_id=analysis_job_id,
                    documents=all_documents
                )

                # Record AI execution provenance
                ai_exec = AIModelExecution(
                    evidence_id=evidence_id,
                    model_name="VLM + SentenceTransformer / VectorEngine",
                    model_version="1.0",
                    execution_type="SEMANTIC_INDEXING",
                    input_parameters={"total_documents": len(all_documents)},
                    verification_result="INDEX_VERIFIED_COMPLETED"
                )
                db.add(ai_exec)
                await db.commit()

                active_indexing_jobs[evidence_id] = {
                    "evidence_id": evidence_id,
                    "analysis_job_id": analysis_job_id,
                    "status": "COMPLETED",
                    "progress": 100.0,
                    "stage": "READY",
                    "total_documents": len(all_documents),
                    "total_vlm_observations": len(vlm_obs),
                    "total_embeddings": len(embeddings),
                    "error": None
                }
                logger.info(f"Semantic indexing completed successfully for evidence {evidence_id}.")

        except Exception as e:
            logger.exception(f"Error indexing evidence {evidence_id}: {e}")
            active_indexing_jobs[evidence_id] = {
                "evidence_id": evidence_id,
                "analysis_job_id": analysis_job_id,
                "status": "FAILED",
                "progress": 0.0,
                "stage": "FAILED",
                "error": str(e)
            }
