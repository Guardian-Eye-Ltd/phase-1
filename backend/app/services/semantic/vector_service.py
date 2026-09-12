import logging
import os
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.semantic import ForensicDocument, EmbeddingRecord

logger = logging.getLogger(__name__)

class VectorService:
    """
    Vector Database & Embedding Service supporting persistent ChromaDB storage 
    with a lightweight TF-IDF / Cosine similarity fallback engine.
    """

    _chroma_client = None
    _chroma_collection = None
    _st_model = None
    _engine_type = "fallback"

    @classmethod
    def initialize(cls):
        """Initializes ChromaDB or SentenceTransformers if available, otherwise activates fallback."""
        if cls._chroma_client is not None or cls._engine_type == "ready":
            return

        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)

        try:
            import chromadb
            cls._chroma_client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
            cls._chroma_collection = cls._chroma_client.get_or_create_collection(
                name="guardianeye_forensic_records"
            )
            cls._engine_type = "chromadb"
            logger.info("VectorService initialized with ChromaDB persistent storage.")
            return
        except Exception as e:
            logger.warning(f"ChromaDB not initialized ({e}). Using built-in persistent local vector engine fallback.")
            cls._engine_type = "fallback"

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Tokenize text for fallback TF-IDF vectorization."""
        return re.findall(r"\w+", text.lower())

    @classmethod
    def _compute_tf_idf_vector(cls, text: str, vocabulary: List[str]) -> List[float]:
        """Compute TF vector over vocabulary."""
        tokens = cls._tokenize(text)
        total_tokens = max(len(tokens), 1)
        tf_dict = {}
        for t in tokens:
            tf_dict[t] = tf_dict.get(t, 0) + 1
        
        vector = []
        for word in vocabulary:
            freq = tf_dict.get(word, 0)
            vector.append(freq / total_tokens)
        return vector

    @classmethod
    def _cosine_similarity(cls, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @classmethod
    async def index_documents(
        cls, 
        db: AsyncSession, 
        evidence_id: int, 
        analysis_job_id: int, 
        documents: List[ForensicDocument]
    ) -> List[EmbeddingRecord]:
        """
        Embeds forensic document records and saves them to vector storage and database metadata.
        """
        cls.initialize()
        embedding_records: List[EmbeddingRecord] = []

        if not documents:
            return embedding_records

        # Chromadb embedding path
        if cls._engine_type == "chromadb" and cls._chroma_collection is not None:
            try:
                ids = [f"doc_{doc.id}" for doc in documents]
                texts = [f"{doc.title}. {doc.content}" for doc in documents]
                metadatas = []

                for doc in documents:
                    meta = {
                        "evidence_id": int(doc.evidence_id),
                        "analysis_job_id": int(doc.analysis_job_id),
                        "track_id": int(doc.track_id) if doc.track_id is not None else -1,
                        "keyframe_id": int(doc.keyframe_id) if doc.keyframe_id is not None else -1,
                        "document_type": str(doc.document_type.value),
                        "source_type": str(doc.source_type.value),
                        "start_time": float(doc.start_time) if doc.start_time is not None else 0.0,
                        "end_time": float(doc.end_time) if doc.end_time is not None else 0.0
                    }
                    metadatas.append(meta)

                cls._chroma_collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )

                for doc, vector_id in zip(documents, ids):
                    rec = EmbeddingRecord(
                        evidence_id=evidence_id,
                        analysis_job_id=analysis_job_id,
                        forensic_document_id=doc.id,
                        vector_id=vector_id,
                        embedding_model=settings.EMBEDDING_MODEL_NAME,
                        document_type=doc.document_type.value,
                        source_type=doc.source_type.value
                    )
                    db.add(rec)
                    embedding_records.append(rec)

                await db.commit()
                logger.info(f"Indexed {len(documents)} documents in ChromaDB for evidence {evidence_id}.")
                return embedding_records
            except Exception as e:
                logger.warning(f"ChromaDB indexing error: {e}. Falling back to DB-grounded keyword vector indexing.")

        # Fallback keyword vector indexing
        for doc in documents:
            vector_id = f"doc_fb_{doc.id}"
            rec = EmbeddingRecord(
                evidence_id=evidence_id,
                analysis_job_id=analysis_job_id,
                forensic_document_id=doc.id,
                vector_id=vector_id,
                embedding_model="Fallback-TFIDF-Keyword-Engine",
                document_type=doc.document_type.value,
                source_type=doc.source_type.value
            )
            db.add(rec)
            embedding_records.append(rec)

        await db.commit()
        logger.info(f"Indexed {len(documents)} documents using Fallback Vector Engine for evidence {evidence_id}.")
        return embedding_records

    @classmethod
    async def search_similar(
        cls, 
        db: AsyncSession, 
        evidence_id: Optional[int], 
        query_text: str, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[ForensicDocument, float]]:
        """
        Executes semantic vector search over forensic document records.
        Returns list of (ForensicDocument, similarity_score).
        """
        cls.initialize()
        filters = filters or {}

        # 1. Fetch eligible ForensicDocuments from SQL database with metadata constraints
        stmt = select(ForensicDocument)
        if evidence_id is not None:
            stmt = stmt.where(ForensicDocument.evidence_id == evidence_id)
        if "document_types" in filters and filters["document_types"]:
            stmt = stmt.where(ForensicDocument.document_type.in_(filters["document_types"]))
        if "track_id" in filters and filters["track_id"] is not None:
            stmt = stmt.where(ForensicDocument.track_id == filters["track_id"])
        if "start_time" in filters and filters["start_time"] is not None:
            stmt = stmt.where(ForensicDocument.end_time >= filters["start_time"])
        if "end_time" in filters and filters["end_time"] is not None:
            stmt = stmt.where(ForensicDocument.start_time <= filters["end_time"])

        res = await db.execute(stmt)
        documents = res.scalars().all()

        if not documents:
            return []

        doc_dict = {doc.id: doc for doc in documents}

        # 2. ChromaDB search if available
        if cls._engine_type == "chromadb" and cls._chroma_collection is not None:
            try:
                where_clause = {}
                if evidence_id is not None:
                    where_clause["evidence_id"] = int(evidence_id)
                if "track_id" in filters and filters["track_id"] is not None:
                    where_clause["track_id"] = int(filters["track_id"])

                query_params = {"query_texts": [query_text], "n_results": min(top_k * 2, len(documents))}
                if where_clause:
                    query_params["where"] = where_clause

                chroma_res = cls._chroma_collection.query(**query_params)

                results = []
                if chroma_res and "ids" in chroma_res and chroma_res["ids"]:
                    retrieved_ids = chroma_res["ids"][0]
                    distances = chroma_res["distances"][0] if "distances" in chroma_res and chroma_res["distances"] else [0.5] * len(retrieved_ids)

                    for v_id, dist in zip(retrieved_ids, distances):
                        # Convert distance to similarity score
                        sim = max(0.0, 1.0 - (dist / 2.0))
                        doc_id_str = v_id.replace("doc_", "").replace("fb_", "")
                        if doc_id_str.isdigit():
                            d_id = int(doc_id_str)
                            if d_id in doc_dict:
                                results.append((doc_dict[d_id], sim))
                    
                    if results:
                        results.sort(key=lambda x: x[1], reverse=True)
                        return results[:top_k]
            except Exception as e:
                logger.warning(f"ChromaDB search query error: {e}. Falling back to term similarity search.")

        # 3. Fallback TF-IDF + Term Cosine Similarity calculation
        STOP_WORDS = {
            "find", "show", "me", "a", "an", "the", "in", "on", "at", "with", "carrying",
            "wearing", "near", "is", "was", "are", "were", "of", "to", "for", "and", "or",
            "some", "any", "all", "around", "about", "between", "what", "happened"
        }
        all_query_tokens = cls._tokenize(query_text)
        content_query_tokens = [tok for tok in all_query_tokens if tok not in STOP_WORDS]
        if not content_query_tokens:
            content_query_tokens = all_query_tokens

        all_texts = [f"{doc.title} {doc.content}" for doc in documents]
        vocab = list(set(w for text in all_texts for w in cls._tokenize(text)).union(set(all_query_tokens)))

        q_vec = cls._compute_tf_idf_vector(query_text, vocab)

        results = []
        for doc in documents:
            doc_text = f"{doc.title} {doc.content}"
            d_vec = cls._compute_tf_idf_vector(doc_text, vocab)
            sim = cls._cosine_similarity(q_vec, d_vec)
            
            # Boost similarity if query content keywords appear directly in title or content
            text_lower = doc_text.lower()
            token_matches = sum(1 for tok in content_query_tokens if tok in text_lower)
            if content_query_tokens:
                keyword_boost = 0.45 * (token_matches / len(content_query_tokens))
                sim = min(1.0, sim + keyword_boost)

            if sim > 0.05:
                results.append((doc, float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
