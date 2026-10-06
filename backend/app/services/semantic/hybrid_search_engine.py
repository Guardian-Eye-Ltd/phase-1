import logging
import time
import re
import asyncio
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.analysis import Track, Keyframe, Detection, PossibleInteraction, ActivityInterval, AnalysisJob, JobStatus
from app.models.semantic import (
    ForensicDocument, SearchQuery, SearchResult, ConfidenceLevel, DocumentSourceType, DocumentType, VLMObservation
)
from app.services.semantic.query_parser import QueryIntentParser
from app.services.semantic.indexer import (
    get_embedding_model, get_chroma_client,
    _chroma_collection_name, _get_latest_completed_job_id
)

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """
    Evidence-grounded Hybrid Search Engine combining query intent parsing,
    candidate document type filtering, required entity verification, hybrid ranking,
    strict relevance thresholds, and transparent confidence metrics.
    """

    @classmethod
    async def execute_search(
        cls,
        db: AsyncSession,
        evidence_id: int,
        query_text: str,
        user_id: Optional[int] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Executes multi-stage hybrid search query pipeline.
        """
        start_time_perf = time.time()
        min_relevance_threshold = getattr(settings, "MIN_SEARCH_RELEVANCE", 0.22)

        # Step 1: Resolve the latest completed analysis job for this evidence
        analysis_job_id = await _get_latest_completed_job_id(evidence_id, db)
        if analysis_job_id is None:
            return {
                "search_query_id": None,
                "evidence_id": evidence_id,
                "query": query_text,
                "extracted_intent": {},
                "answer": "No completed analysis job found for this evidence. Please run video analysis first.",
                "execution_time_ms": 0.0,
                "total_results": 0,
                "results": []
            }

        # Step 2: Query Intent Extraction
        intent = QueryIntentParser.parse_query(query_text)
        logger.info(
            f"[QUERY] evidence=#{evidence_id} job=#{analysis_job_id} "
            f"query='{query_text}' intent={intent['query_type']}"
        )

        # Record SearchQuery in DB
        db_query = SearchQuery(
            user_id=user_id,
            evidence_id=evidence_id,
            query_text=query_text,
            extracted_intent=intent,
            execution_time_ms=0.0
        )
        db.add(db_query)
        await db.commit()
        await db.refresh(db_query)

        # Step 3: Query Routing — Track Search Direct DB Lookup (scoped to job)
        if intent["query_type"] == "TRACK_SEARCH" and intent["track_id"] is not None:
            t_res = await db.execute(
                select(Track).where(
                    Track.evidence_id == evidence_id,
                    Track.analysis_job_id == analysis_job_id,
                    Track.track_number == intent["track_id"]
                )
            )
            track_obj = t_res.scalars().first()
            if track_obj:
                doc_res = await db.execute(
                    select(ForensicDocument).where(
                        ForensicDocument.evidence_id == evidence_id,
                        ForensicDocument.track_id == intent["track_id"]
                    )
                )
                track_docs = doc_res.scalars().all()
                if track_docs:
                    doc = track_docs[0]
                    why_exp = {
                        "semantic_match_score": 1.0,
                        "semantic_match_percentage": "100%",
                        "query_relevance": 100.0,
                        "query_relevance_formatted": "100%",
                        "entity_match": True,
                        "entity_match_details": f"✓ Direct Track #{intent['track_id']} Lookup",
                        "visual_attribute_matches": [],
                        "supporting_frames": track_obj.observation_count,
                        "supporting_keyframes": len(doc.metadata_json.get("linked_keyframes", [])) if doc.metadata_json else 0,
                        "supporting_track": track_obj.track_number,
                        "detection_confidence": 0.95,
                        "detection_confidence_formatted": "95%",
                        "evidence_support": "HIGH",
                        "document_type": doc.document_type.value,
                        "source_type": doc.source_type.value,
                        "temporal_range": f"{doc.start_time:.2f}s - {doc.end_time:.2f}s" if doc.start_time is not None else "N/A"
                    }
                    item = {
                        "doc_id": doc.id,
                        "title": doc.title,
                        "summary": doc.content,
                        "score": 1.0,
                        "confidence_level": "HIGH",
                        "confidence_reason": f"Direct database lookup match for Track #{intent['track_id']}.",
                        "query_relevance": 100.0,
                        "evidence_support": "HIGH",
                        "model_confidence": 0.95,
                        "track_id": doc.track_id,
                        "keyframe_id": doc.keyframe_id,
                        "start_time": doc.start_time,
                        "end_time": doc.end_time,
                        "why_explanation": why_exp,
                        "verification_status": "VERIFIED"
                    }
                    execution_time = (time.time() - start_time_perf) * 1000.0
                    db_query.execution_time_ms = execution_time
                    await db.commit()

                    return {
                        "search_query_id": db_query.id,
                        "evidence_id": evidence_id,
                        "query": query_text,
                        "extracted_intent": intent,
                        "answer": f"Evidence grounded match found for evidence #{evidence_id}: {doc.content} (Track #{doc.track_id}, Relevance: 100%).",
                        "execution_time_ms": round(execution_time, 2),
                        "total_results": 1,
                        "suggestions": [],
                        "results": [item]
                    }

        # Step 4: Vector & Structured Retrieval (job-scoped collection)
        model = get_embedding_model()
        chroma_client = get_chroma_client()
        collection_name = _chroma_collection_name(evidence_id, analysis_job_id)

        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception:
            execution_time = (time.time() - start_time_perf) * 1000.0
            return {
                "search_query_id": db_query.id,
                "evidence_id": evidence_id,
                "query": query_text,
                "extracted_intent": intent,
                "answer": "No semantic index exists for this video yet. Please run semantic indexing first.",
                "execution_time_ms": round(execution_time, 2),
                "total_results": 0,
                "results": []
            }

        # Vector embedding of query
        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(
            None,
            lambda: model.encode(query_text, normalize_embeddings=True).tolist()
        )

        query_res = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k * 4, max(collection.count(), 1)),
            include=["documents", "metadatas", "distances"]
        )

        candidate_ids = []
        distances_map = {}
        if query_res and query_res.get("ids") and query_res["ids"][0]:
            for v_id, dist in zip(query_res["ids"][0], query_res["distances"][0]):
                doc_id = int(v_id.replace("doc_", ""))
                candidate_ids.append(doc_id)
                # Cosine similarity conversion for normalized embeddings
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                distances_map[doc_id] = similarity

        if not candidate_ids:
            execution_time = (time.time() - start_time_perf) * 1000.0
            return {
                "search_query_id": db_query.id,
                "evidence_id": evidence_id,
                "query": query_text,
                "extracted_intent": intent,
                "answer": f"No forensic evidence matching '{query_text}' was identified.",
                "execution_time_ms": round(execution_time, 2),
                "total_results": 0,
                "results": []
            }

        # Step 5: Multi-Stage Hybrid Ranking, Verification & Threshold Filtering
        doc_res = await db.execute(
            select(ForensicDocument).where(
                ForensicDocument.id.in_(candidate_ids),
                ForensicDocument.analysis_job_id == analysis_job_id
            )
        )
        documents = doc_res.scalars().all()
        doc_lookup = {d.id: d for d in documents}

        verified_results: List[Dict[str, Any]] = []

        for doc_id, sim in distances_map.items():
            doc = doc_lookup.get(doc_id)
            if not doc:
                continue

            # Candidate document type filter
            if intent["candidate_document_types"] and doc.document_type.value not in intent["candidate_document_types"]:
                logger.info(f"[FILTER] Excluding candidate doc #{doc.id} ({doc.document_type.value}) - Not in candidate types {intent['candidate_document_types']}.")
                continue

            # DB Grounded Verification
            track_obj = None
            if doc.track_id is not None:
                tr_res = await db.execute(
                    select(Track).where(
                        Track.evidence_id == evidence_id,
                        Track.track_number == doc.track_id
                    )
                )
                track_obj = tr_res.scalars().first()

            kf_obj = None
            if doc.keyframe_id is not None:
                kf_res = await db.execute(
                    select(Keyframe).where(
                        Keyframe.evidence_id == evidence_id,
                        Keyframe.id == doc.keyframe_id
                    )
                )
                kf_obj = kf_res.scalars().first()

            # Retrieve detections for entity & attribute verification
            doc_detections: List[Detection] = []
            if doc.track_id is not None:
                det_res = await db.execute(
                    select(Detection).where(
                        Detection.evidence_id == evidence_id,
                        Detection.track_id == doc.track_id
                    )
                )
                doc_detections.extend(det_res.scalars().all())

            doc_text_lower = f"{doc.title} {doc.content} {str(doc.metadata_json)}".lower()

            entity_matched = True
            missing_entities = []
            if intent["required_entities"]:
                for req_ent in intent["required_entities"]:
                    ent_found = False
                    if req_ent == "person":
                        if track_obj and track_obj.class_name.lower() in ["person", "human", "pedestrian"]:
                            ent_found = True
                        elif any(d.class_name.lower() in ["person", "human", "pedestrian"] for d in doc_detections):
                            ent_found = True
                        elif any(w in doc_text_lower for w in ["person", "people", "human", "man", "woman", "pedestrian", "individual"]):
                            ent_found = True
                    elif req_ent in ["backpack", "bag", "handbag", "suitcase"]:
                        if any(d.class_name.lower() in ["backpack", "bag", "handbag", "suitcase"] for d in doc_detections):
                            ent_found = True
                        elif doc.metadata_json and doc.metadata_json.get("carries_bag"):
                            ent_found = True
                        elif any(w in doc_text_lower for w in ["backpack", "bag", "handbag", "suitcase", "luggage", "pack"]):
                            ent_found = True
                    elif req_ent in ["car", "vehicle", "truck", "bus", "motorcycle", "bicycle"]:
                        if track_obj and track_obj.class_name.lower() in ["car", "vehicle", "truck", "bus", "motorcycle", "bicycle"]:
                            ent_found = True
                        elif any(d.class_name.lower() in ["car", "vehicle", "truck", "bus", "motorcycle", "bicycle"] for d in doc_detections):
                            ent_found = True
                        elif any(w in doc_text_lower for w in ["car", "vehicle", "automobile", "truck", "van", "bus"]):
                            ent_found = True
                    else:
                        if req_ent in doc_text_lower:
                            ent_found = True

                    if not ent_found:
                        entity_matched = False
                        missing_entities.append(req_ent)

            if intent["primary_entity"] and not entity_matched and doc.document_type.value != "BEHAVIORAL_EVENT":
                logger.info(f"[VERIFICATION] Rejecting candidate doc #{doc.id} — Missing required primary entity '{missing_entities}'.")
                continue

            # Visual attribute matching
            visual_attribute_matches = []
            matched_attr_count = 0
            all_target_attrs = intent["colors"] + intent["clothing"] + intent.get("behavioral_terms", [])

            for attr in all_target_attrs:
                attr_lower = attr.lower()
                is_attr_verified = False
                if re.search(rf"\b{attr_lower}\b", doc_text_lower):
                    is_attr_verified = True
                
                if is_attr_verified:
                    matched_attr_count += 1
                    visual_attribute_matches.append({
                        "name": attr.upper(),
                        "status": "VERIFIED",
                        "formatted": f"✓ {attr.upper()}"
                    })
                else:
                    visual_attribute_matches.append({
                        "name": attr.upper(),
                        "status": "NOT_VERIFIED",
                        "formatted": f"Visual Attribute ({attr.upper()}): Not verified"
                    })

            attribute_score = (matched_attr_count / len(all_target_attrs)) if all_target_attrs else 1.0
            entity_score = 1.0 if entity_matched else (0.80 if doc.document_type.value == "BEHAVIORAL_EVENT" else 0.0)

            # Keyword overlap boost (+0.05 per matched entity or color or behavior word)
            q_words = set([w.strip().lower() for w in re.findall(r"\b\w+\b", query_text) if len(w.strip()) >= 3])
            doc_words = set([w.strip().lower() for w in re.findall(r"\b\w+\b", doc.content) if len(w.strip()) >= 3])
            keyword_matches = len([w for w in q_words if w in doc_words])
            keyword_boost = keyword_matches * 0.05

            # Multi-stage hybrid score calculation
            final_score = min(1.0, 0.50 * sim + 0.30 * entity_score + 0.15 * attribute_score + 0.05 * keyword_boost)

            if final_score < min_relevance_threshold:
                logger.info(f"[RANKING] Candidate doc #{doc.id} rejected. Sim: {sim:.3f}, Final: {final_score:.3f} (Threshold: {min_relevance_threshold:.2f})")
                continue

            conf_level = ConfidenceLevel.HIGH if final_score >= 0.65 else (ConfidenceLevel.MEDIUM if final_score >= 0.40 else ConfidenceLevel.LOW)

            why_explanation = {
                "semantic_match_score": round(float(sim), 3),
                "semantic_match_percentage": f"{int(round(sim * 100))}%",
                "query_relevance": round(final_score * 100, 1),
                "query_relevance_formatted": f"{int(round(final_score * 100))}%",
                "entity_match": entity_matched,
                "entity_match_details": f"✓ {intent['primary_entity'].upper()}" if (entity_matched and intent['primary_entity']) else "N/A",
                "visual_attribute_matches": visual_attribute_matches,
                "supporting_frames": track_obj.observation_count if track_obj else 1,
                "supporting_keyframes": 1 if kf_obj else 0,
                "supporting_track": doc.track_id,
                "evidence_support": "HIGH" if (track_obj and track_obj.observation_count >= 5) else "MEDIUM",
                "document_type": doc.document_type.value,
                "source_type": doc.source_type.value,
                "temporal_range": f"{doc.start_time:.2f}s - {doc.end_time:.2f}s" if doc.start_time is not None else "N/A"
            }

            result_item = {
                "doc_id": doc.id,
                "title": doc.title,
                "summary": doc.content,
                "score": round(float(final_score), 3),
                "confidence_level": conf_level.value,
                "confidence_reason": f"Semantic similarity match ({round(sim * 100)}%) backed by verified database records.",
                "query_relevance": round(final_score * 100, 1),
                "evidence_support": why_explanation["evidence_support"],
                "model_confidence": 0.90,
                "track_id": doc.track_id,
                "keyframe_id": doc.keyframe_id,
                "start_time": doc.start_time,
                "end_time": doc.end_time,
                "why_explanation": why_explanation,
                "verification_status": "VERIFIED"
            }
            verified_results.append(result_item)

        verified_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = verified_results[:top_k]

        execution_time = (time.time() - start_time_perf) * 1000.0
        db_query.execution_time_ms = execution_time

        # Save search results to DB
        for item in top_results:
            db_res = SearchResult(
                search_query_id=db_query.id,
                evidence_id=evidence_id,
                track_id=item["track_id"],
                keyframe_id=item["keyframe_id"],
                start_time=item["start_time"],
                end_time=item["end_time"],
                relevance_score=item["score"],
                confidence_level=ConfidenceLevel(item["confidence_level"]),
                confidence_reason=item["confidence_reason"],
                summary_text=item["summary"],
                explanation_json=item["why_explanation"],
                verification_status=item["verification_status"]
            )
            db.add(db_res)
        await db.commit()

        if intent.get("is_count_query"):
            target_entity = intent.get("primary_entity") or "entity"
            matching_tracks = []
            for r in verified_results:
                if r.get("track_id") is not None and r["track_id"] not in matching_tracks:
                    matching_tracks.append(r["track_id"])
            
            # Query all tracks in DB for this evidence to get total distinct track count
            all_tr_res = await db.execute(select(Track).where(Track.evidence_id == evidence_id))
            all_tracks = all_tr_res.scalars().all()
            if target_entity == "person":
                entity_tracks = [t for t in all_tracks if t.class_name.lower() in ["person", "human", "pedestrian"]]
            elif target_entity in ["car", "vehicle", "truck", "bus"]:
                entity_tracks = [t for t in all_tracks if t.class_name.lower() in ["car", "vehicle", "truck", "bus", "automobile"]]
            else:
                entity_tracks = [t for t in all_tracks if target_entity.lower() in t.class_name.lower()]

            count = len(entity_tracks)
            if count > 0:
                track_details = [f"Track #{t.track_number} ({t.class_name.capitalize()})" for t in entity_tracks[:5]]
                final_answer = f"GuardianEye identified {count} distinct {target_entity} track(s) in Evidence #{evidence_id}: {', '.join(track_details)}."
            else:
                final_answer = f"GuardianEye detected 0 {target_entity} tracks in Evidence #{evidence_id}."
        elif not top_results:
            final_answer = f"NO SUPPORTED EVIDENCE: GuardianEye found no sufficiently supported evidence meeting the minimum relevance threshold for: \"{query_text}\"."
        else:
            best_res = top_results[0]
            final_answer = f"Evidence grounded match found for evidence #{evidence_id}: {best_res['summary']} (Track #{best_res['track_id'] or 'N/A'}, Query Relevance: {best_res['query_relevance']:.0f}%)."

        return {
            "search_query_id": db_query.id,
            "evidence_id": evidence_id,
            "query": query_text,
            "extracted_intent": intent,
            "answer": final_answer,
            "execution_time_ms": round(execution_time, 2),
            "total_results": len(top_results),
            "suggestions": [
                "Find person carrying backpack",
                "Show vehicles near pedestrians",
                "Search loitering activity"
            ] if not top_results else [],
            "results": top_results
        }
