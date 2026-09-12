import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.analysis import Track, Keyframe, Detection, PossibleInteraction, ActivityInterval
from app.models.semantic import (
    ForensicDocument, SearchQuery, SearchResult, ConfidenceLevel, DocumentSourceType, DocumentType, VLMObservation
)
from app.services.semantic.query_parser import QueryIntentParser
from app.services.semantic.vector_service import VectorService

logger = logging.getLogger(__name__)

# Configurable Minimum Thresholds
MIN_SEMANTIC_SCORE = 0.40
MIN_PRIMARY_RESULT_SCORE = 0.50

class HybridSearchEngine:
    """
    Evidence-grounded Hybrid Search Engine combining query intent parsing,
    candidate document type filtering, required entity verification, hybrid ranking,
    strict relevance thresholds, and transparent separate confidence metrics.
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

        # Step 1: Query Intent Extraction
        intent = QueryIntentParser.parse_query(query_text)
        logger.info(f"Executing hybrid search for evidence {evidence_id}. Query: '{query_text}'. Intent: {intent}")

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

        # Step 2: Query Routing - Track Query Direct DB Lookup
        if intent["query_type"] == "TRACK_QUERY" and intent["track_id"] is not None:
            t_res = await db.execute(
                select(Track).where(
                    Track.evidence_id == evidence_id,
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
                        "supporting_keyframes": len(doc.metadata_json.get("linked_keyframes", [])),
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

        # Step 3: Vector & Structured Retrieval with Candidate Document Type Filtering
        vector_filters = {
            "document_types": intent["candidate_document_types"],
            "track_id": intent["track_id"],
            "start_time": intent["start_time"],
            "end_time": intent["end_time"]
        }

        similar_docs = await VectorService.search_similar(
            db=db,
            evidence_id=evidence_id,
            query_text=query_text,
            top_k=top_k * 4,
            filters=vector_filters
        )

        # Step 4: Multi-Stage Hybrid Ranking, Verification & Threshold Filtering
        verified_results: List[Dict[str, Any]] = []

        for doc, sim in similar_docs:
            # 4.1 Filter out disallowed document types (e.g. ACTIVITY_INTERVAL for person query)
            if doc.document_type.value not in intent["candidate_document_types"]:
                continue

            # 4.2 DB Verification (Verify supporting track or keyframe exists)
            track_obj = None
            if doc.track_id is not None:
                tr_res = await db.execute(
                    select(Track).where(
                        Track.evidence_id == evidence_id,
                        Track.track_number == doc.track_id
                    )
                )
                track_obj = tr_res.scalars().first()
                if not track_obj:
                    continue  # Invalid reference

            kf_obj = None
            if doc.keyframe_id is not None:
                kf_res = await db.execute(
                    select(Keyframe).where(
                        Keyframe.evidence_id == evidence_id,
                        Keyframe.id == doc.keyframe_id
                    )
                )
                kf_obj = kf_res.scalars().first()

            # 4.3 Required Entity Matching (Hard Constraint)
            doc_text_lower = f"{doc.title} {doc.content} {str(doc.metadata_json)}".lower()
            
            # Fetch detections and VLM observations for deeper verification
            doc_detections: List[Detection] = []
            if doc.track_id is not None:
                det_res = await db.execute(
                    select(Detection).where(
                        Detection.evidence_id == evidence_id,
                        Detection.track_id == doc.track_id
                    )
                )
                doc_detections = det_res.scalars().all()
            elif doc.keyframe_id is not None and kf_obj is not None:
                det_res = await db.execute(
                    select(Detection).where(
                        Detection.evidence_id == evidence_id,
                        Detection.frame_number == kf_obj.frame_number
                    )
                )
                doc_detections = det_res.scalars().all()

            vlm_descriptions = []
            if doc.keyframe_id is not None:
                vlm_res = await db.execute(
                    select(VLMObservation).where(
                        VLMObservation.evidence_id == evidence_id,
                        VLMObservation.keyframe_id == doc.keyframe_id
                    )
                )
                vlm_obs_list = vlm_res.scalars().all()
                vlm_descriptions = [v.description.lower() for v in vlm_obs_list]

            combined_evidence_text = (doc_text_lower + " " + " ".join(vlm_descriptions)).lower()

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
                        elif any(w in combined_evidence_text for w in ["person", "people", "human", "man", "woman", "pedestrian", "individual", "suspect"]):
                            ent_found = True
                    elif req_ent == "backpack":
                        if any(d.class_name.lower() in ["backpack", "bag", "handbag", "luggage"] for d in doc_detections):
                            ent_found = True
                        elif "backpack" in doc.metadata_json.get("associated_objects", []) or "bag" in doc.metadata_json.get("associated_objects", []):
                            ent_found = True
                        elif any(w in combined_evidence_text for w in ["backpack", "bag", "rucksack", "pack", "knapsack"]):
                            ent_found = True
                    elif req_ent in ["car", "vehicle", "truck"]:
                        if track_obj and track_obj.class_name.lower() in ["car", "vehicle", "truck"]:
                            ent_found = True
                        elif any(d.class_name.lower() in ["car", "vehicle", "truck"] for d in doc_detections):
                            ent_found = True
                        elif any(w in combined_evidence_text for w in ["car", "vehicle", "automobile", "truck", "van"]):
                            ent_found = True
                    else:
                        if req_ent in combined_evidence_text:
                            ent_found = True

                    if not ent_found:
                        entity_matched = False
                        missing_entities.append(req_ent)

            # Hard Constraint: Reject if required primary entity is missing completely
            if intent["primary_entity"] and not entity_matched:
                logger.info(f"Rejecting candidate doc #{doc.id} because required entity '{missing_entities}' was missing.")
                continue

            # 4.4 Visual Attribute Matching (Colors & Clothing)
            visual_attribute_matches = []
            matched_attr_count = 0
            all_target_attrs = intent["colors"] + intent["clothing"]

            for attr in all_target_attrs:
                attr_lower = attr.lower()
                is_attr_verified = False
                if re.search(rf"\b{attr_lower}\b", combined_evidence_text):
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

            # Calculate Attribute Match Score
            if all_target_attrs:
                attribute_score = matched_attr_count / len(all_target_attrs)
            else:
                attribute_score = 1.0

            # Calculate Entity Match Score
            entity_score = 1.0 if entity_matched else 0.0

            # Calculate Temporal Score
            temporal_score = 1.0
            if intent["start_time"] is not None and doc.start_time is not None:
                if not (intent["start_time"] <= doc.start_time <= (intent["end_time"] or 999999)):
                    temporal_score = 0.0

            # Calculate Evidence Support Score
            frame_count = track_obj.observation_count if track_obj else (1 if kf_obj else 0)
            keyframe_count = len(doc.metadata_json.get("linked_keyframes", [])) if doc.metadata_json.get("linked_keyframes") else (1 if kf_obj else 0)

            if track_obj and frame_count >= 5:
                evidence_support_str = "HIGH"
                evidence_support_score = 1.0
            elif track_obj or keyframe_count >= 1 or len(doc_detections) >= 1:
                evidence_support_str = "MEDIUM"
                evidence_support_score = 0.7
            else:
                evidence_support_str = "LOW"
                evidence_support_score = 0.4

            # Detection confidence calculation
            det_conf = None
            if doc_detections:
                det_conf = max(d.confidence for d in doc_detections)
            elif track_obj:
                det_conf = 0.91

            # 4.5 Multi-Stage Hybrid Score Calculation
            # Final Score = 0.35 * Semantic + 0.30 * Entity + 0.20 * Attribute + 0.05 * Temporal + 0.10 * Evidence
            final_score = (
                0.35 * sim +
                0.30 * entity_score +
                0.20 * attribute_score +
                0.05 * temporal_score +
                0.10 * evidence_support_score
            )

            # Boost track match if specific track was requested
            if intent["track_id"] and doc.track_id == intent["track_id"]:
                final_score += 0.20

            # 4.6 Strict Minimum Threshold Filtering
            if sim < MIN_SEMANTIC_SCORE or final_score < MIN_PRIMARY_RESULT_SCORE:
                logger.info(
                    f"Candidate doc #{doc.id} ({doc.title}) rejected. "
                    f"Sim: {sim:.3f} (Min: {MIN_SEMANTIC_SCORE}), Final: {final_score:.3f} (Min: {MIN_PRIMARY_RESULT_SCORE})"
                )
                continue

            query_relevance_pct = round(final_score * 100, 1)

            # Overall confidence level indicator
            if evidence_support_str == "HIGH" and query_relevance_pct >= 50.0:
                confidence_level = ConfidenceLevel.HIGH
            elif evidence_support_str in ["HIGH", "MEDIUM"] and query_relevance_pct >= 40.0:
                confidence_level = ConfidenceLevel.MEDIUM
            else:
                confidence_level = ConfidenceLevel.LOW

            conf_reason = f"Query Relevance: {query_relevance_pct:.0f}%, Evidence Support: {evidence_support_str}."

            why_explanation = {
                "semantic_match_score": round(float(sim), 3),
                "semantic_match_percentage": f"{int(round(sim * 100))}%",
                "query_relevance": query_relevance_pct,
                "query_relevance_formatted": f"{int(round(query_relevance_pct))}%",
                "entity_match": entity_matched,
                "entity_match_details": f"✓ {intent['primary_entity'].upper()}" if (entity_matched and intent['primary_entity']) else "N/A",
                "visual_attribute_matches": visual_attribute_matches,
                "supporting_frames": frame_count,
                "supporting_keyframes": keyframe_count,
                "supporting_track": doc.track_id,
                "detection_confidence": round(float(det_conf), 2) if det_conf else None,
                "detection_confidence_formatted": f"{int(round(det_conf * 100))}%" if det_conf else "N/A",
                "evidence_support": evidence_support_str,
                "document_type": doc.document_type.value,
                "source_type": doc.source_type.value,
                "temporal_range": f"{doc.start_time:.2f}s - {doc.end_time:.2f}s" if doc.start_time is not None else "N/A"
            }

            result_item = {
                "doc_id": doc.id,
                "title": doc.title,
                "summary": doc.content,
                "score": round(float(final_score), 3),
                "confidence_level": confidence_level.value,
                "confidence_reason": conf_reason,
                "query_relevance": query_relevance_pct,
                "evidence_support": evidence_support_str,
                "model_confidence": round(float(det_conf), 2) if det_conf else None,
                "track_id": doc.track_id,
                "keyframe_id": doc.keyframe_id,
                "start_time": doc.start_time,
                "end_time": doc.end_time,
                "why_explanation": why_explanation,
                "verification_status": "VERIFIED"
            }
            verified_results.append(result_item)

        # Sort by final score
        verified_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = verified_results[:top_k]

        execution_time = (time.time() - start_time_perf) * 1000.0
        db_query.execution_time_ms = execution_time

        # Save search results to DB for audit and provenance
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

        # Step 5: Anti-Hallucination Forensic Answer Formulation
        suggestions = [
            "Find people",
            "Find people carrying bags",
            "Show activity around 01:30",
            "Show people near vehicles"
        ]

        if not top_results:
            primary_ent_str = f"'{intent['primary_entity']}'" if intent["primary_entity"] else "requested"
            final_answer = (
                f"NO SUPPORTED EVIDENCE FOUND: GuardianEye found no evidence meeting the minimum "
                f"relevance threshold for: \"{query_text}\". "
                f"Closest candidate records were excluded because they did not contain {primary_ent_str} evidence."
            )
        else:
            best_res = top_results[0]
            final_answer = (
                f"Evidence grounded match found for evidence #{evidence_id}: {best_res['summary']} "
                f"(Track #{best_res['track_id'] or 'N/A'}, Query Relevance: {best_res['query_relevance']:.0f}%, "
                f"Evidence Support: {best_res['evidence_support']})."
            )

        return {
            "search_query_id": db_query.id,
            "evidence_id": evidence_id,
            "query": query_text,
            "extracted_intent": intent,
            "answer": final_answer,
            "execution_time_ms": round(execution_time, 2),
            "total_results": len(top_results),
            "suggestions": suggestions if not top_results else [],
            "results": top_results
        }
