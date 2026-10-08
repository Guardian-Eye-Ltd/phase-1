import logging
import uuid
import time
import json
import math
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.evidence import Evidence
from app.models.analysis import Track, Keyframe, Detection, ActivityInterval
from app.models.observation import TrackAttributeAggregate
from app.services.agents.llm_provider import LLMProvider
from app.services.agents.investigation_tools import InvestigationToolSystem
from app.services.agents.evidence_verifier import EvidenceVerifier
from app.services.agents.report_generator import ForensicReportGenerator
from app.services.agents.analysis_job_resolver import resolve_analysis_job
from app.services.pipeline.event_engine import EventEngine
from app.services.semantic.query_parser import (
    QueryIntentParser,
    INTENT_COUNT, INTENT_FRAME_COUNT, INTENT_SEARCH, INTENT_ATTRIBUTE_SEARCH,
    INTENT_TEMPORAL_SEARCH, INTENT_RELATION_SEARCH, INTENT_EVENT_SEARCH, INTENT_SUMMARY,
)

logger = logging.getLogger(__name__)


def _dedup_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove exact-duplicate events using (event_type, primary_track, rounded_t,
    second_track) as the identity. Avoids the "Sudden Acceleration x3" bug
    without discarding genuinely distinct events.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for ev in events:
        tracks = ev.get("involved_track_ids") or []
        primary = tracks[0] if tracks else None
        secondary = tracks[1] if len(tracks) > 1 else None
        key = (
            ev.get("event_type"),
            primary,
            secondary,
            round(float(ev.get("start_time", 0.0)) * 2) / 2,  # 0.5s tolerance
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


class InvestigationOrchestrator:
    """
    Intent-first investigation orchestrator.

    Flow:
        parse_query -> resolve_analysis_job -> route by intent
          COUNT / FRAME_COUNT      -> deterministic DB aggregation
          EVENT_SEARCH             -> job-scoped EventEngine
          SUMMARY                  -> compact deterministic summary
          SEARCH / ATTRIBUTE /
            TEMPORAL / RELATION    -> HybridSearchEngine + verification
    """

    @classmethod
    async def run_investigation(
        cls,
        db: AsyncSession,
        evidence_id: int,
        query_text: str,
        user_id: Optional[int] = None,
        analysis_job_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        start_time_perf = time.time()
        investigation_id = f"INV-{uuid.uuid4().hex[:8].upper()}"

        # ------------------------------------------------------------------
        # 0. Load evidence record (filename, hash)
        # ------------------------------------------------------------------
        ev_res = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
        evidence = ev_res.scalars().first()
        video_filename = evidence.original_filename if evidence else f"Evidence_{evidence_id}.mp4"
        ev_hash = evidence.sha256_hash if evidence and hasattr(evidence, "sha256_hash") else None

        # ------------------------------------------------------------------
        # 1. Parse query intent (deterministic, no LLM)
        # ------------------------------------------------------------------
        parsed = QueryIntentParser.parse_query(query_text)
        structured = parsed.get("structured_intent", {})
        intent_type = structured.get("intent", INTENT_SEARCH)

        # ------------------------------------------------------------------
        # 2. Resolve analysis job
        # ------------------------------------------------------------------
        job_id = await resolve_analysis_job(db, evidence_id, analysis_job_id)

        logger.info(
            "[INVESTIGATION] id=%s evidence=%s job=%s query=%r intent=%s entity=%s "
            "aggregation=%s timestamp=%s",
            investigation_id, evidence_id, job_id, query_text, intent_type,
            structured.get("entity"), structured.get("aggregation"),
            structured.get("timestamp"),
        )

        progress_history: List[Dict[str, Any]] = [
            {"stage": "INTENT_PARSING", "status": "COMPLETED",
             "message": f"Routed as {intent_type} intent."},
            {"stage": "JOB_RESOLUTION", "status": "COMPLETED",
             "message": f"Operating on analysis_job_id={job_id}."},
        ]

        if job_id is None:
            return cls._no_job_response(
                investigation_id, evidence_id, query_text, parsed, start_time_perf,
                progress_history,
            )

        # ------------------------------------------------------------------
        # 3. Route by intent
        # ------------------------------------------------------------------
        if intent_type == INTENT_COUNT:
            return await cls._handle_count(
                db, investigation_id, evidence, evidence_id, job_id,
                query_text, parsed, structured, video_filename, ev_hash,
                progress_history, start_time_perf,
            )

        if intent_type == INTENT_FRAME_COUNT:
            return await cls._handle_frame_count(
                db, investigation_id, evidence, evidence_id, job_id,
                query_text, parsed, structured, video_filename, ev_hash,
                progress_history, start_time_perf,
            )

        if intent_type == INTENT_ATTRIBUTE_SEARCH:
            return await cls._handle_attribute_search(
                db, investigation_id, evidence_id, job_id,
                query_text, parsed, structured, video_filename, ev_hash,
                progress_history, start_time_perf,
            )

        if intent_type == INTENT_EVENT_SEARCH:
            return await cls._handle_event_search(
                db, investigation_id, evidence_id, job_id,
                query_text, parsed, structured, video_filename, ev_hash,
                progress_history, start_time_perf,
            )

        # SEARCH / ATTRIBUTE_SEARCH / TEMPORAL_SEARCH / RELATION_SEARCH / SUMMARY
        return await cls._handle_semantic(
            db, investigation_id, evidence_id, job_id,
            query_text, parsed, structured, video_filename, ev_hash,
            progress_history, start_time_perf,
        )

    # ======================================================================
    # Intent handlers
    # ======================================================================

    @classmethod
    async def _handle_count(
        cls, db, investigation_id, evidence, evidence_id, job_id,
        query_text, parsed, structured, video_filename, ev_hash,
        progress_history, start_time_perf,
    ) -> Dict[str, Any]:
        entity = structured.get("entity") or "entity"
        progress_history.append({
            "stage": "COUNTING", "status": "IN_PROGRESS",
            "message": f"Running deterministic distinct-track count for '{entity}'.",
        })

        result = await InvestigationToolSystem.count_entities(
            db=db, evidence_id=evidence_id, analysis_job_id=job_id,
            entity=entity,
            start_time=structured.get("start_time"),
            end_time=structured.get("end_time"),
        )
        verification = await EvidenceVerifier.verify_count_result(
            db=db, evidence_id=evidence_id, count_result=result,
        )
        progress_history.append({
            "stage": "COUNTING", "status": "COMPLETED",
            "message": f"Count: {result['total']} unique {entity}(s).",
        })

        report_md = ForensicReportGenerator.generate_count_report(
            investigation_id=investigation_id,
            video_filename=video_filename,
            query_text=query_text,
            count_result=result,
            vlm_frame_observations=None,
            evidence_hash=ev_hash,
        )

        execution_time_ms = round((time.time() - start_time_perf) * 1000.0, 2)
        return {
            "investigation_id": investigation_id,
            "evidence_id": evidence_id,
            "analysis_job_id": job_id,
            "query": query_text,
            "status": "COMPLETED",
            "execution_time_ms": execution_time_ms,
            "intent": {
                "type": INTENT_COUNT,
                "entity": entity,
                "scope": result.get("scope"),
                "aggregation": result.get("aggregation"),
            },
            "result": {
                "total": result["total"],
                "breakdown": result["breakdown"],
                "track_ids": result["track_ids"],
            },
            "evidence_basis": {"primary": "TRACKS", "secondary": ["VLM_FRAME_OBSERVATIONS"]},
            "verification": verification,
            "plan": {"goal": f"Count distinct {entity} tracks", "tasks": ["count_entities"]},
            "progress": progress_history,
            "events": [],
            "findings": [],
            "report_markdown": report_md,
            "diagnostic": result.get("diagnostic", {}),
        }

    @classmethod
    async def _handle_frame_count(
        cls, db, investigation_id, evidence, evidence_id, job_id,
        query_text, parsed, structured, video_filename, ev_hash,
        progress_history, start_time_perf,
    ) -> Dict[str, Any]:
        entity = structured.get("entity") or "entity"
        timestamp = float(structured.get("timestamp") or 0.0)
        progress_history.append({
            "stage": "FRAME_COUNTING", "status": "IN_PROGRESS",
            "message": f"Counting '{entity}' tracks active at t={timestamp:.2f}s.",
        })

        result = await InvestigationToolSystem.count_entities_at_time(
            db=db, evidence_id=evidence_id, analysis_job_id=job_id,
            entity=entity, timestamp=timestamp,
        )
        verification = await EvidenceVerifier.verify_count_result(
            db=db, evidence_id=evidence_id, count_result=result,
        )

        progress_history.append({
            "stage": "FRAME_COUNTING", "status": "COMPLETED",
            "message": f"Active {entity}(s) at {timestamp:.2f}s: {result['count']}.",
        })

        report_md = ForensicReportGenerator.generate_frame_count_report(
            investigation_id=investigation_id,
            video_filename=video_filename,
            query_text=query_text,
            frame_count_result=result,
            evidence_hash=ev_hash,
        )

        return {
            "investigation_id": investigation_id,
            "evidence_id": evidence_id,
            "analysis_job_id": job_id,
            "query": query_text,
            "status": "COMPLETED",
            "execution_time_ms": round((time.time() - start_time_perf) * 1000.0, 2),
            "intent": {
                "type": INTENT_FRAME_COUNT,
                "entity": entity,
                "scope": "TIMESTAMP",
                "aggregation": "DISTINCT_ACTIVE_TRACK_COUNT",
                "timestamp": timestamp,
            },
            "result": {
                "count": result["count"],
                "breakdown": result["breakdown"],
                "track_ids": result["track_ids"],
            },
            "evidence_basis": {"primary": "TRACKS", "secondary": []},
            "verification": verification,
            "plan": {"goal": f"Count active {entity} tracks at t={timestamp}s"},
            "progress": progress_history,
            "events": [],
            "findings": [],
            "report_markdown": report_md,
            "diagnostic": result.get("diagnostic", {}),
        }

    @classmethod
    async def _handle_attribute_search(
        cls, db, investigation_id, evidence_id, job_id,
        query_text, parsed, structured, video_filename, ev_hash,
        progress_history, start_time_perf,
    ) -> Dict[str, Any]:
        """
        Structured attribute filtering over aggregated observations, with a
        semantic-retrieval fallback when no attribute data exists for the job.
        """
        entity = structured.get("entity")
        attributes = dict(structured.get("attributes") or {})
        plate_query = attributes.get("license_plate_text")

        progress_history.append({
            "stage": "ATTRIBUTE_FILTER", "status": "IN_PROGRESS",
            "message": f"Structured filtering on {attributes} over aggregated observations.",
        })

        if plate_query:
            matches = await InvestigationToolSystem.find_vehicle_by_plate(
                db=db, evidence_id=evidence_id, analysis_job_id=job_id,
                plate_text=plate_query, allow_fuzzy=False,
            )
            # Normalise plate hits into the shared match shape.
            matches = [
                {
                    "track_id": m["track_id"],
                    "class_name": "vehicle",
                    "start_time": None, "end_time": None, "duration": None,
                    "match_confidence": m["confidence"],
                    "evidence_basis": m["evidence_basis"],
                    "attributes": {
                        "license_plate_text": {
                            "value": m["plate_text"],
                            "confidence": m["confidence"],
                            "observation_count": m["observation_count"],
                            "supporting_count": m["supporting_count"],
                            "dissenting_count": m["observation_count"] - m["supporting_count"],
                            "status": m["status"],
                            "confidence_breakdown": {},
                            "supporting_frames": [],
                        }
                    },
                }
                for m in matches
            ]
        else:
            matches = await InvestigationToolSystem.find_tracks_by_attributes(
                db=db, evidence_id=evidence_id, analysis_job_id=job_id,
                attributes=attributes, entity=entity,
            )

        progress_history.append({
            "stage": "ATTRIBUTE_FILTER", "status": "COMPLETED",
            "message": f"{len(matches)} track(s) matched all requested attributes.",
        })

        # If this job predates attribute persistence there is nothing to filter.
        # Say so explicitly rather than silently returning a negative finding.
        attr_rows = (await db.execute(
            select(func.count()).select_from(TrackAttributeAggregate).where(
                TrackAttributeAggregate.analysis_job_id == job_id
            )
        )).scalar() or 0

        degraded_note = None
        if attr_rows == 0:
            degraded_note = (
                f"Analysis job #{job_id} contains no aggregated attribute observations. "
                "It likely predates attribute persistence — re-run analysis to populate."
            )
            progress_history.append({
                "stage": "ATTRIBUTE_FILTER", "status": "DEGRADED",
                "message": degraded_note,
            })

        report_md = ForensicReportGenerator.generate_attribute_report(
            investigation_id=investigation_id,
            video_filename=video_filename,
            query_text=query_text,
            entity=entity,
            requested_attributes=attributes,
            matches=matches,
            analysis_job_id=job_id,
            evidence_hash=ev_hash,
        )
        if degraded_note:
            report_md += f"\n> **Note**: {degraded_note}\n"

        return {
            "investigation_id": investigation_id,
            "evidence_id": evidence_id,
            "analysis_job_id": job_id,
            "query": query_text,
            "status": "COMPLETED" if matches else "NO_SUPPORTED_EVIDENCE",
            "execution_time_ms": round((time.time() - start_time_perf) * 1000.0, 2),
            "intent": {
                "type": INTENT_ATTRIBUTE_SEARCH,
                "entity": entity,
                "attributes": attributes,
            },
            "result": {"matches": matches, "total_matches": len(matches)},
            "evidence_basis": {
                "primary": "TRACK_ATTRIBUTE_AGGREGATE",
                "secondary": ["VISUAL_ATTRIBUTE_OBSERVATION", "TRACKS"],
            },
            "plan": {"goal": f"Find {entity} matching {attributes}",
                     "tasks": ["find_tracks_by_attributes"]},
            "progress": progress_history,
            "events": [],
            "findings": matches,
            "report_markdown": report_md,
            "diagnostic": {
                "aggregate_rows_in_job": attr_rows,
                "requested_attributes": attributes,
            },
        }

    @classmethod
    async def _handle_event_search(
        cls, db, investigation_id, evidence_id, job_id,
        query_text, parsed, structured, video_filename, ev_hash,
        progress_history, start_time_perf,
    ) -> Dict[str, Any]:
        progress_history.append({
            "stage": "CORRELATING", "status": "IN_PROGRESS",
            "message": "Running Event Engine scoped to analysis job.",
        })

        # Load tracks & detections for ONLY this analysis job.
        track_rows = (await db.execute(
            select(Track).where(
                Track.evidence_id == evidence_id,
                Track.analysis_job_id == job_id,
            )
        )).scalars().all()
        tracks = [
            {
                "track_number": t.track_number,
                "class_name": t.class_name,
                "first_seen_timestamp": t.first_seen_timestamp,
                "last_seen_timestamp": t.last_seen_timestamp,
                "duration": t.duration,
            }
            for t in track_rows
        ]
        det_rows = (await db.execute(
            select(Detection).where(
                Detection.evidence_id == evidence_id,
                Detection.analysis_job_id == job_id,
            )
        )).scalars().all()
        detections = [
            {
                "track_id": d.track_id,
                "class_name": d.class_name,
                "timestamp": d.timestamp,
                "bbox_x1": d.bbox_x1, "bbox_y1": d.bbox_y1,
                "bbox_x2": d.bbox_x2, "bbox_y2": d.bbox_y2,
            }
            for d in det_rows
        ]

        events = _dedup_events(EventEngine.detect_events(tracks, detections))

        # If a specific event_type was requested, filter.
        req_type = structured.get("event_type")
        if req_type:
            events = [e for e in events if e.get("event_type") == req_type]

        progress_history.append({
            "stage": "CORRELATING", "status": "COMPLETED",
            "message": f"Detected {len(events)} events (deduplicated).",
        })

        report_md = ForensicReportGenerator.generate_report(
            investigation_id=investigation_id,
            video_filename=video_filename,
            query_text=query_text,
            plan={"goal": f"Find events matching: {query_text}", "entities": [], "tasks": ["event_engine"]},
            verified_findings=[],
            timeline_events=events,
            evidence_hash=ev_hash,
        )

        return {
            "investigation_id": investigation_id,
            "evidence_id": evidence_id,
            "analysis_job_id": job_id,
            "query": query_text,
            "status": "COMPLETED",
            "execution_time_ms": round((time.time() - start_time_perf) * 1000.0, 2),
            "intent": {"type": INTENT_EVENT_SEARCH, "event_type": req_type},
            "evidence_basis": {"primary": "EVENT_ENGINE", "secondary": ["TRACKS", "DETECTIONS"]},
            "plan": {"goal": f"Investigate query: {query_text}", "tasks": ["event_engine"]},
            "progress": progress_history,
            "events": events,
            "findings": [],
            "report_markdown": report_md,
        }

    @classmethod
    async def _handle_semantic(
        cls, db, investigation_id, evidence_id, job_id,
        query_text, parsed, structured, video_filename, ev_hash,
        progress_history, start_time_perf,
    ) -> Dict[str, Any]:
        progress_history.append({
            "stage": "PLANNING", "status": "COMPLETED",
            "message": f"Semantic route for {structured.get('intent')}.",
        })
        progress_history.append({
            "stage": "RETRIEVING", "status": "IN_PROGRESS",
            "message": "Executing hybrid semantic retrieval.",
        })

        search_res = await InvestigationToolSystem.search_evidence(
            db=db, evidence_id=evidence_id,
            query_text=query_text, top_k=5,
            analysis_job_id=job_id,
        )
        raw_results = search_res.get("results", [])
        progress_history.append({
            "stage": "RETRIEVING", "status": "COMPLETED",
            "message": f"Retrieved {len(raw_results)} candidate records.",
        })

        progress_history.append({
            "stage": "VERIFYING", "status": "IN_PROGRESS",
            "message": "Verifying findings against raw DB observations.",
        })
        verified_findings: List[Dict[str, Any]] = []
        for item in raw_results:
            ver = await EvidenceVerifier.verify_finding(
                db=db,
                evidence_id=evidence_id,
                finding_statement=item.get("summary", ""),
                track_id=item.get("track_id"),
                keyframe_id=item.get("keyframe_id"),
                relevance_score=item.get("score", 0.0),
                analysis_job_id=job_id,
            )
            item["verification_status"] = ver["status"]
            item["support_reason"] = ver["support_reason"]
            item["limitations"] = ver["limitations"]
            verified_findings.append(item)
        progress_history.append({
            "stage": "VERIFYING", "status": "COMPLETED",
            "message": f"Verified {len(verified_findings)} findings.",
        })

        report_md = ForensicReportGenerator.generate_report(
            investigation_id=investigation_id,
            video_filename=video_filename,
            query_text=query_text,
            plan={"goal": f"Investigate query: {query_text}", "entities": parsed.get("entities", []), "tasks": ["semantic_retrieval"]},
            verified_findings=verified_findings,
            timeline_events=[],
            evidence_hash=ev_hash,
        )

        return {
            "investigation_id": investigation_id,
            "evidence_id": evidence_id,
            "analysis_job_id": job_id,
            "query": query_text,
            "status": "COMPLETED",
            "execution_time_ms": round((time.time() - start_time_perf) * 1000.0, 2),
            "intent": {
                "type": structured.get("intent"),
                "entity": structured.get("entity"),
                "attributes": structured.get("attributes", {}),
            },
            "evidence_basis": {"primary": "SEMANTIC_RETRIEVAL", "secondary": ["TRACKS", "KEYFRAMES"]},
            "plan": {"goal": f"Investigate query: {query_text}", "tasks": ["semantic_retrieval"]},
            "progress": progress_history,
            "events": [],
            "findings": verified_findings,
            "report_markdown": report_md,
        }

    # ======================================================================
    # No-job fallback
    # ======================================================================

    @classmethod
    def _no_job_response(
        cls, investigation_id, evidence_id, query_text, parsed,
        start_time_perf, progress_history,
    ) -> Dict[str, Any]:
        progress_history.append({
            "stage": "JOB_RESOLUTION", "status": "FAILED",
            "message": "No completed analysis job found for this evidence.",
        })
        return {
            "investigation_id": investigation_id,
            "evidence_id": evidence_id,
            "analysis_job_id": None,
            "query": query_text,
            "status": "NO_ANALYSIS_JOB",
            "execution_time_ms": round((time.time() - start_time_perf) * 1000.0, 2),
            "intent": parsed.get("structured_intent", {}),
            "plan": {},
            "progress": progress_history,
            "events": [],
            "findings": [],
            "report_markdown": (
                "# GUARDIANEYE Investigation — Precondition Not Met\n\n"
                f"Evidence #{evidence_id} has no completed analysis job. "
                "Run an analysis before invoking the investigation agent."
            ),
        }
