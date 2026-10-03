import logging
import uuid
import time
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.evidence import Evidence
from app.models.analysis import Track, Keyframe, Detection, ActivityInterval
from app.services.agents.llm_provider import LLMProvider
from app.services.agents.investigation_tools import InvestigationToolSystem
from app.services.agents.evidence_verifier import EvidenceVerifier
from app.services.agents.report_generator import ForensicReportGenerator
from app.services.pipeline.event_engine import EventEngine
from app.services.semantic.query_parser import QueryIntentParser

logger = logging.getLogger(__name__)

class InvestigationOrchestrator:
    """
    Multi-Agent Investigation Orchestrator executing:
    Planner Agent -> Investigator Tool System -> Evidence Verifier -> Forensic Report Generator.
    Enforces maximum step limits and evidence traceability.
    """

    @classmethod
    async def run_investigation(
        cls,
        db: AsyncSession,
        evidence_id: int,
        query_text: str,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        start_time_perf = time.time()
        investigation_id = f"INV-{uuid.uuid4().hex[:8].upper()}"

        logger.info(f"[INVESTIGATION] Starting agentic investigation '{investigation_id}' for evidence #{evidence_id}. Query: '{query_text}'")

        # 0. Load Evidence record
        ev_res = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
        evidence = ev_res.scalars().first()
        video_filename = evidence.original_filename if evidence else f"Evidence_{evidence_id}.mp4"

        progress_history = []
        
        # Step 1: Query Understanding & Planning Agent
        progress_history.append({"stage": "PLANNING", "status": "IN_PROGRESS", "message": "Decomposing natural language query into structured investigation tasks..."})
        
        parsed_intent = QueryIntentParser.parse_query(query_text)
        planner_prompt = (
            f"Generate a forensic investigation plan for query: '{query_text}'.\n"
            f"Extracted Intent: {parsed_intent}\n"
            f"Format response as structured JSON with keys 'goal', 'entities', 'tasks'."
        )
        plan_raw = await LLMProvider.generate(planner_prompt, json_mode=True)
        try:
            import json
            plan_json = json.loads(plan_raw)
        except Exception:
            plan_json = {
                "goal": f"Investigate query: '{query_text}'",
                "entities": parsed_intent.get("entities", ["person"]),
                "tasks": ["retrieve_evidence_tracks", "verify_attributes", "generate_report"]
            }

        progress_history.append({"stage": "PLANNING", "status": "COMPLETED", "message": f"Plan created: {plan_json.get('goal')}"})

        # Step 2: Evidence Retrieval Agent (Investigator Tool System)
        progress_history.append({"stage": "RETRIEVING", "status": "IN_PROGRESS", "message": "Executing controlled tool queries over database tracks and observations..."})
        
        search_res = await InvestigationToolSystem.search_evidence(
            db=db,
            evidence_id=evidence_id,
            query_text=query_text,
            top_k=5
        )

        raw_results = search_res.get("results", [])

        progress_history.append({"stage": "RETRIEVING", "status": "COMPLETED", "message": f"Retrieved {len(raw_results)} evidence candidate records."})

        # Step 3: Event Engine & Correlation Agent
        progress_history.append({"stage": "CORRELATING", "status": "IN_PROGRESS", "message": "Running Event Engine for loitering, proximity, and carried objects..."})
        
        trk_res = await db.execute(select(Track).where(Track.evidence_id == evidence_id))
        all_tracks = trk_res.scalars().all()
        track_dicts = [
            {
                "track_number": t.track_number,
                "class_name": t.class_name,
                "first_seen_timestamp": t.first_seen_timestamp,
                "last_seen_timestamp": t.last_seen_timestamp,
                "duration": t.duration
            }
            for t in all_tracks
        ]

        det_res = await db.execute(select(Detection).where(Detection.evidence_id == evidence_id))
        all_dets = [
            {
                "track_id": d.track_id,
                "class_name": d.class_name,
                "timestamp": d.timestamp,
                "bbox_x1": d.bbox_x1,
                "bbox_y1": d.bbox_y1,
                "bbox_x2": d.bbox_x2,
                "bbox_y2": d.bbox_y2,
                "upper_garment_color": getattr(d, 'upper_garment_color', 'unknown'),
                "lower_garment_color": getattr(d, 'lower_garment_color', 'unknown'),
                "carries_bag": getattr(d, 'carries_bag', False)
            }
            for d in det_res.scalars().all()
        ]

        detected_events = EventEngine.detect_events(track_dicts, all_dets)
        progress_history.append({"stage": "CORRELATING", "status": "COMPLETED", "message": f"Detected {len(detected_events)} forensic event correlations."})

        # Step 4: Evidence Verification Agent
        progress_history.append({"stage": "VERIFYING", "status": "IN_PROGRESS", "message": "Verifying finding claims against raw DB observations..."})
        
        verified_findings = []
        for item in raw_results:
            ver = await EvidenceVerifier.verify_finding(
                db=db,
                evidence_id=evidence_id,
                finding_statement=item.get("summary", ""),
                track_id=item.get("track_id"),
                keyframe_id=item.get("keyframe_id"),
                relevance_score=item.get("score", 0.0)
            )
            item["verification_status"] = ver["status"]
            item["support_reason"] = ver["support_reason"]
            item["limitations"] = ver["limitations"]
            verified_findings.append(item)

        progress_history.append({"stage": "VERIFYING", "status": "COMPLETED", "message": f"Verified {len(verified_findings)} findings."})

        # Step 5: Report Generator Agent
        progress_history.append({"stage": "GENERATING_REPORT", "status": "IN_PROGRESS", "message": "Compiling formal digital forensics report..."})
        
        report_markdown = ForensicReportGenerator.generate_report(
            investigation_id=investigation_id,
            video_filename=video_filename,
            query_text=query_text,
            plan=plan_json,
            verified_findings=verified_findings,
            timeline_events=detected_events
        )

        progress_history.append({"stage": "GENERATING_REPORT", "status": "COMPLETED", "message": "Report generation complete."})

        execution_time_ms = round((time.time() - start_time_perf) * 1000.0, 2)

        return {
            "investigation_id": investigation_id,
            "evidence_id": evidence_id,
            "query": query_text,
            "status": "COMPLETED",
            "execution_time_ms": execution_time_ms,
            "plan": plan_json,
            "progress": progress_history,
            "events": detected_events,
            "findings": verified_findings,
            "report_markdown": report_markdown
        }
