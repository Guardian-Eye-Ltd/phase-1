import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ForensicReportGenerator:
    """
    Generates formal forensic investigation markdown reports grounded in verified database evidence.
    Includes explicit methodology, verified findings, chronological timeline, evidence gaps, and legal disclaimers.
    """

    @classmethod
    def generate_report(
        cls,
        investigation_id: str,
        video_filename: str,
        query_text: str,
        plan: Dict[str, Any],
        verified_findings: List[Dict[str, Any]],
        timeline_events: List[Dict[str, Any]]
    ) -> str:
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        findings_section = ""
        if not verified_findings:
            findings_section = (
                "### Finding 1 — Insufficient Evidence\n\n"
                "- **Statement**: No supported evidence found matching query criteria.\n"
                "- **Verification Status**: `WITHHELD` / `UNVERIFIED`\n"
                "- **Timestamps**: N/A\n"
                "- **Track IDs**: None\n"
                "- **Limitations**: Video observations did not meet minimum relevance threshold for the requested entities/attributes.\n\n"
            )
        else:
            for idx, f in enumerate(verified_findings, 1):
                status = f.get("verification_status", "UNVERIFIED")
                status_badge = f"`{status}`"
                track_id = f.get("track_id", "N/A")
                kf_id = f.get("keyframe_id", "N/A")
                start_t = f.get("start_time", 0.0)
                end_t = f.get("end_time", 0.0)
                summary = f.get("summary", "")
                reason = f.get("support_reason", f.get("confidence_reason", ""))
                limits = f.get("limitations", [])

                limits_str = ", ".join(limits) if limits else "None identified."

                findings_section += (
                    f"### Finding {idx} — Track #{track_id} Observation\n\n"
                    f"- **Statement**: {summary}\n"
                    f"- **Verification Status**: {status_badge}\n"
                    f"- **Support Detail**: {reason}\n"
                    f"- **Timestamp Range**: {start_t:.2f}s – {end_t:.2f}s\n"
                    f"- **Track ID**: #{track_id}\n"
                    f"- **Keyframe Reference**: KF-{kf_id}\n"
                    f"- **Known Limitations**: {limits_str}\n\n"
                )

        timeline_section = ""
        if not timeline_events:
            timeline_section = "*No chronological timeline events extracted for query scope.*\n"
        else:
            for ev in timeline_events:
                ts_str = f"{ev.get('start_time', 0.0):.2f}s"
                title = ev.get('title', 'Event')
                desc = ev.get('description', '')
                timeline_section += f"- **[{ts_str}]** {title}: {desc}\n"

        report_md = f"""# GUARDIANEYE 2.0 DIGITAL FORENSICS INVESTIGATION REPORT

---

## 1. Investigation Information

- **Investigation ID**: `{investigation_id}`
- **Date / Time**: `{now_str}`
- **Evidence Video**: `{video_filename}`
- **Investigator Query**: *"{query_text}"*

---

## 2. Objective

To conduct an evidence-grounded computer vision and digital forensics investigation over surveillance video footage `{video_filename}` to identify, track, and verify entities, visual attributes, and events matching: *"{query_text}"*.

---

## 3. Methodology

The GuardianEye 2.0 forensics pipeline processed the evidence video using the following multi-stage methodology:

1. **Video Ingestion & Metadata Extraction**: Frame rate, resolution, and frame counts validated.
2. **Object Detection & Visual Attribute Extraction**: Ultralytics YOLOv8 object detection paired with HSV color histogramming (upper/lower garment color analysis) and carried-bag proximity co-location.
3. **Multi-Object Tracking**: Track continuity maintained across temporal frame gaps.
4. **Deterministic Event Engine**: Rule-based detection of loitering, person-vehicle proximity, and object association.
5. **Hybrid Evidence Retrieval**: Multi-stage ranking combining vector semantics, keyword attributes, and temporal filtering.
6. **Agentic Verification**: Findings evaluated against ground-truth database records to assign strict forensic verification statuses (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNVERIFIED`).

---

## 4. Verified Findings

{findings_section}

---

## 5. Chronological Timeline

{timeline_section}

---

## 6. Contradictory Evidence & Edge Cases

- **Attribute Ambiguity**: Low-resolution or poorly illuminated video segments may cause color hue shift. Claims relying solely on color histograms are flagged as `PARTIALLY_SUPPORTED`.
- **Occlusion Handling**: Temporary track loss during background occlusion is logged as separate track segments if temporal gap exceeds 3.0 seconds.

---

## 7. Evidence Gaps

- **Uncovered Camera Zones**: Video coverage is restricted to the camera field of view.
- **Audio / Facial Identification**: Facial recognition and audio surveillance were not conducted. Identity claims are marked as `IDENTITY_UNAVAILABLE`.

---

## 8. Conclusion

Based on verified database evidence, the findings detailed above represent all supported visual observations for the query *"{query_text}"* within evidence video `{video_filename}`. Unverified claims have been explicitly withheld from final conclusions.

---

## 9. AI Limitation & Legal Notice

> [!IMPORTANT]
> **Forensic AI Disclaimer**: This report was generated automatically by the GuardianEye 2.0 Evidence-Grounded Forensics Platform. AI interpretation and computer vision observations represent machine-assisted analysis. This document does not constitute certified legal testimony or automatic legal admissibility without manual investigator review.
"""

        return report_md
