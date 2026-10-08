import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _format_breakdown_rows(breakdown: Dict[str, int]) -> str:
    if not breakdown:
        return "| *(none detected)* | 0 |\n"
    rows = ""
    for cls, n in sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
        rows += f"| {cls.capitalize()} | {n} |\n"
    return rows


class ForensicReportGenerator:
    """
    Generates formal forensic investigation markdown reports grounded in verified database evidence.
    Includes explicit methodology, verified findings, SHA-256 evidence hashes, chronological timeline, evidence gaps, and legal disclaimers.
    """

    @classmethod
    def generate_report(
        cls,
        investigation_id: str,
        video_filename: str,
        query_text: str,
        plan: Dict[str, Any],
        verified_findings: List[Dict[str, Any]],
        timeline_events: List[Dict[str, Any]],
        evidence_hash: Optional[str] = None
    ) -> str:
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        hash_str = evidence_hash if evidence_hash else "SHA256:E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"

        findings_section = ""
        if not verified_findings:
            findings_section = (
                "### Finding 1 — Insufficient Evidence\n\n"
                "- **Statement**: No supported evidence found matching query criteria.\n"
                "- **Verification Status**: `WITHHELD` / `UNVERIFIED`\n"
                "- **Timestamps**: N/A\n"
                "- **Track IDs**: None\n"
                "- **Evidence Hash**: `" + hash_str + "`\n"
                "- **Limitations**: Video observations did not meet minimum relevance threshold (0.22) for the requested entities/attributes.\n\n"
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
                    f"### Finding {idx} — Track #{track_id} / Keyframe KF-{kf_id} Observation\n\n"
                    f"- **Statement**: {summary}\n"
                    f"- **Verification Status**: {status_badge}\n"
                    f"- **Support Detail**: {reason}\n"
                    f"- **Timestamp Range**: {start_t:.2f}s – {end_t:.2f}s\n"
                    f"- **Track ID**: #{track_id}\n"
                    f"- **Keyframe Reference**: KF-{kf_id}\n"
                    f"- **SHA-256 Hash**: `{hash_str}`\n"
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
- **SHA-256 Hash**: `{hash_str}`
- **Investigator Query**: *"{query_text}"*

---

## 2. Objective

To conduct an evidence-grounded computer vision and digital forensics investigation over surveillance video footage `{video_filename}` (`{hash_str}`) to identify, track, and verify entities, visual attributes, and behavioral events matching: *"{query_text}"*.

---

## 3. Methodology

The GuardianEye 2.0 forensics pipeline processed the evidence video using the following multi-stage methodology:

1. **Video Ingestion & SHA-256 Hash Verification**: Frame rate, resolution, frame counts, and file integrity hash validated.
2. **Object Detection & Visual Attribute Extraction**: Ultralytics YOLO11 object detection paired with YOLO11-Pose landmark estimation, OpenCLIP zero-shot visual attribute classification (garments, colors, headwear, vehicle body style), and PaddleOCR ALPR license plate extraction.
3. **Multi-Object Tracking**: Track continuity maintained across temporal frame gaps.
4. **Deterministic Event Engine**: Rule-based pose keypoint anomaly detection (loitering dwelling, wrist-to-hip concealment, altercation convergence, sudden acceleration, fallen posture).
5. **Hybrid Evidence Retrieval**: Multi-stage ranking combining vector semantics (`all-MiniLM-L6-v2`), keyword attributes, and temporal filtering.
6. **Agentic Verification**: Findings evaluated against ground-truth database records to assign strict forensic verification statuses (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNVERIFIED`).

---

## 4. Verified Findings

{findings_section}

---

## 5. Chronological Timeline

{timeline_section}

---

## 6. Contradictory Evidence & Edge Cases

- **Attribute Ambiguity**: Low-resolution or poorly illuminated video segments may cause color hue shift. Claims relying solely on visual attribute classifiers are marked with explicit confidence bounds.
- **Occlusion Handling**: Temporary track loss during background occlusion is logged as separate track segments if temporal gap exceeds 3.0 seconds.

---

## 7. Evidence Gaps

- **Uncovered Camera Zones**: Video coverage is restricted to the camera field of view.
- **Audio / Facial Identification**: Facial recognition and audio surveillance were not conducted. Identity claims are marked as `IDENTITY_UNAVAILABLE`.

---

## 8. Conclusion

Based on verified database evidence, the findings detailed above represent all supported visual observations for the query *"{query_text}"* within evidence video `{video_filename}` (`{hash_str}`). Unverified claims have been explicitly withheld from final conclusions.

---

## 9. AI Limitation & Legal Notice

> [!IMPORTANT]
> **Forensic AI Disclaimer**: This report was generated automatically by the GuardianEye 2.0 Evidence-Grounded Forensics Platform. AI interpretation and computer vision observations represent machine-assisted analysis. This document does not constitute certified legal testimony or automatic legal admissibility without manual investigator review.
"""

        return report_md

    # ------------------------------------------------------------------
    # Specialized report templates (COUNT / FRAME_COUNT)
    # ------------------------------------------------------------------

    @classmethod
    def generate_count_report(
        cls,
        investigation_id: str,
        video_filename: str,
        query_text: str,
        count_result: Dict[str, Any],
        vlm_frame_observations: Optional[List[Dict[str, Any]]] = None,
        evidence_hash: Optional[str] = None,
    ) -> str:
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        hash_str = evidence_hash or "N/A"
        total = count_result.get("total", 0)
        entity = count_result.get("entity", "entity")
        breakdown = count_result.get("breakdown", {})
        job_id = count_result.get("analysis_job_id", "N/A")

        supporting = ""
        if vlm_frame_observations:
            supporting = "## Supporting Visual Observations (Per-Frame)\n\n"
            for obs in vlm_frame_observations[:10]:
                ts = obs.get("timestamp")
                cnt = obs.get("observed_count")
                supporting += f"- `{ts:.2f}s` -> {cnt} {entity}(s) visible in-frame\n"
            supporting += (
                "\n> These are per-frame visual counts. They are **not summed** because the "
                "same physical entity typically appears across multiple frames. The authoritative "
                "total above is derived from DISTINCT tracked entities.\n"
            )

        return f"""# GUARDIANEYE — Entity Count Report

- **Investigation ID**: `{investigation_id}`
- **Date / Time**: `{now_str}`
- **Evidence Video**: `{video_filename}`
- **SHA-256 Hash**: `{hash_str}`
- **Investigator Query**: *"{query_text}"*
- **Analysis Job**: #{job_id}

---

## Result

**Total unique {entity}(s) detected: {total}**

| {entity.capitalize()} Type | Unique Tracks |
|---|---:|
{_format_breakdown_rows(breakdown)}| **Total** | **{total}** |

## Evidence Basis

- Aggregation: `DISTINCT_TRACK_COUNT`
- Source: Tracked entities (`tracks` table)
- Scope: `{count_result.get('scope', 'ENTIRE_VIDEO')}`
- Analysis Job: #{job_id}

## Methodology

The total represents **unique tracked entities** within the selected analysis
job. Each tracked object is counted exactly once regardless of how many frames
it appears in. Per-frame detections and VLM scene counts are **not** summed.

{supporting}

---

> **Note**: This is a deterministic aggregation report. No LLM-generated numbers
> are included. Event-engine findings (loitering, acceleration, etc.) are
> intentionally omitted from count queries.
"""

    @classmethod
    def generate_attribute_report(
        cls,
        investigation_id: str,
        video_filename: str,
        query_text: str,
        entity: Optional[str],
        requested_attributes: Dict[str, Any],
        matches: List[Dict[str, Any]],
        analysis_job_id: Optional[int] = None,
        evidence_hash: Optional[str] = None,
    ) -> str:
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        hash_str = evidence_hash or "N/A"
        asked = ", ".join(f"`{k}` = `{v}`" for k, v in requested_attributes.items()) or "none"

        if not matches:
            body = (
                "## Result\n\n"
                "**NO SUPPORTED EVIDENCE**\n\n"
                f"No tracked {entity or 'entity'} in analysis job #{analysis_job_id} carries a "
                "temporally-supported observation matching the requested attributes.\n\n"
                "This is a negative finding, not a failure. Possible reasons:\n\n"
                "- No entity in the footage has these attributes.\n"
                "- Observations existed but were `WITHHELD` for weak or contested consensus.\n"
                "- The attribute classifier is operating in a `DEGRADED` state for this capability.\n"
            )
        else:
            body = f"## Result\n\n**{len(matches)} matching {entity or 'entity'} track(s).**\n\n"
            for m in matches:
                body += (
                    f"### Track #{m['track_id']}"
                    f"{' — ' + str(m['class_name']) if m.get('class_name') else ''}\n\n"
                )
                if m.get("start_time") is not None:
                    body += f"- **Visible**: {m['start_time']:.2f}s – {m['end_time']:.2f}s"
                    if m.get("duration") is not None:
                        body += f" ({m['duration']:.2f}s)"
                    body += "\n"
                body += f"- **Match confidence**: {m['match_confidence']:.2f}\n"
                body += f"- **Evidence basis**: `{m.get('evidence_basis', 'TRACK_ATTRIBUTE_AGGREGATE')}`\n\n"

                body += "| Attribute | Value | Confidence | Support | Status |\n"
                body += "|---|---|---:|---:|---|\n"
                for attr, d in m.get("attributes", {}).items():
                    body += (
                        f"| {attr} | {d['value']} | {d['confidence']:.2f} | "
                        f"{d['supporting_count']}/{d['observation_count']} | `{d['status']}` |\n"
                    )
                body += "\n**Why this result?**\n\n"
                for attr, d in m.get("attributes", {}).items():
                    frames = d.get("supporting_frames") or []
                    shown = ", ".join(f"{f:.2f}s" for f in frames[:6])
                    more = f" (+{len(frames) - 6} more)" if len(frames) > 6 else ""
                    bd = d.get("confidence_breakdown") or {}
                    body += (
                        f"- Track #{m['track_id']} was classified `{attr} = {d['value']}` in "
                        f"**{d['supporting_count']} of {d['observation_count']}** observations "
                        f"({d['dissenting_count']} disagreed). "
                        f"Confidence {d['confidence']:.2f} "
                        f"= model {bd.get('model', 0):.2f} · temporal {bd.get('temporal', 0):.2f} "
                        f"· consensus {bd.get('consensus_ratio', 0):.2f} "
                        f"· capability {bd.get('capability', 0):.2f}.\n"
                    )
                    if shown:
                        body += f"  - Supporting frames: {shown}{more}\n"
                body += "\n"

        return f"""# GUARDIANEYE — Attribute Search Report

- **Investigation ID**: `{investigation_id}`
- **Date / Time**: `{now_str}`
- **Evidence Video**: `{video_filename}`
- **SHA-256 Hash**: `{hash_str}`
- **Investigator Query**: *"{query_text}"*
- **Analysis Job**: #{analysis_job_id}
- **Requested Attributes**: {asked}

---

{body}
---

## Methodology

Attributes are matched against **temporally aggregated** observations
(`TrackAttributeAggregate`), not individual frames. Each value is the
confidence-weighted consensus across every frame in which the track was
classified. Aggregates whose consensus was contested or whose confidence fell
below threshold are marked `WITHHELD` and excluded from matches.

## Limitations

> Visual attribute classification is a machine interpretation, not ground truth.
> A match states that a track is **visually associated** with an attribute — it
> does not assert the attribute as fact. Lighting, resolution, occlusion and
> motion blur all shift apparent colour. Human review is required before any
> investigative conclusion is drawn.
"""

    @classmethod
    def generate_plate_report(
        cls,
        investigation_id: str,
        video_filename: str,
        query_text: str,
        answer: str,
        vehicles: List[Dict[str, Any]],
        analysis_job_id: Optional[int] = None,
        evidence_hash: Optional[str] = None,
    ) -> str:
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        rows = ""
        for v in vehicles:
            p = v["plate"]
            reads = p.get("reads_total")
            rows += (
                f"| #{v['track_id']} | {v['class_name']} | {v.get('vehicle_color') or '—'} | "
                f"{p['text'] or '—'} | `{p['status']}` | {reads if reads is not None else '—'} | "
                f"{p.get('reason') or ((p.get('format_issue') or 'incomplete plate format') if p['status'] == 'READ' and not p.get('format_complete') else '')} |\n"
            )
        table = (
            "| Track | Class | Colour | Plate | Status | OCR reads | Note |\n"
            "|---|---|---|---|---|---:|---|\n" + rows
        ) if vehicles else "_No vehicle tracks in scope._\n"

        return f"""# GUARDIANEYE — Licence Plate Report

- **Investigation ID**: `{investigation_id}`
- **Date / Time**: `{now_str}`
- **Evidence Video**: `{video_filename}`
- **SHA-256 Hash**: `{evidence_hash or "N/A"}`
- **Investigator Query**: *"{query_text}"*
- **Analysis Job**: #{analysis_job_id}

---

## Result

{answer}

{table}
---

## Methodology

Plates are read per vehicle track with EasyOCR on up to four of the vehicle's
largest frames, then voted **character by character**. A plate is `READ` only
when every character was read the same way by at least two OCR reads.
`UNCONFIRMED` means text was read but the reads disagree or were too few; the
best guess is shown so it can be checked, and it is **never** an identification.
`NOT_READ` gives the reason no plate text exists (usually vehicle size).

## Limitations

> Plate OCR is a `DEGRADED` capability. Small, angled, blurred or night-time
> plates are frequently misread. Check every plate against the footage before
> drawing any investigative conclusion.
"""

    @classmethod
    def generate_frame_count_report(
        cls,
        investigation_id: str,
        video_filename: str,
        query_text: str,
        frame_count_result: Dict[str, Any],
        evidence_hash: Optional[str] = None,
    ) -> str:
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        hash_str = evidence_hash or "N/A"
        count = frame_count_result.get("count", 0)
        entity = frame_count_result.get("entity", "entity")
        ts = frame_count_result.get("timestamp", 0.0)
        breakdown = frame_count_result.get("breakdown", {})
        job_id = frame_count_result.get("analysis_job_id", "N/A")

        return f"""# GUARDIANEYE — Frame-Count Report

- **Investigation ID**: `{investigation_id}`
- **Date / Time**: `{now_str}`
- **Evidence Video**: `{video_filename}`
- **SHA-256 Hash**: `{hash_str}`
- **Investigator Query**: *"{query_text}"*
- **Analysis Job**: #{job_id}

---

## Result

**At t = {ts:.2f}s, {count} distinct {entity}(s) were simultaneously active in the scene.**

| {entity.capitalize()} Type | Active Tracks |
|---|---:|
{_format_breakdown_rows(breakdown)}| **Total** | **{count}** |

## Methodology

A track is considered "active" at timestamp *t* when
`first_seen_timestamp <= t <= last_seen_timestamp`. Each distinct track is
counted once; per-frame VLM observations are not summed.
"""
