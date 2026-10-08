"""
Temporal attribute aggregation (Phase 10) and confidence decomposition (Phase 20).

A single frame is never authoritative. For each (track, attribute) we collect
every per-frame observation, take a confidence-weighted vote, and emit one
aggregate carrying a decomposed confidence so a reader can see *why* the number
is what it is.
"""
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# Attributes we aggregate, mapped to the entity type they describe.
PERSON_ATTRIBUTES = [
    "upper_garment_presence",
    "upper_garment_color",
    "lower_garment_color",
    "upper_garment_type",
    "lower_garment_type",
    "headwear",
    "carried_item",
]

VEHICLE_ATTRIBUTES = [
    "vehicle_color",
    "vehicle_body_style",
]

# Values that mean "the model declined" — never aggregated into a consensus.
_NULL_VALUES = {"unknown", "none", "n/a", "", None}

# An aggregate below this confidence is marked WITHHELD rather than OBSERVED.
MIN_AGGREGATE_CONFIDENCE = 0.35

# Minimum fraction of observations that must agree for a consensus to stand.
MIN_CONSENSUS_RATIO = 0.50


def _is_null_value(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in _NULL_VALUES)


def compute_confidence_breakdown(
    *,
    mean_model_confidence: float,
    consensus_ratio: float,
    observation_count: int,
    track_observation_count: int,
    capability_state: str,
) -> Dict[str, float]:
    """
    Deterministic confidence decomposition. Documented and reproducible —
    the LLM never assigns these.

    Axes:
      model      — mean confidence the classifier reported
      temporal   — how many frames agreed, relative to how many were observed
      coverage   — how much of the track's lifetime produced an observation
      consensus_ratio — fraction of observations backing the winning value
      support    — discount applied to the agreement axes when few frames exist
      capability — penalty for a DEGRADED capability (not ground truth)

    `support` exists because a single observation trivially has perfect
    consensus and perfect coverage with itself. Without discounting, one weak
    frame would score as highly as five agreeing ones.
    """
    coverage = 0.0
    if track_observation_count > 0:
        coverage = min(1.0, observation_count / float(track_observation_count))

    # Temporal support saturates: 1 frame is weak, 5+ agreeing frames is strong.
    temporal = min(1.0, observation_count / 5.0) * consensus_ratio

    # Agreement axes earn full credit only once 3+ observations exist.
    support = min(1.0, observation_count / 3.0)

    capability_factor = {
        "AVAILABLE": 1.0,
        "DEGRADED": 0.80,
        "NOT_AVAILABLE": 0.0,
    }.get(capability_state, 0.5)

    return {
        "model": round(float(mean_model_confidence), 4),
        "temporal": round(float(temporal), 4),
        "coverage": round(float(coverage), 4),
        "consensus_ratio": round(float(consensus_ratio), 4),
        "support": round(float(support), 4),
        "capability": round(float(capability_factor), 4),
    }


def overall_confidence(breakdown: Dict[str, float]) -> float:
    """
    Weighted combination of the decomposed axes. Weights are fixed and
    documented so the same inputs always yield the same score.

    Coverage and consensus are multiplied by `support` so that thin evidence
    cannot inherit full marks on axes that are trivially satisfied by a
    single observation.
    """
    support = breakdown.get("support", 1.0)
    score = (
        0.45 * breakdown.get("model", 0.0)
        + 0.30 * breakdown.get("temporal", 0.0)
        + 0.10 * breakdown.get("coverage", 0.0) * support
        + 0.15 * breakdown.get("consensus_ratio", 0.0) * support
    )
    return round(float(score * breakdown.get("capability", 1.0)), 4)


class AttributeAggregator:
    """Turns per-frame observations into per-track consensus attributes."""

    @classmethod
    def extract_observations(
        cls,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Flatten tracked detections into individual attribute observations.
        Only detections that carry a track_id produce observations — an
        untracked detection cannot be attributed to an entity.
        """
        observations: List[Dict[str, Any]] = []

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            cls_name = (det.get("class_name") or "").lower()
            if cls_name == "person":
                entity_type, attrs = "PERSON", PERSON_ATTRIBUTES
            elif cls_name in ("car", "truck", "bus", "motorcycle", "van", "bicycle", "vehicle"):
                entity_type, attrs = "VEHICLE", VEHICLE_ATTRIBUTES
            else:
                entity_type, attrs = "OBJECT", []

            for attr in attrs:
                value = det.get(attr)
                if _is_null_value(value):
                    continue
                confidence = det.get(f"{attr}_confidence")
                if confidence is None:
                    # Classifier ran but confidence was not captured — record it
                    # at a deliberately low value rather than assuming certainty.
                    confidence = 0.30
                observations.append({
                    "track_number": track_id,
                    "entity_type": entity_type,
                    "attribute": attr,
                    "value": str(value).strip().lower(),
                    "confidence": float(confidence),
                    "frame_number": det.get("frame_number", 0),
                    "timestamp": float(det.get("timestamp", 0.0)),
                    "source": det.get(f"{attr}_source", "CLIP_ZERO_SHOT"),
                })

            # carries_bag is boolean, handled separately so False is preserved
            if entity_type == "PERSON" and det.get("carries_bag") is not None:
                observations.append({
                    "track_number": track_id,
                    "entity_type": "PERSON",
                    "attribute": "carries_bag",
                    "value": "true" if det.get("carries_bag") else "false",
                    "confidence": float(det.get("carries_bag_confidence") or 0.30),
                    "frame_number": det.get("frame_number", 0),
                    "timestamp": float(det.get("timestamp", 0.0)),
                    "source": det.get("carries_bag_source", "CLIP_ZERO_SHOT"),
                })

        # Licence plates are read per vehicle track (alpr_service.read_vehicle_plates),
        # not per detection, and are merged into the observation list by the runner.
        return observations

    @classmethod
    def aggregate(
        cls,
        observations: List[Dict[str, Any]],
        track_observation_counts: Optional[Dict[int, int]] = None,
        capability_states: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Confidence-weighted vote per (track, attribute).

        Returns one aggregate per (track, attribute) with the winning value, its
        decomposed confidence, and how many observations supported vs dissented.
        """
        track_observation_counts = track_observation_counts or {}
        capability_states = capability_states or {}

        # (track, entity_type, attribute) -> value -> [confidences]
        grouped: Dict[Tuple[int, str, str], Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        timespans: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)

        for obs in observations:
            key = (obs["track_number"], obs["entity_type"], obs["attribute"])
            grouped[key][obs["value"]].append(obs["confidence"])
            timespans[key].append(obs["timestamp"])

        aggregates: List[Dict[str, Any]] = []

        for (track_number, entity_type, attribute), value_map in grouped.items():
            total_obs = sum(len(v) for v in value_map.values())
            if total_obs == 0:
                continue

            if attribute == "license_plate_text":
                aggregates.append(cls._aggregate_plate(
                    track_number, entity_type, value_map,
                    timespans[(track_number, entity_type, attribute)],
                    capability_states.get("license_plate_text", "DEGRADED"),
                ))
                continue

            # Weighted vote: a value's score is the sum of its confidences, so
            # three low-confidence reads don't beat one strong consistent one
            # unless they genuinely accumulate.
            scored = {
                value: (sum(confs), len(confs))
                for value, confs in value_map.items()
            }
            winner, (winner_score, winner_count) = max(
                scored.items(), key=lambda kv: (kv[1][0], kv[1][1])
            )

            consensus_ratio = winner_count / float(total_obs)
            mean_conf = sum(value_map[winner]) / float(winner_count)

            capability_key = {
                "upper_garment_presence": "person_garment_color",
                "upper_garment_color": "person_garment_color",
                "lower_garment_color": "person_garment_color",
                "vehicle_color": "vehicle_color",
                "license_plate_text": "license_plate_text",
            }.get(attribute, attribute)
            cap_state = capability_states.get(capability_key, "DEGRADED")

            breakdown = compute_confidence_breakdown(
                mean_model_confidence=mean_conf,
                consensus_ratio=consensus_ratio,
                observation_count=winner_count,
                track_observation_count=track_observation_counts.get(track_number, total_obs),
                capability_state=cap_state,
            )
            final_conf = overall_confidence(breakdown)

            # Weak or contested consensus is withheld, not asserted.
            if final_conf < MIN_AGGREGATE_CONFIDENCE or consensus_ratio < MIN_CONSENSUS_RATIO:
                status = "WITHHELD"
            else:
                status = "OBSERVED"

            ts_list = timespans[(track_number, entity_type, attribute)]
            aggregates.append({
                "track_number": track_number,
                "entity_type": entity_type,
                "attribute": attribute,
                "value": winner,
                "confidence": final_conf,
                "observation_count": total_obs,
                "supporting_count": winner_count,
                "dissenting_count": total_obs - winner_count,
                "first_observed_at": min(ts_list) if ts_list else 0.0,
                "last_observed_at": max(ts_list) if ts_list else 0.0,
                "status": status,
                "confidence_breakdown": breakdown,
            })

        cls._apply_cross_attribute_consistency(aggregates)

        logger.info(
            "[ATTR_AGG] aggregated %d observations into %d track attributes "
            "(%d withheld, %d contradicted)",
            len(observations), len(aggregates),
            sum(1 for a in aggregates if a["status"] == "WITHHELD"),
            sum(1 for a in aggregates if a["status"] == "CONTRADICTED"),
        )
        return aggregates

    @staticmethod
    def _apply_cross_attribute_consistency(aggregates: List[Dict[str, Any]]) -> None:
        """
        A shirt colour cannot stand when the same track's consensus is that no
        upper garment is worn: such a colour comes only from the minority of
        frames where skin was misread. Seen on real footage — swimmers whose
        presence vote was 39/42 "absent" still produced a 3/3 "white" colour.
        """
        by_track: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for a in aggregates:
            by_track[a["track_number"]][a["attribute"]] = a
        for attrs in by_track.values():
            presence = attrs.get("upper_garment_presence")
            if not presence or presence["value"] != "absent" or presence["status"] != "OBSERVED":
                continue
            for dependent in ("upper_garment_color", "upper_garment_type"):
                a = attrs.get(dependent)
                if a and a["status"] in ("OBSERVED", "WITHHELD"):
                    a["status"] = "CONTRADICTED"
                    a["confidence_breakdown"] = {
                        **(a.get("confidence_breakdown") or {}),
                        "contradicted_by": (
                            f"upper_garment_presence=absent in {presence['supporting_count']}/"
                            f"{presence['observation_count']} observations"
                        ),
                    }

    @staticmethod
    def _aggregate_plate(track_number, entity_type, value_map, ts_list, cap_state) -> Dict[str, Any]:
        """
        Plates are voted character by character, not as whole strings: three
        reads of KL01AB1234 and one of KL01AB1284 still agree on 9 of 10
        positions, while two reads that differ on a character must not be
        resolved by picking one.
        """
        from app.services.pipeline.alpr_service import vote_plate

        reads = [(value, conf) for value, confs in value_map.items() for conf in confs]
        vote = vote_plate(reads)
        capability = {"AVAILABLE": 1.0, "DEGRADED": 0.8, "NOT_AVAILABLE": 0.0}.get(cap_state, 0.5)
        confidence = round(vote.get("confidence", 0.0) * capability, 4)
        exact = vote.get("exact_matches", 0)
        return {
            "track_number": track_number,
            "entity_type": entity_type,
            "attribute": "license_plate_text",
            "value": vote["value"] or "",
            "confidence": confidence,
            "observation_count": len(reads),
            "supporting_count": exact,
            "dissenting_count": len(reads) - exact,
            "first_observed_at": min(ts_list) if ts_list else 0.0,
            "last_observed_at": max(ts_list) if ts_list else 0.0,
            "status": vote["status"],
            "confidence_breakdown": {
                "method": "character_vote",
                "agreement": vote.get("agreement", 0.0),
                "min_position_majority": vote.get("min_position_majority", 0.0),
                "reads_considered": vote.get("reads_considered", 0),
                "format_complete": vote.get("format_complete", False),
                "format_issue": vote.get("format_issue"),
                "capability": capability,
                "withheld_reason": vote.get("reason", ""),
            },
        }
