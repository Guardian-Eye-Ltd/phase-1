import re
import logging
from typing import Dict, Any, Optional, List

from app.services.agents.taxonomy import ENTITY_CLASSES, is_known_entity

logger = logging.getLogger(__name__)


# Investigator-facing intent types for the orchestrator's router. These
# supplement the legacy "query_type" field consumed by HybridSearchEngine.
INTENT_COUNT = "COUNT"
INTENT_FRAME_COUNT = "FRAME_COUNT"
INTENT_SEARCH = "SEARCH"
INTENT_ATTRIBUTE_SEARCH = "ATTRIBUTE_SEARCH"
INTENT_TEMPORAL_SEARCH = "TEMPORAL_SEARCH"
INTENT_RELATION_SEARCH = "RELATION_SEARCH"
INTENT_EVENT_SEARCH = "EVENT_SEARCH"
INTENT_SUMMARY = "SUMMARY"


# Phrases that unambiguously indicate aggregation.
_COUNT_TRIGGERS = [
    r"\btotal number of\b",
    r"\bnumber of\b",
    r"\bhow many\b",
    r"\bcount(?:\s+the)?\b",
    r"\btotal\s+(?:count\s+of|of)?\b",
    r"\bquantity of\b",
]

# Entity-token -> investigator-facing entity key. Built from the canonical
# taxonomy so a single edit flows everywhere.
_ENTITY_TOKEN_MAP: Dict[str, str] = {}
for entity_key, class_names in ENTITY_CLASSES.items():
    for cn in class_names:
        _ENTITY_TOKEN_MAP[cn] = entity_key
    _ENTITY_TOKEN_MAP[entity_key] = entity_key
# A few natural-language aliases the raw taxonomy doesn't need to know about.
_ENTITY_TOKEN_MAP.update({
    "people": "person",
    "persons": "person",
    "humans": "person",
    "individual": "person",
    "cars": "car",
    "automobile": "car",
    "automobiles": "car",
    "vehicles": "vehicle",
    "traffic": "vehicle",
    "buses": "bus",
    "trucks": "truck",
    "lorry": "truck",
    "lorries": "truck",
    "motorcycles": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "bike": "bicycle",
    "bikes": "bicycle",
    "vans": "van",
    "backpacks": "backpack",
    "bags": "backpack",
})


def _detect_entity(text_lower: str) -> Optional[str]:
    """
    Deterministic entity detection. Prefers the most specific concrete class
    name over the generic bucket (e.g. "cars and buses" -> "car" primary, but
    "vehicles" -> "vehicle").
    """
    hits: List[str] = []
    for token, entity in _ENTITY_TOKEN_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", text_lower):
            hits.append(entity)
    if not hits:
        return None
    # Priority: specific > generic. "vehicle" is the only generic bucket.
    specific = [h for h in hits if h != "vehicle"]
    if specific:
        return specific[0]
    return hits[0]


# Words that look plate-shaped but are ordinary English / query vocabulary.
_PLATE_STOPWORDS = {
    "VEHICLE", "VEHICLES", "PERSON", "PEOPLE", "SEARCH", "TRACK", "TRACKS",
    "CAMERA", "VIDEO", "EVIDENCE", "SHIRT", "BLACK", "WHITE", "GREEN", "BROWN",
    "FIND", "SHOW", "WEARING", "CARRYING", "BETWEEN", "SECONDS", "AROUND",
}


def _extract_plate_candidate(text_lower: str) -> Optional[str]:
    """
    Pull an explicit licence-plate token out of a query.

    Deliberately conservative: requires a letter+digit mix of 5-10 characters,
    so ordinary words and bare numbers never match. Returning None is always
    preferable to inventing a plate.
    """
    upper = text_lower.upper()

    # Explicit cue ("plate KL01AB1234", "number plate ABC123") takes priority.
    cued = re.search(
        r"(?:PLATE|REGISTRATION|REGO|NUMBER\s+PLATE)\s*(?:NO\.?|NUMBER|IS|=|:)?\s*([A-Z0-9][A-Z0-9\- ]{3,11}[A-Z0-9])",
        upper,
    )
    candidates = []
    if cued:
        candidates.append(cued.group(1))
    else:
        candidates.extend(re.findall(r"\b[A-Z0-9]{5,10}\b", upper))

    for raw in candidates:
        token = re.sub(r"[^A-Z0-9]", "", raw)
        if not (5 <= len(token) <= 10):
            continue
        if token in _PLATE_STOPWORDS:
            continue
        # A plate must mix letters and digits — "SHIRT" and "123456" are not plates.
        if not (re.search(r"[A-Z]", token) and re.search(r"\d", token)):
            continue
        return token
    return None


def _canonical_colour(colour: str) -> str:
    """Map query words onto the stored colour vocabulary (color_naming.COLOR_NAMES)."""
    return {"gray": "grey", "silver": "grey", "cream": "beige", "khaki": "beige"}.get(colour, colour)


def _parse_timestamp_phrase(text_lower: str) -> Optional[float]:
    """Pull "at 9 seconds", "at 01:20", "at 9s", "at 9.5 seconds" style."""
    m = re.search(r"\bat\s+(\d{1,2}:\d{2})\b", text_lower)
    if m:
        parts = m.group(1).split(":")
        return float(int(parts[0]) * 60 + int(parts[1]))
    m = re.search(r"\bat\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", text_lower)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(?:at\s+)?second\s+(\d+(?:\.\d+)?)\b", text_lower)
    if m:
        return float(m.group(1))
    return None


class QueryIntentParser:
    """
    Parses investigator natural language queries into structured forensic search intents
    (OBJECT_SEARCH, TRACK_SEARCH, ATTRIBUTE_SEARCH, RELATIONSHIP_SEARCH, BEHAVIORAL_SEARCH, TEMPORAL_SEARCH, ACTIVITY_SEARCH)
    and enforces document type candidate filtering rules.
    """

    KNOWN_ENTITIES = {
        "person": ["person", "people", "human", "pedestrian", "man", "woman", "guy", "individual", "suspect"],
        "car": ["car", "vehicle", "automobile", "sedan", "suv", "hatchback", "van"],
        "truck": ["truck", "pickup", "lorry", "van"],
        "bus": ["bus", "coach"],
        "bicycle": ["bicycle", "bike", "cyclist"],
        "motorcycle": ["motorcycle", "motorbike"],
        "backpack": ["backpack", "bag", "rucksack", "pack", "knapsack", "handbag", "luggage", "suitcase", "box"]
    }

    KNOWN_CLOTHING = [
        "shirt", "t-shirt", "tshirt", "pants", "jeans", "shorts", "skirt", "trousers", "jacket",
        "hoodie", "coat", "dress", "hat", "cap", "helmet", "suit", "sweater", "top", "vest"
    ]

    KNOWN_COLORS = [
        "red", "blue", "green", "black", "white", "yellow", "dark", "light",
        "grey", "gray", "brown", "orange", "purple", "pink", "silver", "khaki",
        "beige", "cream"
    ]

    BEHAVIORAL_KEYWORDS = [
        "loitering", "loiter", "dwelling", "casing", "concealment", "conceal", "theft", "stealing",
        "altercation", "aggression", "fighting", "fight", "brawling", "sudden acceleration",
        "acceleration", "speeding", "running", "fell", "fallen", "collapse", "proximity"
    ]

    ACTIVITY_KEYWORDS = [
        "activity", "movement", "motion", "busy period", "quiet period", "busy",
        "quiet", "happened", "timeline", "chronological", "event", "events", "what happened",
        "unusual activity", "motion spike"
    ]

    @classmethod
    def parse_time_to_seconds(cls, time_str: str) -> Optional[float]:
        """Converts MM:SS or H:MM:SS format into total seconds."""
        time_str = time_str.strip().lower()
        parts = time_str.split(":")
        try:
            if len(parts) == 2:
                mins, secs = int(parts[0]), int(parts[1])
                return float(mins * 60 + secs)
            elif len(parts) == 3:
                hrs, mins, secs = int(parts[0]), int(parts[1]), int(parts[2])
                return float(hrs * 3600 + mins * 60 + secs)
        except ValueError:
            pass
        return None

    @classmethod
    def parse_query(cls, query_text: str) -> Dict[str, Any]:
        """
        Extracts structured intent parameters and candidate document type constraints from natural language query.
        """
        text_lower = query_text.lower().strip()
        intent: Dict[str, Any] = {
            "raw_query": query_text,
            "primary_entity": None,
            "entities": [],
            "objects": [],
            "required_entities": [],
            "clothing": [],
            "colors": [],
            "behavioral_terms": [],
            "track_id": None,
            "start_time": None,
            "end_time": None,
            "spatial_relation": None,
            "spatial_target": None,
            "is_timeline_query": False,
            "is_activity_query": False,
            "is_behavioral_query": False,
            "is_count_query": bool(re.search(r"\b(how many|count|total number|number of)\b", text_lower)),
            "query_type": "OBJECT_SEARCH",
            "candidate_document_types": ["TRACK", "KEYFRAME", "BEHAVIORAL_EVENT"],
            "required_evidence": []
        }

        # 1. Track ID Extraction ("Track 17", "track #4", "track 12")
        track_match = re.search(r"\btrack\s*#?\s*(\d+)\b", text_lower)
        if track_match:
            intent["track_id"] = int(track_match.group(1))

        # 2. Time Range / Timestamp Extraction
        between_match = re.search(r"between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})", text_lower)
        if between_match:
            intent["start_time"] = cls.parse_time_to_seconds(between_match.group(1))
            intent["end_time"] = cls.parse_time_to_seconds(between_match.group(2))

        around_match = re.search(r"(?:around|at|@)\s+(\d{1,2}:\d{2})", text_lower)
        if around_match:
            ts = cls.parse_time_to_seconds(around_match.group(1))
            if ts is not None:
                intent["start_time"] = max(0.0, ts - 30.0)
                intent["end_time"] = ts + 30.0

        after_match = re.search(r"after\s+(\d{1,2}:\d{2})", text_lower)
        if after_match:
            intent["start_time"] = cls.parse_time_to_seconds(after_match.group(1))

        before_match = re.search(r"before\s+(\d{1,2}:\d{2})", text_lower)
        if before_match:
            intent["end_time"] = cls.parse_time_to_seconds(before_match.group(1))

        # 3. Behavioral Terms Extraction
        for beh in cls.BEHAVIORAL_KEYWORDS:
            if re.search(rf"\b{beh}\b", text_lower):
                intent["behavioral_terms"].append(beh)
                intent["is_behavioral_query"] = True

        # 4. Activity / Timeline Query Detection
        for kw in cls.ACTIVITY_KEYWORDS:
            if kw in text_lower:
                intent["is_activity_query"] = True
                if kw in ["what happened", "timeline", "chronological", "event", "events"]:
                    intent["is_timeline_query"] = True

        # 5. Entity & Object Extraction
        extracted_entities = []
        extracted_objects = []
        for category, keywords in cls.KNOWN_ENTITIES.items():
            for kw in keywords:
                if re.search(rf"\b{kw}\b", text_lower):
                    if category in ["backpack", "bag"]:
                        extracted_objects.append(category)
                    else:
                        extracted_entities.append(category)
                    break

        intent["entities"] = list(set(extracted_entities))
        intent["objects"] = list(set(extracted_objects))
        intent["required_entities"] = list(set(extracted_entities + extracted_objects))

        if "person" in intent["entities"]:
            intent["primary_entity"] = "person"
        elif intent["entities"]:
            intent["primary_entity"] = intent["entities"][0]
        elif intent["objects"]:
            intent["primary_entity"] = intent["objects"][0]

        # 6. Clothing Attribute Extraction
        for item in cls.KNOWN_CLOTHING:
            if re.search(rf"\b{item}\b", text_lower):
                intent["clothing"].append(item)

        # 7. Color Attribute Extraction
        for color in cls.KNOWN_COLORS:
            if re.search(rf"\b{color}\b", text_lower):
                intent["colors"].append(color)

        # 8. Spatial Relation Extraction
        near_match = re.search(r"near\s+(?:the\s+)?(vehicle|car|truck|bus|entrance|door|building|person|gate)", text_lower)
        if near_match:
            intent["spatial_relation"] = "near"
            intent["spatial_target"] = near_match.group(1)

        # 9. Categorize Query into Forensic Intent Types & Assign Candidate Document Types
        if intent["track_id"] is not None:
            intent["query_type"] = "TRACK_SEARCH"
            intent["candidate_document_types"] = ["TRACK"]
            intent["required_evidence"] = ["TRACK"]

        elif intent["is_behavioral_query"]:
            intent["query_type"] = "BEHAVIORAL_SEARCH"
            intent["candidate_document_types"] = ["BEHAVIORAL_EVENT", "TRACK", "KEYFRAME", "SPATIAL_RELATION"]
            intent["required_evidence"] = ["BEHAVIORAL_EVENT", "TRACK"]

        elif intent["clothing"] or (intent["colors"] and intent["primary_entity"] == "person"):
            intent["query_type"] = "ATTRIBUTE_SEARCH"
            intent["candidate_document_types"] = ["TRACK", "KEYFRAME"]
            intent["required_evidence"] = ["TRACK", "KEYFRAME"]

        elif intent["spatial_relation"] is not None or (len(intent["entities"]) >= 2):
            intent["query_type"] = "RELATIONSHIP_SEARCH"
            intent["candidate_document_types"] = ["SPATIAL_RELATION", "BEHAVIORAL_EVENT", "TRACK", "KEYFRAME"]
            intent["required_evidence"] = ["SPATIAL_RELATION", "TRACK", "KEYFRAME"]

        elif intent["is_activity_query"] or ("happened" in text_lower and not intent["primary_entity"]):
            intent["query_type"] = "ACTIVITY_SEARCH"
            intent["candidate_document_types"] = ["ACTIVITY_INTERVAL", "BEHAVIORAL_EVENT", "TRACK", "KEYFRAME", "SPATIAL_RELATION"]
            intent["required_evidence"] = ["ACTIVITY_INTERVAL", "KEYFRAME"]

        elif intent["start_time"] is not None or intent["end_time"] is not None:
            intent["query_type"] = "TEMPORAL_SEARCH"
            intent["candidate_document_types"] = ["ACTIVITY_INTERVAL", "BEHAVIORAL_EVENT", "TRACK", "KEYFRAME", "SPATIAL_RELATION"]
            intent["required_evidence"] = ["TRACK", "KEYFRAME", "ACTIVITY_INTERVAL"]

        else:
            intent["query_type"] = "OBJECT_SEARCH"
            intent["candidate_document_types"] = ["TRACK", "KEYFRAME", "BEHAVIORAL_EVENT"]
            intent["required_evidence"] = ["TRACK", "KEYFRAME"]

        # ---------------------------------------------------------------
        # Structured investigator intent (COUNT / FRAME_COUNT / ... / SUMMARY).
        # This is the field the orchestrator routes on. Legacy "query_type"
        # above stays for HybridSearchEngine's existing candidate filtering.
        # ---------------------------------------------------------------
        structured = cls._classify_structured_intent(text_lower, intent)
        intent["structured_intent"] = structured

        logger.info(
            f"[INTENT] Query='{query_text}' | legacy_type={intent['query_type']} | "
            f"structured={structured['intent']} entity={structured.get('entity')} "
            f"aggregation={structured.get('aggregation')} timestamp={structured.get('timestamp')}"
        )

        return intent

    @classmethod
    def _classify_structured_intent(cls, text_lower: str, legacy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce the investigator-facing intent record consumed by the
        InvestigationOrchestrator router. Deterministic — no LLM.
        """
        base: Dict[str, Any] = {
            "intent": INTENT_SEARCH,
            "entity": None,
            "scope": "ENTIRE_VIDEO",
            "aggregation": None,
            "timestamp": None,
            "start_time": legacy.get("start_time"),
            "end_time": legacy.get("end_time"),
            "attributes": {},
            "event_type": None,
            "confidence": 0.9,
        }

        entity = _detect_entity(text_lower)
        base["entity"] = entity

        has_count_trigger = any(re.search(p, text_lower) for p in _COUNT_TRIGGERS)
        timestamp = _parse_timestamp_phrase(text_lower)

        # ----- SUMMARY -----
        if re.search(r"\b(summari[sz]e|give me a summary|overview of the video)\b", text_lower):
            base["intent"] = INTENT_SUMMARY
            base["confidence"] = 0.95
            return base

        # ----- EVENT_SEARCH -----
        if legacy.get("is_behavioral_query") and not has_count_trigger:
            base["intent"] = INTENT_EVENT_SEARCH
            # Canonicalize to the EventEngine labels where possible.
            terms = legacy.get("behavioral_terms") or []
            if any("acceler" in t or "speed" in t for t in terms):
                base["event_type"] = "SUDDEN_ACCELERATION"
            elif any("loiter" in t or "dwell" in t for t in terms):
                base["event_type"] = "LOITERING"
            elif any("conceal" in t or "theft" in t or "stealing" in t for t in terms):
                base["event_type"] = "SUSPICIOUS_CONCEALMENT"
            elif any("altercation" in t or "fight" in t or "aggress" in t or "brawl" in t for t in terms):
                base["event_type"] = "PHYSICAL_ALTERCATION"
            elif any("fallen" in t or "fell" in t or "collapse" in t for t in terms):
                base["event_type"] = "FALLEN_POSTURE"
            elif any("proximity" in t for t in terms):
                base["event_type"] = "PERSON_VEHICLE_PROXIMITY"
            base["confidence"] = 0.92
            return base

        # ----- FRAME_COUNT -----
        if has_count_trigger and timestamp is not None and entity:
            base["intent"] = INTENT_FRAME_COUNT
            base["scope"] = "TIMESTAMP"
            base["timestamp"] = timestamp
            base["aggregation"] = "DISTINCT_ACTIVE_TRACK_COUNT"
            base["confidence"] = 0.97
            return base

        # ----- COUNT -----
        if has_count_trigger and entity:
            base["intent"] = INTENT_COUNT
            base["aggregation"] = "DISTINCT_TRACK_COUNT"
            if base["start_time"] is not None or base["end_time"] is not None:
                base["scope"] = "TIME_WINDOW"
            base["confidence"] = 0.98
            return base

        # ----- ATTRIBUTE_SEARCH: license plate -----
        # Checked before colour/clothing because a plate is a far stronger
        # identifier than any visual attribute in the same query.
        plate = _extract_plate_candidate(text_lower)
        if plate:
            base["intent"] = INTENT_ATTRIBUTE_SEARCH
            base["entity"] = entity or "vehicle"
            base["attributes"] = {"license_plate_text": plate}
            base["confidence"] = 0.95
            return base

        # ----- ATTRIBUTE_SEARCH: no upper garment -----
        if re.search(r"\b(shirtless|topless|bare[- ]chest(ed)?|no shirt|without (a )?shirt)\b", text_lower):
            base["intent"] = INTENT_ATTRIBUTE_SEARCH
            base["entity"] = "person"
            base["attributes"] = {"upper_garment_presence": "absent"}
            return base

        # ----- ATTRIBUTE_SEARCH -----
        # Attribute keys are the canonical column names used by
        # TrackAttributeAggregate, so tools can filter without translation.
        if legacy.get("clothing") or legacy.get("objects") or (legacy.get("colors") and entity):
            base["intent"] = INTENT_ATTRIBUTE_SEARCH
            attrs: Dict[str, Any] = {}
            colour = _canonical_colour(legacy["colors"][0]) if legacy["colors"] else None
            if entity == "person":
                if colour:
                    # "wearing black" refers to the upper garment unless the
                    # query names a lower garment explicitly.
                    lower_named = any(
                        c in text_lower
                        for c in ("pants", "jeans", "trousers", "shorts", "skirt")
                    )
                    key = "lower_garment_color" if lower_named else "upper_garment_color"
                    attrs[key] = colour
                if legacy["clothing"]:
                    attrs["clothing"] = legacy["clothing"]
            elif entity in ("car", "truck", "bus", "motorcycle", "van", "vehicle", "bicycle"):
                if colour:
                    attrs["vehicle_color"] = colour
            elif colour:
                attrs["color"] = colour

            if legacy.get("objects"):
                # Carried items are detector classes; a generic "bag" only
                # asserts that some bag is carried.
                attrs["carries_bag"] = "true"
                for item, words in (("suitcase", ("suitcase", "luggage")),
                                    ("handbag", ("handbag", "purse")),
                                    ("backpack", ("backpack", "rucksack", "knapsack"))):
                    if any(re.search(rf"\b{w}\b", text_lower) for w in words):
                        attrs["carried_item"] = item
                        break
                if not base["entity"]:
                    base["entity"] = "person"
            base["attributes"] = attrs
            base["confidence"] = 0.9
            return base

        # ----- RELATION_SEARCH -----
        if legacy.get("spatial_relation") is not None or len(legacy.get("entities", [])) >= 2:
            base["intent"] = INTENT_RELATION_SEARCH
            return base

        # ----- TEMPORAL_SEARCH -----
        if base["start_time"] is not None or base["end_time"] is not None:
            base["intent"] = INTENT_TEMPORAL_SEARCH
            return base

        # ----- default SEARCH -----
        base["intent"] = INTENT_SEARCH
        return base
