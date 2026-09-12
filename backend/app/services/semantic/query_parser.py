import re
from typing import Dict, Any, Optional, List

class QueryIntentParser:
    """
    Parses investigator natural language queries into structured search intents 
    without requiring heavy LLM inference overhead.
    """

    KNOWN_ENTITIES = {
        "person": ["person", "people", "human", "pedestrian", "man", "woman", "guy", "individual", "suspect"],
        "car": ["car", "vehicle", "automobile", "sedan", "suv"],
        "truck": ["truck", "van", "lorry", "pickup"],
        "bicycle": ["bicycle", "bike", "cyclist"],
        "motorcycle": ["motorcycle", "motorbike"],
        "backpack": ["backpack", "bag", "rucksack", "pack", "knapsack", "handbag", "luggage", "suitcase"]
    }

    KNOWN_CLOTHING = [
        "shirt", "t-shirt", "tshirt", "pants", "jeans", "trousers", "jacket",
        "hoodie", "coat", "dress", "shorts", "hat", "cap", "suit", "sweater", "top", "vest"
    ]

    KNOWN_COLORS = [
        "red", "blue", "green", "black", "white", "yellow", "dark", "light",
        "grey", "gray", "brown", "orange", "purple", "pink"
    ]

    ACTIVITY_KEYWORDS = [
        "activity", "movement", "motion", "busy period", "quiet period", "busy",
        "quiet", "happened", "timeline", "chronological", "event", "events", "what happened"
    ]

    @classmethod
    def parse_time_to_seconds(cls, time_str: str) -> Optional[float]:
        """Converts MM:SS or H:MM:SS or HH:MM format into seconds."""
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
        Extracts structured intent parameters from natural language query.
        """
        text_lower = query_text.lower().strip()
        intent: Dict[str, Any] = {
            "raw_query": query_text,
            "primary_entity": None,
            "required_entities": [],
            "clothing": [],
            "colors": [],
            "objects": [],
            "track_id": None,
            "start_time": None,
            "end_time": None,
            "spatial_relation": None,
            "spatial_target": None,
            "is_timeline_query": False,
            "is_activity_query": False,
            "query_type": "GENERAL_SEMANTIC_QUERY",
            "candidate_document_types": ["TRACK", "KEYFRAME", "SPATIAL_RELATION", "DETECTION_SUMMARY"],
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

        # 3. Activity / Timeline Query Intent
        for kw in cls.ACTIVITY_KEYWORDS:
            if kw in text_lower:
                intent["is_activity_query"] = True
                if kw in ["what happened", "timeline", "chronological", "event", "events"]:
                    intent["is_timeline_query"] = True

        # 4. Known Entity & Object Extraction
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

        # 5. Clothing Attribute Extraction
        for item in cls.KNOWN_CLOTHING:
            if re.search(rf"\b{item}\b", text_lower):
                intent["clothing"].append(item)

        # 6. Color Attribute Extraction
        for color in cls.KNOWN_COLORS:
            if re.search(rf"\b{color}\b", text_lower):
                intent["colors"].append(color)

        # 7. Spatial Relation Extraction
        near_match = re.search(r"near\s+(?:the\s+)?(vehicle|car|truck|entrance|door|building|person|gate)", text_lower)
        if near_match:
            intent["spatial_relation"] = "near"
            intent["spatial_target"] = near_match.group(1)

        # 8. Determine Query Type & Candidate Document Types
        if intent["track_id"] is not None and ("track" in text_lower or len(text_lower.split()) <= 4):
            intent["query_type"] = "TRACK_QUERY"
            intent["candidate_document_types"] = ["TRACK"]
            intent["required_evidence"] = ["PERSON_TRACK", "OBJECT_TRACK"]

        elif intent["is_activity_query"] and not intent["primary_entity"] and not intent["clothing"]:
            intent["query_type"] = "ACTIVITY_QUERY"
            intent["candidate_document_types"] = [
                "ACTIVITY_INTERVAL", "TIMELINE_EVENT", "SPATIAL_RELATION", "TRACK", "KEYFRAME"
            ]
            intent["required_evidence"] = ["ACTIVITY_INTERVAL", "TIMELINE_EVENT"]

        elif intent["primary_entity"] == "person" or intent["clothing"] or "person" in text_lower or "people" in text_lower:
            intent["query_type"] = "PERSON_CLOTHING_QUERY"
            intent["candidate_document_types"] = ["TRACK", "KEYFRAME", "DETECTION_SUMMARY"]
            intent["required_evidence"] = ["PERSON_TRACK", "KEYFRAME", "VLM_OBSERVATION", "VISUAL_ATTRIBUTE_OBSERVATION"]

        elif intent["objects"] or (intent["primary_entity"] and intent["primary_entity"] != "person"):
            intent["query_type"] = "OBJECT_QUERY"
            intent["candidate_document_types"] = ["TRACK", "KEYFRAME", "DETECTION_SUMMARY"]
            intent["required_evidence"] = ["OBJECT_DETECTION", "KEYFRAME", "VLM_OBSERVATION"]

        else:
            intent["query_type"] = "GENERAL_SEMANTIC_QUERY"
            intent["candidate_document_types"] = ["TRACK", "KEYFRAME", "SPATIAL_RELATION", "DETECTION_SUMMARY"]
            intent["required_evidence"] = ["TRACK", "KEYFRAME", "VLM_OBSERVATION"]

        return intent
