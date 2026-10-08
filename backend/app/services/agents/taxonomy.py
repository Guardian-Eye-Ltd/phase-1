"""
Canonical entity taxonomy for the investigation subsystem.

Single source of truth for which concrete class names count as which
investigator-facing entity. If a new vehicle type (e.g. "tractor") is added to
the detector, update it here and every call site inherits the change.
"""
from typing import Dict, List, Optional, Set


# Each key is an investigator-facing entity. Values are the concrete track
# class_names (lower-case) that belong to it.
ENTITY_CLASSES: Dict[str, List[str]] = {
    "vehicle": ["car", "truck", "bus", "motorcycle", "van", "vehicle"],
    "car": ["car"],
    "bus": ["bus"],
    "truck": ["truck"],
    "motorcycle": ["motorcycle", "motorbike"],
    "van": ["van"],
    "bicycle": ["bicycle", "bike"],
    "person": ["person", "human", "pedestrian"],
    "backpack": ["backpack", "bag", "handbag", "suitcase", "rucksack"],
}

# Buckets whose aggregate roll-up should prefer the most specific class when a
# track is both specific (e.g. "car") and generic (e.g. "vehicle"). Keeps
# "total vehicles" from double-counting a tracked car as car + vehicle.
GENERIC_TO_SPECIFIC_PRIORITY: Dict[str, List[str]] = {
    "vehicle": ["car", "truck", "bus", "motorcycle", "van"],
}


def classes_for_entity(entity: str) -> List[str]:
    """Return the list of concrete class_names that belong to this entity."""
    if entity is None:
        return []
    return list(ENTITY_CLASSES.get(entity.lower(), [entity.lower()]))


def normalize_entity_class(class_name: Optional[str]) -> Optional[str]:
    """
    Map a raw track class_name to its canonical investigator-facing bucket.
    Falls back to the raw name (lower-cased) when no bucket matches.
    """
    if not class_name:
        return None
    cn = class_name.lower()
    for bucket in ("car", "truck", "bus", "motorcycle", "van", "bicycle", "person", "backpack"):
        if cn in ENTITY_CLASSES[bucket]:
            return bucket
    # fall back: generic "vehicle" literal stays as-is
    if cn in ENTITY_CLASSES["vehicle"]:
        return "vehicle"
    return cn


def is_known_entity(entity: str) -> bool:
    return bool(entity) and entity.lower() in ENTITY_CLASSES


def all_vehicle_classes() -> List[str]:
    """Convenience accessor used by investigation tools."""
    return classes_for_entity("vehicle")
