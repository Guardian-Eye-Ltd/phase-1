"""
Declares which analytical capabilities this deployment can actually deliver.

The rule: a capability that no installed model can support returns an explicit
NOT_AVAILABLE record with a reason. It never returns a fabricated value and
never silently omits the field — a caller must be able to tell "we looked and
could not determine" apart from "we never looked".
"""
import importlib.util
import logging
from typing import Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


def _module_installed(name: str) -> bool:
    """Cheap importability probe — does not import or initialise the module."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _ocr_engine_available() -> bool:
    return _module_installed("paddleocr") or _module_installed("easyocr")


class CapabilityState:
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"            # Works, but with materially reduced reliability
    NOT_AVAILABLE = "NOT_AVAILABLE"  # No model installed that can do this


# Capabilities the pipeline exposes, with why each is in its current state.
# Keep reasons specific — they surface in forensic reports.
_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "person_garment_color": {
        "state": CapabilityState.DEGRADED,
        "reason": "Pose-guided torso/leg regions with the person's own face tone as the "
                  "skin reference and deterministic HSV colour naming. Regions that are "
                  "not visible produce no claim. No garment-segmentation model; lighting "
                  "and shadow can shift colour names.",
        "source": "POSE_CROP_COLOR_MODEL",
    },
    "vehicle_color": {
        "state": CapabilityState.DEGRADED,
        "reason": "CLIP with object-aware prompts combined with the pixel colour of the "
                  "body below the windscreen. Silver is reported as white or grey; glossy "
                  "reflections lower confidence.",
        "source": "COMBINED",
    },
    # Resolved at runtime below — depends on whether an OCR engine is installed.
    "license_plate_text": {
        "state": CapabilityState.DEGRADED,
        "reason": "EasyOCR on the lower part of each vehicle track's best frames with "
                  "character-level voting across frames. Vehicles narrower than 120px "
                  "are not read and no dedicated plate detector is installed, so distant "
                  "or angled plates usually yield no result.",
        "source": "ALPR_OCR",
    },
    "vehicle_body_style": {
        "state": CapabilityState.DEGRADED,
        "reason": "CLIP zero-shot (sedan / SUV / hatchback / pickup / van ...). Not validated "
                  "against labelled ground truth; overhead camera angles bias it towards 'SUV'. "
                  "The detector class (car / truck / bus / motorcycle) is the more reliable type.",
        "source": "CLIP_ZERO_SHOT",
    },
    "vehicle_make": {
        "state": CapabilityState.NOT_AVAILABLE,
        "reason": "Measured on real footage (2026-10-08): CLIP zero-shot make recognition "
                  "misidentified 5 of 5 vehicles, including a Ford F-150 reported as GMC "
                  "with 0.41 confidence. Needs a fine-grained make/model classifier and "
                  "close-range footage.",
        "source": "NOT_AVAILABLE",
    },
    "vehicle_model": {
        "state": CapabilityState.NOT_AVAILABLE,
        "reason": "Model-level recognition is finer-grained than make, which already "
                  "failed on real footage; requires a dedicated classifier.",
        "source": "NOT_AVAILABLE",
    },
    "calibrated_speed_kmh": {
        "state": CapabilityState.NOT_AVAILABLE,
        "reason": "No camera calibration (homography / road-plane geometry) is "
                  "configured. Real-world speed cannot be derived from pixel motion.",
        "source": "NOT_AVAILABLE",
    },
    "vlm_captioning": {
        "state": CapabilityState.DEGRADED,
        "reason": "General-purpose BLIP captioning: useful scene context, but not "
                  "authoritative for counts, identity, colours or plates, and "
                  "known to misname objects (e.g. buses as a train).",
        "source": "VLM",
    },
    # Resolved at runtime below — depends on whether insightface is installed.
    "face_recognition": {
        "state": CapabilityState.AVAILABLE,
        "reason": "InsightFace buffalo_l (SCRFD + ArcFace), detection and recognition only. "
                  "Results are similarity candidates (POSSIBLE_MATCH) for human review, never "
                  "identifications; accuracy falls sharply for small, blurred or turned-away faces.",
        "source": "FACE_EMBEDDING",
    },
    "fire_smoke_detection": {
        "state": CapabilityState.NOT_AVAILABLE,
        "reason": "No fire/smoke detection model is installed. Color-based inference "
                  "would produce unacceptable false positives.",
        "source": "NOT_AVAILABLE",
    },
}


def _resolve_runtime_state(name: str, cap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Downgrade a declared state when the environment cannot actually support it.
    A capability is only as good as the packages installed next to it.
    """
    if name == "license_plate_text":
        if not settings.ENABLE_ALPR:
            return {
                **cap,
                "state": CapabilityState.NOT_AVAILABLE,
                "reason": "ALPR is disabled by configuration (ENABLE_ALPR=false).",
                "source": "NOT_AVAILABLE",
            }
        if not _ocr_engine_available():
            return {
                **cap,
                "state": CapabilityState.NOT_AVAILABLE,
                "reason": "No OCR engine installed (neither paddleocr nor easyocr is "
                          "importable). License plate text cannot be read.",
                "source": "NOT_AVAILABLE",
            }

    if name == "face_recognition":
        if not settings.FACE_RECOGNITION_ENABLED:
            return {**cap, "state": CapabilityState.NOT_AVAILABLE,
                    "reason": "Face recognition disabled by configuration (FACE_RECOGNITION_ENABLED=false).",
                    "source": "NOT_AVAILABLE"}
        if not (_module_installed("insightface") and _module_installed("onnxruntime")):
            return {**cap, "state": CapabilityState.NOT_AVAILABLE,
                    "reason": "insightface / onnxruntime is not installed.",
                    "source": "NOT_AVAILABLE"}

    if name == "vlm_captioning":
        if not settings.VLM_ENABLED:
            return {**cap, "state": CapabilityState.NOT_AVAILABLE,
                    "reason": "VLM disabled by configuration (VLM_ENABLED=false).",
                    "source": "NOT_AVAILABLE"}
        if not _module_installed("transformers"):
            return {**cap, "state": CapabilityState.NOT_AVAILABLE,
                    "reason": "transformers is not installed.", "source": "NOT_AVAILABLE"}

    if name in ("person_garment_color", "vehicle_color") and not settings.ENABLE_ATTRIBUTE_CLASSIFICATION:
        return {
            **cap,
            "state": CapabilityState.NOT_AVAILABLE,
            "reason": "Attribute classification is disabled by configuration "
                      "(ENABLE_ATTRIBUTE_CLASSIFICATION=false).",
            "source": "NOT_AVAILABLE",
        }

    return cap


def get_capability(name: str) -> Dict[str, Any]:
    cap = _CAPABILITIES.get(name)
    if cap is None:
        return {
            "capability": name,
            "state": CapabilityState.NOT_AVAILABLE,
            "reason": f"Unknown capability '{name}'.",
            "source": "NOT_AVAILABLE",
        }
    return {"capability": name, **_resolve_runtime_state(name, cap)}


def is_available(name: str) -> bool:
    return get_capability(name)["state"] == CapabilityState.AVAILABLE


def is_usable(name: str) -> bool:
    """AVAILABLE or DEGRADED — usable, but DEGRADED must be labelled in output."""
    return get_capability(name)["state"] in (CapabilityState.AVAILABLE, CapabilityState.DEGRADED)


def unavailable_result(name: str, **extra) -> Dict[str, Any]:
    """
    Build the standard 'we cannot determine this' payload. Use this instead of
    returning None or a guessed value.
    """
    cap = get_capability(name)
    return {
        "value": None,
        "confidence": 0.0,
        "status": "WITHHELD",
        "source": "NOT_AVAILABLE",
        "capability_state": cap["state"],
        "reason": cap["reason"],
        **extra,
    }


def capability_report() -> Dict[str, Any]:
    """
    Snapshot of every capability as resolved for THIS environment — embedded in
    analysis manifests so a report reader can tell what the run could actually do.
    """
    out = {}
    for name in _CAPABILITIES:
        c = get_capability(name)
        out[name] = {"state": c["state"], "reason": c["reason"]}
    return out
