"""
Licence-plate reading per vehicle *track*.

Pipeline: pick the largest few frames of each vehicle track → lower part of
the vehicle crop (where plates sit) → contrast enhancement → EasyOCR (its own
text detector locates the plate characters) → normalise + validate format →
character-level voting across frames.

A plate is only asserted when several reads agree on every character; one
read, or reads that disagree on any position, are WITHHELD — the system never
fills in characters it did not see consistently.
"""
import logging
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Indian formats: KL01AB1234 style and Bharat series 22BH1234AB.
INDIAN_PLATE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$|^\d{2}BH\d{4}[A-Z]{1,2}$")
# State / union-territory prefixes (incl. legacy OR, UA and the 2024 TG).
# Used only to label a consensus read as a complete number, never to filter
# or rewrite OCR output: "JA3K961" is a consistent read with an impossible prefix.
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TG", "TN", "TR", "TS", "UA", "UK", "UP", "WB",
}
MIN_VEHICLE_WIDTH_FOR_OCR_PX = 120
MAX_FRAMES_PER_TRACK = 4
MIN_SECONDS_BETWEEN_SAMPLES = 0.5
# Measured on synthetic plates: reads of plates narrower than ~80px came back
# wrong at OCR confidence 0.08-0.25; correct reads were 0.67-0.98.
MIN_READ_CONFIDENCE = 0.35

# Visually confusable glyphs, used only to fit a read to the registered format.
_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "T": "7"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "7": "T", "4": "A"}

_reader = None
_reader_error: Optional[str] = None
_lock = threading.Lock()


def _get_reader():
    global _reader, _reader_error
    if _reader is not None or _reader_error is not None:
        return _reader
    with _lock:
        if _reader is None and _reader_error is None:
            try:
                import easyocr
                _reader = easyocr.Reader(["en"], gpu=settings.DEVICE != "cpu", verbose=False)
                logger.info("[ALPR] EasyOCR ready.")
            except Exception as e:
                _reader_error = f"{type(e).__name__}: {e}"
                logger.warning(f"[ALPR] OCR unavailable: {_reader_error}")
    return _reader


@dataclass
class PlateRead:
    text: str               # normalised A-Z0-9
    raw_text: str
    ocr_confidence: float
    validation_score: float  # 1.0 known format, 0.7 letters+digits, 0.4 digits only

    @property
    def score(self) -> float:
        return self.ocr_confidence * self.validation_score


def normalise_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def correct_to_indian_format(text: str, max_substitutions: int = 3) -> Optional[str]:
    """
    Fit an OCR read to the Indian plate format by swapping only visually
    confusable glyphs (O<->0, I<->1, B<->8, ...). Returns the corrected string
    with the fewest substitutions, or None if no such fit exists. Nothing is
    inserted or deleted: a character that was not read is never invented.
    """
    if INDIAN_PLATE.match(text):
        return text
    n = len(text)
    # Structure: 2 letters, 1-2 digits, 0-3 letters, 1-4 digits.
    for d1 in (2, 1):
        for l2 in range(0, 4):
            d2 = n - 2 - d1 - l2
            if not (1 <= d2 <= 4):
                continue
            pattern = "L" * 2 + "D" * d1 + "L" * l2 + "D" * d2
            out, subs = [], 0
            for ch, want in zip(text, pattern):
                if want == "D" and not ch.isdigit():
                    if ch not in _TO_DIGIT:
                        break
                    ch, subs = _TO_DIGIT[ch], subs + 1
                elif want == "L" and not ch.isalpha():
                    if ch not in _TO_LETTER:
                        break
                    ch, subs = _TO_LETTER[ch], subs + 1
                out.append(ch)
            else:
                if subs <= max_substitutions and INDIAN_PLATE.match("".join(out)):
                    return "".join(out)
    return None


def plate_format_issue(text: str) -> Optional[str]:
    """None when text is a complete Indian plate number, else why it is not."""
    if not INDIAN_PLATE.match(text):
        return "does not match the Indian plate pattern (characters may be missing)"
    if not text[:2].isdigit() and text[:2] not in INDIAN_STATE_CODES:
        return f"'{text[:2]}' is not an Indian state code, so at least one letter is likely misread"
    return None


def validation_score(text: str) -> float:
    if not (4 <= len(text) <= 10) or not re.search(r"\d", text):
        return 0.0
    if INDIAN_PLATE.match(text):
        return 1.0
    return 0.7 if re.search(r"[A-Z]", text) else 0.4


def _merge_line_fragments(results) -> List[Tuple[str, float]]:
    """EasyOCR often splits a plate into pieces; join fragments on the same line."""
    items = []
    for box, text, conf in results:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        items.append({"x1": min(xs), "x2": max(xs), "cy": (min(ys) + max(ys)) / 2,
                      "h": max(ys) - min(ys), "text": text, "conf": float(conf)})
    items.sort(key=lambda i: (round(i["cy"] / max(i["h"], 1)), i["x1"]))
    lines: List[List[dict]] = []
    for it in items:
        if lines and abs(lines[-1][-1]["cy"] - it["cy"]) < 0.6 * max(it["h"], lines[-1][-1]["h"]) \
                and it["x1"] - lines[-1][-1]["x2"] < 1.5 * it["h"]:
            lines[-1].append(it)
        else:
            lines.append([it])
    out = []
    for line in lines:
        out.append(("".join(i["text"] for i in line), float(np.mean([i["conf"] for i in line]))))
        out.extend((i["text"], i["conf"]) for i in line if len(line) > 1)
    return out


def read_plate_candidates(vehicle_crop: np.ndarray) -> List[PlateRead]:
    """All plausible plate strings in one vehicle crop, best first."""
    reader = _get_reader()
    if reader is None or vehicle_crop is None or vehicle_crop.size == 0:
        return []
    h, w = vehicle_crop.shape[:2]
    region = vehicle_crop[int(0.35 * h):, :]
    scale = min(3.0, max(1.0, 640.0 / max(w, 1)))
    region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    results = reader.readtext(gray, allowlist=PLATE_ALLOWLIST, detail=1, paragraph=False)
    reads = []
    for raw, conf in _merge_line_fragments(results):
        if conf < MIN_READ_CONFIDENCE:
            continue
        text = normalise_plate(raw)
        corrected = correct_to_indian_format(text) if 8 <= len(text) <= 10 else None
        if corrected:
            text = corrected
        v = validation_score(text)
        if v > 0:
            reads.append(PlateRead(text, raw, round(conf, 4), v))
    reads.sort(key=lambda r: r.score, reverse=True)
    return reads


def vote_plate(reads: List[Tuple[str, float]], min_reads: int = 2,
               min_position_majority: float = 0.6) -> Dict[str, Any]:
    """
    Character-level consensus over (text, confidence) reads of one vehicle.

    Withheld unless at least `min_reads` reads share the winning length and
    every character position has a clear (>= min_position_majority) majority.
    """
    if not reads:
        return {"value": None, "status": "WITHHELD", "reason": "no reads"}
    lengths = Counter(len(t) for t, _ in reads)
    best_len = max(lengths, key=lambda L: (lengths[L], sum(c for t, c in reads if len(t) == L)))
    same = [(t, c) for t, c in reads if len(t) == best_len]

    consensus, fractions, disputed = [], [], []
    for i in range(best_len):
        weights: Dict[str, float] = defaultdict(float)
        votes: Dict[str, int] = defaultdict(int)
        for t, c in same:
            weights[t[i]] += max(c, 1e-3)
            votes[t[i]] += 1
        ch, wt = max(weights.items(), key=lambda kv: kv[1])
        consensus.append(ch)
        fractions.append(wt / sum(weights.values()))
        # Confidence weighting alone let one confident read outvote one less
        # confident read ("CA82545" from EA82545 vs CA82545): every asserted
        # character must be read the same way at least min_reads times.
        if votes[ch] < min_reads:
            disputed.append(f"character {i + 1}: " + " vs ".join(sorted(votes)))
    value = "".join(consensus)
    agreement = float(np.mean(fractions))
    mean_conf = float(np.mean([c for _, c in same]))
    exact = sum(1 for t, _ in same if t == value)
    confidence = agreement * mean_conf * min(1.0, len(same) / 3.0)

    if len(same) < min_reads:
        status, reason = "WITHHELD", f"only {len(same)} read(s) of this length"
    elif disputed:
        status, reason = "WITHHELD", "reads disagree on " + "; ".join(disputed)
    elif min(fractions) < min_position_majority:
        status, reason = "WITHHELD", "reads disagree on at least one character"
    else:
        status, reason = "OBSERVED", ""
    return {
        "value": value, "status": status, "reason": reason,
        # A consistent read can still be partial (OCR dropped characters);
        # only a full match of the plate format is a complete number.
        "format_complete": plate_format_issue(value) is None,
        "format_issue": plate_format_issue(value),
        "confidence": round(confidence, 4), "agreement": round(agreement, 4),
        "reads_considered": len(same), "reads_total": len(reads), "exact_matches": exact,
        "min_position_majority": round(min(fractions), 4),
    }


def plate_unread_reason(width_px: Optional[int]) -> str:
    """Why a vehicle track has no plate read at all (shared by /entities and search)."""
    if width_px is not None and width_px < MIN_VEHICLE_WIDTH_FOR_OCR_PX:
        return (f"vehicle too small to read a plate (largest view {width_px}px wide; "
                f"needs at least {MIN_VEHICLE_WIDTH_FOR_OCR_PX}px)")
    return "no readable plate text in the vehicle's clearest frames"


def _best_frames(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(dets, key=lambda d: (d["bbox_x2"] - d["bbox_x1"]) * (d["bbox_y2"] - d["bbox_y1"]), reverse=True)
    chosen: List[Dict[str, Any]] = []
    for d in ranked:
        if all(abs(d["timestamp"] - c["timestamp"]) >= MIN_SECONDS_BETWEEN_SAMPLES for c in chosen):
            chosen.append(d)
        if len(chosen) >= MAX_FRAMES_PER_TRACK:
            break
    return chosen


def read_vehicle_plates(
    sampled_frames: List[Tuple[int, float, np.ndarray]],
    detections: List[Dict[str, Any]],
    vehicle_classes=("car", "truck", "bus", "motorcycle", "van"),
    read_fn=read_plate_candidates,
) -> List[Dict[str, Any]]:
    """One license_plate_text observation per (vehicle track, frame) that yielded a plausible read."""
    frames = {fn: frame for fn, _, frame in sampled_frames}
    by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for d in detections:
        if d.get("class_name") in vehicle_classes and d.get("track_id") is not None and d["frame_number"] in frames:
            by_track[d["track_id"]].append(d)

    observations = []
    ocr_runs = skipped_small = 0
    for track_id, dets in by_track.items():
        for rank, d in enumerate(_best_frames(dets)):
            frame = frames[d["frame_number"]]
            fh, fw = frame.shape[:2]
            x1, y1 = int(d["bbox_x1"] * fw), int(d["bbox_y1"] * fh)
            x2, y2 = int(d["bbox_x2"] * fw), int(d["bbox_y2"] * fh)
            if x2 - x1 < MIN_VEHICLE_WIDTH_FOR_OCR_PX:
                skipped_small += 1
                continue
            ocr_runs += 1
            reads = read_fn(frame[max(0, y1):y2, max(0, x1):x2])
            if not reads:
                # Frames are tried largest first; if the clearest view shows
                # no plate text, smaller views will not (saves ~3 s per frame).
                if rank == 0:
                    break
                continue
            best = reads[0]
            observations.append({
                "track_number": track_id,
                "entity_type": "VEHICLE",
                "attribute": "license_plate_text",
                "value": best.text,
                "confidence": round(best.score, 4),
                "frame_number": d["frame_number"],
                "timestamp": float(d["timestamp"]),
                "source": "ALPR_OCR",
            })
    logger.info(
        f"[ALPR] vehicle_tracks={len(by_track)} ocr_runs={ocr_runs} "
        f"skipped_too_small={skipped_small} plate_reads={len(observations)}"
    )
    return observations
