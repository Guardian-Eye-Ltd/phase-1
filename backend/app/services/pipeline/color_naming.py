"""
Deterministic colour naming for garment and vehicle regions.

Replaces closed-set CLIP colour classification, which has no "none" option and
therefore always names *some* colour — it reported bare skin as a "brown shirt"
and water as "white trousers". Here every pixel is named with fixed HSV rules,
skin pixels are measured separately, and the result carries the share of pixels
that agree, so a region that is mostly skin or mixed reports exactly that.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

COLOR_NAMES = ["black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"]


@dataclass
class ColorReading:
    color: Optional[str]          # None when no colour dominates
    share: float                  # fraction of non-skin pixels with that name
    skin_fraction: float          # fraction of all pixels that look like skin
    pixel_count: int
    distribution: Dict[str, float]


def skin_mask(region_bgr: np.ndarray) -> np.ndarray:
    """Classic YCrCb skin range. Crude: also catches some beige/orange cloth."""
    ycrcb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    return (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135) & (y >= 60)


def name_pixels(region_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel colour index into COLOR_NAMES."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    h = hsv[..., 0] * 2          # degrees 0-358
    s = hsv[..., 1]
    v = hsv[..., 2]
    out = np.full(h.shape, COLOR_NAMES.index("grey"), dtype=np.int32)

    chromatic = s >= 50
    # Hue families for saturated pixels.
    hue_rules = [
        ("red", (h < 15) | (h >= 345)),
        ("orange", (h >= 15) & (h < 40)),
        ("yellow", (h >= 40) & (h < 70)),
        ("green", (h >= 70) & (h < 165)),
        ("blue", (h >= 165) & (h < 255)),
        ("purple", (h >= 255) & (h < 295)),
        ("pink", (h >= 295) & (h < 345)),
    ]
    for name, mask in hue_rules:
        out[chromatic & mask] = COLOR_NAMES.index(name)

    # Dark warm hues read as brown; pale reds as pink.
    out[chromatic & (((h < 40) | (h >= 345)) & (v < 140))] = COLOR_NAMES.index("brown")
    out[chromatic & ((h >= 40) & (h < 70)) & (v < 120)] = COLOR_NAMES.index("brown")
    out[chromatic & ((h < 15) | (h >= 345)) & (s < 110) & (v >= 170)] = COLOR_NAMES.index("pink")

    # Beige / cream: warm hue, low-to-moderate saturation, bright.
    out[((h >= 15) & (h < 55)) & (s >= 20) & (s < 90) & (v >= 140)] = COLOR_NAMES.index("beige")
    chromatic = chromatic & (out != COLOR_NAMES.index("beige"))

    # Achromatic pixels by brightness.
    achromatic = ~chromatic & (out != COLOR_NAMES.index("beige"))
    out[achromatic & (v >= 185)] = COLOR_NAMES.index("white")
    out[achromatic & (v < 185) & (v >= 70)] = COLOR_NAMES.index("grey")
    out[(v < 55) | (achromatic & (v < 70))] = COLOR_NAMES.index("black")
    return out


def skin_reference(face_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Median Lab colour of this person's facial skin, or None. A person-specific
    reference separates bare skin from skin-coloured cloth (a beige coat sits in
    the generic skin range but is far from the wearer's own face tone).
    """
    if face_bgr is None or face_bgr.size == 0:
        return None
    mask = skin_mask(face_bgr)
    if mask.sum() < 20:
        return None
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab[mask.reshape(-1)], axis=0)


def _person_skin_mask(region_bgr: np.ndarray, ref_lab: np.ndarray, max_chroma_dist: float = 9.0) -> np.ndarray:
    """
    Chroma (Lab a,b) distance only — lightness is ignored. Measured on real
    frames: shirtless torsos sat 3.6–7.6 from the same person's face tone while
    being much brighter (wet, sunlit), a cream coat sat at 12.4, dark jackets
    at 23–28. Including lightness misclassified bright wet skin as cloth.
    """
    lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.sqrt((lab[..., 1] - ref_lab[1]) ** 2 + (lab[..., 2] - ref_lab[2]) ** 2)
    return skin_mask(region_bgr) & (chroma < max_chroma_dist)


def read_color(region_bgr: Optional[np.ndarray], exclude_skin: bool = True,
               min_share: float = 0.40, min_pixels: int = 150,
               skin_ref_lab: Optional[np.ndarray] = None) -> ColorReading:
    if region_bgr is None or region_bgr.size == 0:
        return ColorReading(None, 0.0, 0.0, 0, {})
    # Large regions are downsampled for speed; small ones are never upscaled.
    if region_bgr.shape[0] * region_bgr.shape[1] > 120_000:
        scale = (120_000 / (region_bgr.shape[0] * region_bgr.shape[1])) ** 0.5
        region_bgr = cv2.resize(region_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    region_bgr = cv2.GaussianBlur(region_bgr, (3, 3), 0)

    total = region_bgr.shape[0] * region_bgr.shape[1]
    if not exclude_skin:
        skin = np.zeros(region_bgr.shape[:2], bool)
    elif skin_ref_lab is not None:
        skin = _person_skin_mask(region_bgr, skin_ref_lab)
    else:
        skin = skin_mask(region_bgr)
    skin_fraction = float(skin.mean()) if total else 0.0

    names = name_pixels(region_bgr)[~skin]
    if names.size < min_pixels:
        return ColorReading(None, 0.0, round(skin_fraction, 3), int(names.size), {})

    counts = np.bincount(names, minlength=len(COLOR_NAMES)).astype(float)
    dist = counts / counts.sum()
    top = int(np.argmax(dist))
    distribution = {COLOR_NAMES[i]: round(float(p), 3) for i, p in enumerate(dist) if p >= 0.05}
    color = COLOR_NAMES[top] if dist[top] >= min_share else None
    return ColorReading(color, round(float(dist[top]), 3), round(skin_fraction, 3), int(names.size), distribution)


def crop(frame: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """Crop with pixel box (x1, y1, x2, y2), clipped; None if empty."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]
