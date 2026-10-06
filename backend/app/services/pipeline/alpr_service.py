import cv2
import numpy as np
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy initialization of OCR readers
_PADDLE_OCR = None
_EASY_OCR = None
_OCR_ATTEMPTED = False

def _init_ocr_engine():
    global _PADDLE_OCR, _EASY_OCR, _OCR_ATTEMPTED
    if _OCR_ATTEMPTED:
        return
    _OCR_ATTEMPTED = True
    
    try:
        from paddleocr import PaddleOCR
        logger.info("[ALPR] Initializing PaddleOCR engine...")
        _PADDLE_OCR = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        logger.info("[ALPR] PaddleOCR initialized successfully.")
        return
    except Exception as e:
        logger.debug(f"[ALPR] PaddleOCR not available ({e}). Trying EasyOCR...")

    try:
        import easyocr
        logger.info("[ALPR] Initializing EasyOCR engine...")
        _EASY_OCR = easyocr.Reader(['en'], gpu=False)
        logger.info("[ALPR] EasyOCR initialized successfully.")
    except Exception as e:
        logger.warning(f"[ALPR] OCR engines (PaddleOCR / EasyOCR) unavailable ({e}). Fallback mode active.")

class ALPRService:
    """
    Automatic License Plate Recognition (ALPR) Service.
    Extracts license plate crops, applies adaptive thresholding, runs OCR,
    and validates text against standard alphanumeric patterns.
    """

    PLATE_REGEX = re.compile(r"^[A-Z0-9\-\s]{4,12}$")

    @classmethod
    def preprocess_plate_crop(cls, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Enhances contrast and cleans plate image via adaptive thresholding.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard height for OCR stability
        h, w = gray.shape[:2]
        if h < 60:
            scale = 60.0 / float(h)
            gray = cv2.resize(gray, (int(w * scale), 60), interpolation=cv2.INTER_CUBIC)

        # Bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Adaptive thresholding to enhance high-contrast text
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary

    @classmethod
    def locate_plate_region(cls, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Locates license plate candidate region within a vehicle crop using aspect ratio & contour analysis.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        h, w = vehicle_crop.shape[:2]
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # Plate candidate heuristic focusing on lower 60% of vehicle box
        roi_y1 = int(h * 0.35)
        roi = gray[roi_y1:h, :]
        
        # Edge detection for plate localization
        grad_x = cv2.Sobel(roi, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_plate_crop = None
        max_area = 0
        
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            area = rw * rh
            
            # Typical license plate aspect ratio is between 2.0 and 6.0
            if 2.0 <= aspect <= 6.5 and area > 400 and area > max_area:
                max_area = area
                bx, by, bw, bh = cv2.boundingRect(cnt)
                best_plate_crop = vehicle_crop[roi_y1 + by : roi_y1 + by + bh, bx : bx + bw]
                
        return best_plate_crop if best_plate_crop is not None else vehicle_crop

    @classmethod
    def extract_license_plate(cls, vehicle_crop: np.ndarray) -> Optional[str]:
        """
        Runs full ALPR pipeline on a vehicle crop and returns sanitized plate string or None.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        _init_ocr_engine()

        plate_region = cls.locate_plate_region(vehicle_crop)
        if plate_region is None:
            plate_region = vehicle_crop

        binary_plate = cls.preprocess_plate_crop(plate_region)

        raw_text = ""

        # 1. Try PaddleOCR
        if _PADDLE_OCR is not None:
            try:
                res = _PADDLE_OCR.ocr(binary_plate, cls=False)
                if res and res[0]:
                    lines = [line[1][0] for line in res[0] if line and line[1]]
                    raw_text = " ".join(lines)
            except Exception as e:
                logger.debug(f"[ALPR] PaddleOCR recognition error: {e}")

        # 2. Try EasyOCR if PaddleOCR failed or unavailable
        if not raw_text and _EASY_OCR is not None:
            try:
                res = _EASY_OCR.readtext(binary_plate)
                if res:
                    raw_text = " ".join([r[1] for r in res])
            except Exception as e:
                logger.debug(f"[ALPR] EasyOCR recognition error: {e}")

        if not raw_text:
            return None

        # Sanitize and format
        clean_text = re.sub(r"[^A-Z0-9\-\s]", "", raw_text.upper()).strip()
        clean_text = re.sub(r"\s+", " ", clean_text)

        if cls.PLATE_REGEX.match(clean_text):
            logger.info(f"[ALPR] Extracted valid license plate: '{clean_text}'")
            return clean_text

        # Secondary fallback: match substrings
        matches = re.findall(r"[A-Z0-9]{4,10}", clean_text.replace(" ", "").replace("-", ""))
        if matches:
            plate_candidate = matches[0]
            logger.info(f"[ALPR] Extracted sanitized license plate candidate: '{plate_candidate}'")
            return plate_candidate

        return None
