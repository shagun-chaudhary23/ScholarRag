import io
import shutil
import base64
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import fitz  # PyMuPDF
import pytesseract
import httpx
from PIL import Image
from backend.app.config import settings
from backend.app.core.logging import logger

_TESSERACT_AVAILABLE: Optional[bool] = None

def is_tesseract_installed() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is None:
        cmd = shutil.which("tesseract")
        if cmd:
            _TESSERACT_AVAILABLE = True
        else:
            win_paths = [
                Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
                Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
            ]
            for p in win_paths:
                if p.exists():
                    pytesseract.pytesseract.tesseract_cmd = str(p)
                    _TESSERACT_AVAILABLE = True
                    break
            else:
                _TESSERACT_AVAILABLE = False
    return _TESSERACT_AVAILABLE

def is_ocr_available() -> bool:
    """Check if any OCR engine (Gemini Vision or Tesseract) is available."""
    if settings.GEMINI_API_KEY:
        return True
    return is_tesseract_installed()

def is_page_scanned(page: fitz.Page, min_char_threshold: int = 25) -> bool:
    """Classify if page contains native digital text or is a scanned/handwritten image."""
    text = page.get_text("text").strip()
    if len(text) >= min_char_threshold:
        return False
    if not is_ocr_available():
        return False
    images = page.get_images()
    return len(images) > 0 or len(text) < min_char_threshold

def _ocr_with_gemini(page: fitz.Page, dpi: int = 150) -> Optional[Tuple[str, float, List[Dict[str, Any]]]]:
    """Use Gemini Vision to transcribe printed or handwritten page content."""
    if not settings.GEMINI_API_KEY:
        return None
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img.thumbnail((1200, 1200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        model = settings.GEMINI_MODEL or "gemini-3.5-flash-lite"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Transcribe all visible text, diagrams, and handwriting from this academic study page accurately. "
                                "Preserve headings, bullet points, numbered lists, and mathematical formulas in markdown."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1
            }
        }
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text:
                    logger.info("Page %s transcribed via Gemini Vision (%d chars)", page.number + 1, len(text))
                    return text, 0.95, [{"text": text, "bbox": [0.05, 0.05, 0.95, 0.95]}]
            else:
                logger.warning("Gemini Vision OCR returned status %s: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Gemini Vision OCR error on page %s: %s", page.number + 1, e)
    return None

def ocr_page(page: fitz.Page, dpi: int = 150) -> Tuple[str, float, List[Dict[str, Any]]]:
    """Perform OCR on a scanned page and compute mean confidence score.
    
    Returns:
        (extracted_text, average_confidence, bounding_boxes)
    """
    raw_text = page.get_text("text").strip()
    
    # Priority 1: Gemini Vision for handwritten and scanned material
    if settings.GEMINI_API_KEY:
        gemini_res = _ocr_with_gemini(page, dpi=dpi)
        if gemini_res and gemini_res[0]:
            return gemini_res
            
    # Priority 2: Local Tesseract OCR
    if not is_tesseract_installed():
        return raw_text, 0.5, []
        
    try:
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        words = []
        confidences = []
        bboxes = []
        img_w, img_h = image.size
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text_word = data['text'][i].strip()
            conf = float(data['conf'][i])
            if text_word and conf > 0:
                words.append(text_word)
                confidences.append(conf / 100.0)
                
                x0 = data['left'][i] / img_w
                y0 = data['top'][i] / img_h
                x1 = (data['left'][i] + data['width'][i]) / img_w
                y1 = (data['top'][i] + data['height'][i]) / img_h
                bboxes.append({"text": text_word, "bbox": [x0, y0, x1, y1]})
                
        extracted_text = " ".join(words)
        if not extracted_text and raw_text:
            extracted_text = raw_text
        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
        return extracted_text, round(avg_confidence, 3), bboxes
    except Exception as e:
        logger.warning("OCR failed for page %s: %s", page.number + 1, str(e))
        return raw_text, 0.5, []

