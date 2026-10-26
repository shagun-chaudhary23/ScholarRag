import io
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
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
                logger.info("Tesseract binary not found on system. Ingestion will use native PDF text.")
    return _TESSERACT_AVAILABLE

def is_page_scanned(page: fitz.Page, min_char_threshold: int = 15) -> bool:
    """Classify if page contains native digital text or is a scanned image."""
    text = page.get_text("text").strip()
    if len(text) >= min_char_threshold:
        return False
    # If Tesseract is not installed, avoid OCR attempt to prevent blocking
    if not is_tesseract_installed():
        return False
    images = page.get_images()
    return len(images) > 0

def ocr_page(page: fitz.Page, dpi: int = 150) -> Tuple[str, float, List[Dict[str, Any]]]:
    """Perform OCR on a scanned page and compute mean confidence score.
    
    Returns:
        (extracted_text, average_confidence, bounding_boxes)
    """
    raw_text = page.get_text("text").strip()
    if not is_tesseract_installed():
        return raw_text, 0.5, []
        
    try:
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        
        # Get detailed OCR data with confidence
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
                
                # Normalize bounding box [x0/w, y0/h, x1/w, y1/h]
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
