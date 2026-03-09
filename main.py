#1 SRC/OCR/Extractor - Module

import pdfplumber
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract


def is_scanned_pdf(pdf_path: str) -> bool:
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        return text is None


def extract_digital_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def _preprocess_image(pil_image):
    open_cv_image = np.array(pil_image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh


def extract_scanned_text(pdf_path: str, dpi: int = 300) -> str:
    pages = convert_from_path(pdf_path, dpi=dpi)
    full_text = ""
    for page in pages:
        processed = _preprocess_image(page)
        text = pytesseract.image_to_string(processed, lang="eng")
        full_text += text + "\n"
    return full_text


def extract_text(pdf_path: str) -> str:
    if is_scanned_pdf(pdf_path):
        return extract_scanned_text(pdf_path)
    return extract_digital_text(pdf_path)